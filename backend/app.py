import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

APP = FastAPI(title="Movie Recut AI")
APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = Path(tempfile.gettempdir()) / "movie_recut_ai"
BASE.mkdir(parents=True, exist_ok=True)

MAX_SECONDS = 15 * 60
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB hard upload guard

VOICE_PRESETS = {
    "coral": "နူးညံ့ပြီး သဘာဝကျသော မြန်မာအမျိုးသမီးအသံပုံစံ",
    "marin": "နူးညံ့ပြီး ငြိမ်သက်သော မြန်မာအသံပုံစံ",
    "shimmer": "ပျော့ပျောင်းပြီး ကြည်လင်သော မြန်မာအသံပုံစံ",
    "nova": "နွေးထွေးပြီး သဘာဝကျသော မြန်မာအသံပုံစံ",
    "sage": "အေးဆေးပြီး တည်ငြိမ်သော မြန်မာအသံပုံစံ",
}

def run(cmd: list[str], check=True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=check)

def ffprobe_duration(path: Path) -> float:
    p = run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ])
    try:
        return float(p.stdout.strip())
    except Exception:
        raise HTTPException(400, "Video duration could not be read. Make sure ffprobe is installed.")

def extract_audio(video: Path, audio: Path):
    run([
        "ffmpeg", "-y", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(audio)
    ])

async def openai_transcribe(api_key: str, audio_path: Path) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": "whisper-1",
        "response_format": "verbose_json",
        "timestamp_granularities[]": "segment",
        "temperature": "0",
    }
    async with httpx.AsyncClient(timeout=1800) as client:
        with audio_path.open("rb") as f:
            r = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers=headers,
                data=data,
                files={"file": (audio_path.name, f, "audio/wav")},
            )
    if r.status_code >= 400:
        raise HTTPException(r.status_code, f"Transcription API error: {r.text[:1000]}")
    return r.json()

def response_output_text(obj: dict[str, Any]) -> str:
    # Responses API commonly exposes output_text in SDKs; raw HTTP returns output items.
    if isinstance(obj.get("output_text"), str):
        return obj["output_text"]
    chunks = []
    for item in obj.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                chunks.append(c.get("text", ""))
    return "".join(chunks)

async def translate_segments(api_key: str, segments: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    compact = [
        {
            "id": i,
            "start": round(float(s.get("start", 0)), 3),
            "end": round(float(s.get("end", 0)), 3),
            "text": str(s.get("text", "")).strip(),
        }
        for i, s in enumerate(segments)
        if str(s.get("text", "")).strip()
    ]

    schema = {
        "type": "json_schema",
        "name": "burmese_segments",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "text_my": {"type": "string"},
                        },
                        "required": ["id", "text_my"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["segments"],
            "additionalProperties": False,
        },
    }

    prompt = """Translate the supplied movie dialogue into natural Myanmar Burmese.
Rules:
1. Preserve the meaning; do not add or remove story information.
2. Use correct modern Myanmar spelling and punctuation.
3. Make it sound like spoken Burmese, not a literal dictionary translation.
4. Keep names, numbers, places and technical terms sensible.
5. Keep each Burmese line concise enough to be spoken naturally within the original dialogue timing.
6. Do not split a sentence into unnatural word-by-word fragments; preserve normal Burmese phrasing and flow.
7. Return exactly one translated item for each input id.
Input segments:
""" + json.dumps(compact, ensure_ascii=False)

    payload = {
        "model": model,
        "store": False,
        "input": prompt,
        "text": {"format": schema},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=1800) as client:
        r = await client.post("https://api.openai.com/v1/responses", headers=headers, json=payload)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, f"Translation API error: {r.text[:1200]}")

    raw = response_output_text(r.json())
    try:
        data = json.loads(raw)
    except Exception:
        raise HTTPException(502, "Translation returned invalid JSON.")

    by_id = {int(x["id"]): x["text_my"].strip() for x in data.get("segments", [])}
    out = []
    for i, s in enumerate(compact):
        out.append({
            "id": i,
            "start": s["start"],
            "end": s["end"],
            "text": s["text"],
            "text_my": by_id.get(i, s["text"]),
        })
    return out

async def tts(api_key: str, text: str, voice: str, out_path: Path, speed: float):
    payload = {
        "model": "gpt-4o-mini-tts",
        "input": text[:4096],
        "voice": voice,
        "instructions": VOICE_PRESETS.get(voice, VOICE_PRESETS["coral"]) +
                        ". Speak Burmese clearly, gently, naturally, with conversational emotion. Do not sound robotic.",
        "response_format": "wav",
        "speed": max(0.25, min(4.0, speed)),
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=600) as client:
        r = await client.post("https://api.openai.com/v1/audio/speech", headers=headers, json=payload)
    if r.status_code >= 400:
        raise HTTPException(r.status_code, f"TTS API error: {r.text[:1000]}")
    out_path.write_bytes(r.content)

def fit_audio_to_duration(src: Path, dst: Path, duration: float) -> float:
    """Prepare a TTS clip without cutting speech.

    Natural TTS is preferred. If it is longer than the available slot,
    speed it up only up to 1.15x. The function NEVER trims the tail.
    Returns the actual output duration so the scheduler can prevent overlap.
    """
    p = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(src)
    ])
    try:
        src_dur = max(float(p.stdout.strip()), 0.05)
    except Exception:
        src_dur = max(float(duration), 0.05)

    target = max(float(duration), 0.08)
    speed_factor = 1.0
    if src_dur > target:
        speed_factor = min(src_dur / target, 1.15)

    run([
        "ffmpeg", "-y", "-i", str(src),
        "-filter:a", f"atempo={speed_factor:.6f}",
        "-ar", "48000", "-ac", "2", str(dst)
    ])

    p2 = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(dst)
    ])
    try:
        return max(float(p2.stdout.strip()), 0.05)
    except Exception:
        return src_dur / speed_factor

def make_srt(segments: list[dict[str, Any]], path: Path):
    def ts(sec):
        ms = int(round(sec * 1000))
        h = ms // 3600000
        ms %= 3600000
        m = ms // 60000
        ms %= 60000
        s = ms // 1000
        ms %= 1000
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines = []
    for i, s in enumerate(segments, 1):
        lines += [str(i), f"{ts(s['start'])} --> {ts(s['end'])}", s["text_my"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def mux_video(video: Path, audio: Path, out: Path):
    # Map ONLY video from the source and ONLY the generated voice track.
    # No source audio stream is mapped, so the original dialogue cannot leak
    # into the final MP4.
    run([
        "ffmpeg", "-y",
        "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-map_metadata", "-1",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-t", ffprobe_duration(video).__str__(),
        str(out)
    ])

@APP.post("/api/analyze")
async def analyze(
    video: UploadFile = File(...),
    api_key: str = Form(...),
):
    if not api_key.strip():
        raise HTTPException(400, "API key is required.")

    job = BASE / uuid.uuid4().hex
    job.mkdir(parents=True, exist_ok=True)
    video_path = job / (Path(video.filename or "video.mp4").name)

    size = 0
    with video_path.open("wb") as f:
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "File is too large.")
            f.write(chunk)

    duration = ffprobe_duration(video_path)
    if duration > MAX_SECONDS + 0.5:
        shutil.rmtree(job, ignore_errors=True)
        raise HTTPException(400, "Video must be 15 minutes or shorter.")

    audio = job / "audio.wav"
    extract_audio(video_path, audio)
    tr = await openai_transcribe(api_key.strip(), audio)

    segments = tr.get("segments", [])
    if not segments:
        # Fallback when a provider returns text without segments.
        text = tr.get("text", "").strip()
        segments = [{"start": 0, "end": duration, "text": text}] if text else []

    return {
        "job_id": job.name,
        "duration": duration,
        "segments": [
            {
                "id": i,
                "start": float(s.get("start", 0)),
                "end": float(s.get("end", 0)),
                "text": str(s.get("text", "")).strip(),
            }
            for i, s in enumerate(segments)
            if str(s.get("text", "")).strip()
        ],
    }

@APP.post("/api/translate")
async def translate(
    job_id: str = Form(...),
    api_key: str = Form(...),
    model: str = Form("gpt-5.6-luna"),
    segments_json: str = Form(...),
):
    job = BASE / job_id
    if not job.exists():
        raise HTTPException(404, "Job not found.")
    try:
        segments = json.loads(segments_json)
    except Exception:
        raise HTTPException(400, "Invalid segments JSON.")
    result = await translate_segments(api_key.strip(), segments, model.strip())
    (job / "segments.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"segments": result}

@APP.post("/api/export")
async def export(
    job_id: str = Form(...),
    api_key: str = Form(...),
    voice: str = Form("coral"),
    speed: float = Form(1.0),
    segments_json: str = Form(...),
):
    job = BASE / job_id
    if not job.exists():
        raise HTTPException(404, "Job not found.")
    if voice not in VOICE_PRESETS:
        raise HTTPException(400, "Unknown voice.")

    video_candidates = [p for p in job.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}]
    if not video_candidates:
        raise HTTPException(404, "Original video not found.")
    video = video_candidates[0]

    try:
        segments = json.loads(segments_json)
    except Exception:
        raise HTTPException(400, "Invalid segments JSON.")

    # Generate speech clips and schedule them using their ACTUAL durations.
    # This is the key anti-overlap rule: a later clip can never start while
    # an earlier clip is still speaking. We never trim a sentence to force it
    # into a slot. This produces one smooth generated-voice track.
    fitted = []
    cursor = 0.0
    for idx, seg in enumerate(segments):
        text = str(seg.get("text_my", "")).strip()
        if not text:
            continue

        original_start = max(0.0, float(seg.get("start", 0)))
        original_end = max(original_start + 0.08, float(seg.get("end", original_start + 0.08)))
        slot = max(0.08, original_end - original_start)

        raw = job / f"tts_raw_{idx}.wav"
        fit = job / f"tts_fit_{idx}.wav"
        await tts(api_key.strip(), text, voice, raw, speed)
        actual_dur = fit_audio_to_duration(raw, fit, slot)

        # Keep the original timing whenever possible. If a previous sentence
        # is still speaking, gently move this one forward instead of allowing
        # two voices to overlap.
        start_time = max(original_start, cursor)
        fitted.append((idx, fit, start_time, actual_dur))
        cursor = start_time + actual_dur + 0.02

    # Build a SILENT base track, then place ONLY the generated TTS clips on it.
    # The original video's audio stream is never read into this mix.
    duration = ffprobe_duration(video)
    base = job / "base.wav"
    track_duration = max(duration, min(cursor + 0.05, duration))
    run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "anullsrc=r=48000:cl=stereo",
        "-t", f"{track_duration:.3f}", "-c:a", "pcm_s16le", str(base)
    ])

    input_args = ["ffmpeg", "-y", "-i", str(base)]
    labels = []
    filter_parts = ["[0:a]anull[base]"]
    for n, (_idx, fit, start_time, _actual_dur) in enumerate(fitted, start=1):
        input_args += ["-i", str(fit)]
        delay = max(0, int(round(start_time * 1000)))
        filter_parts.append(f"[{n}:a]adelay={delay}:all=1[d{n}]")
        labels.append(f"[d{n}]")

    if labels:
        filter_parts.append(
            "[base]" + "".join(labels) +
            f"amix=inputs={len(labels)+1}:duration=first:dropout_transition=0:normalize=0[final]"
        )
        voice_track = job / "voice_track.wav"
        run(input_args + [
            "-filter_complex", ";".join(filter_parts),
            "-map", "[final]", "-ar", "48000", "-ac", "2", str(voice_track)
        ])
    else:
        voice_track = base

    out = job / "recut_myanmar.mp4"
    mux_video(video, voice_track, out)

    srt = job / "myanmar.srt"
    make_srt(segments, srt)

    return {
        "video_url": f"/api/file/{job.name}/recut_myanmar.mp4",
        "srt_url": f"/api/file/{job.name}/myanmar.srt",
    }

@APP.get("/api/file/{job_id}/{filename}")
def get_file(job_id: str, filename: str):
    job = BASE / job_id
    safe = Path(filename).name
    path = job / safe
    if not path.exists():
        raise HTTPException(404, "File not found.")
    media = "video/mp4" if path.suffix == ".mp4" else "application/x-subrip"
    return FileResponse(path, media_type=media, filename=safe)

@APP.get("/health")
def health():
    return {"name": "Movie Recut AI", "status": "ok"}

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    APP.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

