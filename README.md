# Movie Recut AI

A runnable MVP for:
- Uploading one video up to 15 minutes
- Entering an OpenAI API key in the UI
- Extracting speech with timestamps
- Translating the speech to natural Burmese
- Correcting Burmese spelling/wording
- Generating Burmese TTS with a soft voice
- Replacing the original audio with generated voice
- Exporting MP4 + SRT

## Requirements

- Python 3.11+
- FFmpeg + ffprobe installed and available on PATH
- An OpenAI API key

## Run

### 1) Backend

```bash
cd backend
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

### 2) Frontend

The frontend is plain HTML/JS, so you can open `frontend/index.html` directly.
For best browser behavior, run:

```bash
cd frontend
python -m http.server 5500
```

Then open http://localhost:5500

## FFmpeg

Ubuntu/Debian:
```bash
sudo apt update
sudo apt install ffmpeg
```

macOS:
```bash
brew install ffmpeg
```

Windows:
Install FFmpeg and add its `bin` directory to PATH.

## Notes

- The API key is sent to the backend only for the current request and is not written to disk.
- This MVP keeps the original video and replaces its audio track.
- Speech segments are time-fitted with FFmpeg `atempo` so generated speech is closer to the original timing.
- For long videos, TTS is generated per speech segment, so API usage can increase.
- Voice cloning is intentionally not included. The UI uses selectable built-in TTS voices.
