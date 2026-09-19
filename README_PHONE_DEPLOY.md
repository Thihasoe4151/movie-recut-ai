# Movie Recut AI — Phone Web App Deployment

This version serves the frontend and FastAPI backend from one public URL. It includes FFmpeg in Docker, so no Windows terminal or local FFmpeg install is needed after deployment.

## Render deployment
1. Put this project in a GitHub repository.
2. In Render, choose New → Web Service and connect the repository.
3. Choose Docker runtime. Render can build from the included Dockerfile.
4. Deploy. The public URL will be an `onrender.com` address.
5. Open that URL on an Android/iPhone browser.

The frontend uses `window.location.origin`, so it talks to the same deployed backend automatically.

## Important
- The app currently asks the user for an OpenAI API key in the browser and sends it to the backend for each request; it does not save it in the app.
- Do not publish a personal OpenAI API key inside source code.
- Video files are processed in temporary server storage. For a production service, add persistent/object storage and background jobs.
- Fifteen-minute video processing can require more than a free/small server can comfortably handle; increase the server plan if jobs fail due to memory/CPU/time limits.


### Single-voice timing update
This build explicitly excludes the source video's audio tracks and prevents generated TTS clips from overlapping, so only one generated Burmese voice is heard at a time.
