# app/main.py
import os
import json
import shutil
import tempfile
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

# Import your processing function
from app.dance import process_video

app = FastAPI(title="Dance Movement Analysis API", version="1.0")

# Directory to store temporary files
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)


# ------------------------------------------------------
# 1️⃣ POST /analyze - Process uploaded video
# ------------------------------------------------------
@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    trails: bool = Form(True),
):
    try:
        # Save uploaded file to temp dir
        input_path = os.path.join(TEMP_DIR, file.filename)
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Define output paths
        output_video = os.path.join(TEMP_DIR, "processed_" + file.filename)
        output_json = os.path.join(
            TEMP_DIR, "processed_" + Path(file.filename).stem + ".json"
        )

        # Process video (run in threadpool to prevent blocking)
        await run_in_threadpool(
            process_video,
            input_path,
            output_video,
            output_json,
            trails,
            10,  # trail_len
            0.5,  # min_detection_confidence
            0.5,  # min_tracking_confidence
        )

        # Read keypoints JSON
        with open(output_json, "r") as jf:
            keypoints_data = json.load(jf)

        # Return response JSON
        return JSONResponse(
            status_code=200,
            content={
                "message": "Video processed successfully",
                "temp_video_file": os.path.basename(output_video),
                "temp_json_file": os.path.basename(output_json),
                "keypoints_count_frames": len(keypoints_data),
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------
# 2️⃣ GET /download/{filename} - Download processed files
# ------------------------------------------------------
@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Detect MIME type dynamically (for video or JSON)
    mime_type = (
        "video/mp4" if filename.endswith(".mp4") else "application/json"
    )
    return FileResponse(file_path, media_type=mime_type, filename=filename)
