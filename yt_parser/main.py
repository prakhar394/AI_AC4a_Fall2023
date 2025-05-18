import os
import sys
import csv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
from typing import List, Dict, Optional
from config import YT_LINKS_FILE, OUTPUT_DIR
from utils import clean_url, create_directory
from transcript_downloader import download_transcript, get_transcript
from video_details import get_video_details
from history_extractor import extract_history, process_video, read_history_file  # UPDATED
from models import run_go_emotions  # Import the function from models.py
import boto3
import io
from io import StringIO

app = FastAPI()

# Initialize S3 client
s3 = boto3.client('s3')
BUCKET_NAME = "yt-user-history-bucket"

class UsernameRequest(BaseModel):
    username: str

@app.post("/users/history")
async def store_user_history(request: UsernameRequest):
    try:
        username = request.username
        
        # Step 1: Extract basic browser history
        df = extract_history()

        # Step 2: Enrich with transcripts, seconds watched, category
        category_dir = os.path.join(OUTPUT_DIR, "history")
        create_directory(category_dir)

        enriched_rows = []
        for _, row in df.iterrows():
            video_details = process_video(row["video_url"], category_dir)
            enriched_rows.append({
                "browser": row["browser"],
                "date_watched": row["date_watched"],
                "video_title": row["video_title"],
                "url": row["video_url"],
                "transcript": video_details["transcript"],
                "seconds_watched": video_details["seconds_watched"],
                "category_watched": video_details["category_watched"]
            })

        enriched_df = pd.DataFrame(enriched_rows)

        # Step 3: Save to CSV buffer
        csv_buffer = StringIO()
        enriched_df.to_csv(csv_buffer, index=False)

        # Step 4: Upload enriched CSV to S3
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=f"{username}/history.csv",
            Body=csv_buffer.getvalue()
        )

        return {"status": "success", "message": "Enriched history stored in S3"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class VideoData(BaseModel):
    browser: str
    date_watched: str
    video_title: str
    url: str
    transcript: Optional[str]
    seconds_watched: int
    category_watched: str

@app.get("/users/{username}/history", response_model=List[VideoData])
async def get_user_history(username: str):
    try:
        # Increase field size limit
        csv.field_size_limit(sys.maxsize)

        # Step 1: Fetch enriched CSV from S3
        response = s3.get_object(
            Bucket=BUCKET_NAME,
            Key=f"{username}/history.csv"
        )
        csv_content = response['Body'].read().decode('utf-8')

        # Step 2: Parse CSV
        videos = []
        reader = csv.DictReader(csv_content.splitlines())
        for row in reader:
            videos.append(VideoData(
                browser=row["browser"],
                date_watched=row["date_watched"],
                video_title=row["video_title"],
                url=row["url"],
                transcript=row["transcript"],
                seconds_watched=int(row["seconds_watched"]),
                category_watched=row["category_watched"]
            ))

        return videos
    except s3.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="User history not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to test go-roberta model
@app.get("/run_models")
def run_models(model_name: str, transcript: str):
    try:
        if model_name == "go_emotions" or model_name == "roberta_go_emotions":
            result = run_go_emotions(transcript, model_name)
            return result
        else:
            raise HTTPException(status_code=400, detail="Model not recognized.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Excel download endpoint
@app.get("/download_excel")
def download_excel(model_name: str, transcript: str):
    try:
        if model_name == "go_emotions" or model_name == "roberta_go_emotions":
            _, df = run_go_emotions(transcript, model_name)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name="Emotions")
            output.seek(0)

            headers = {
                'Content-Disposition': 'attachment; filename="emotions_output.xlsx"'
            }

            return StreamingResponse(output, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', headers=headers)
        else:
            raise HTTPException(status_code=400, detail="Model not recognized.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
