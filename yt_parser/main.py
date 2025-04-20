import os
import csv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from config import YT_LINKS_FILE, OUTPUT_DIR
from utils import clean_url, create_directory
from transcript_downloader import download_transcript, get_transcript
from video_details import get_video_details
from history_extractor import extract_history
from models import run_go_emotions  # Import the function from models.py
import boto3
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
        # Extract history data
        df = extract_history()
        # Convert DataFrame to CSV
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        # Upload to S3
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=f"{username}/history.csv",
            Body=csv_buffer.getvalue()
        )
        return {"status": "success", "message": "History stored in S3"}
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

def process_video(url: str, category_dir: str) -> Dict:
    """Processes a single video URL and returns its details."""
    transcript_available = download_transcript(url, category_dir)
    if transcript_available:
        transcript = get_transcript(url, category_dir)
        seconds_watched, category_watched = get_video_details(url)
        return {
            "transcript": transcript,
            "seconds_watched": seconds_watched,
            "category_watched": category_watched
        }
    else:
        return {
            "transcript": "NA",
            "seconds_watched": 0,
            "category_watched": "NA"
        }

def read_history_file(history_file: str) -> List[Dict]:
    """Reads the history CSV file and returns a list of video data."""
    videos = []
    with open(history_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) < 4:
                continue  # Skip invalid rows
            browser, date_watched, video_title, url = row[:4]
            videos.append({
                "browser": browser,
                "date_watched": date_watched,
                "video_title": video_title,
                "url": url
            })
    return videos

@app.get("/users/{username}/history", response_model=List[VideoData])
async def get_user_history(username: str):
    try:
        # Fetch CSV from S3
        response = s3.get_object(
            Bucket=BUCKET_NAME,
            Key=f"{username}/history.csv"
        )
        csv_content = response['Body'].read().decode('utf-8')
        
        # Parse CSV
        videos = []
        reader = csv.reader(csv_content.splitlines())
        next(reader)  # Skip header
        for row in reader:
            if len(row) < 4:
                continue
            browser, date_watched, video_title, url = row[:4]
            videos.append({
                "browser": browser,
                "date_watched": date_watched,
                "video_title": video_title,
                "url": url
            })
        
        # Process videos (existing code)
        category_dir = os.path.join(OUTPUT_DIR, "history")
        create_directory(category_dir)
        
        enriched_videos = []
        for video in videos:
            video_details = process_video(video["url"], category_dir)
            enriched_videos.append(VideoData(
                browser=video["browser"],
                date_watched=video["date_watched"],
                video_title=video["video_title"],
                url=video["url"],
                transcript=video_details["transcript"],
                seconds_watched=video_details["seconds_watched"],
                category_watched=video_details["category_watched"]
            ))
        
        return enriched_videos
    except s3.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="User history not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/run_models")
def run_models(model_name: str, transcript: str):
    """API endpoint to run selected model on the given transcript."""
    try:
        if model_name == "go_emotions" or model_name == "roberta_go_emotions":
            # Run the GoEmotions model using the transcript provided by the user
            result = run_go_emotions(transcript, model_name)
            return result
        else:
            raise HTTPException(status_code=400, detail="Model not recognized.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
