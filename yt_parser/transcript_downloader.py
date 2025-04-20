import os
import logging
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from utils import get_video_id, create_directory

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def download_transcript(video_url, output_folder):
    """Downloads and saves the transcript of a YouTube video."""
    try:
        logging.info(f"Fetching transcript for: {video_url}")

        video_id = get_video_id(video_url)
        if not video_id:
            logging.error(f"Could not extract video ID from: {video_url}")
            return False

        # Fetch transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # Format transcript
        formatter = TextFormatter()
        formatted_transcript = formatter.format_transcript(transcript)

        # Ensure output folder exists
        create_directory(output_folder)

        # Save transcript
        output_file = os.path.join(output_folder, f"{video_id}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(formatted_transcript)

        logging.info(f"Transcript saved: {output_file}")
        return True

    except Exception as e:
        logging.error(f"Error downloading transcript for {video_url}: {e}")
        return False

def get_transcript(url, output_folder):

    video_id = get_video_id(url)
    output_file = os.path.join(output_folder, f"{video_id}.txt")

    with open(output_file, "r", encoding="utf-8") as file:
        file_contents = file.read()
    
    return file_contents