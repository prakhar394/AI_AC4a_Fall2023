import os
import logging
from urllib.parse import urlparse, parse_qs

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def clean_url(url):
    """Clean and validate a YouTube URL."""
    url = url.strip()
    if not url.startswith("http"):
        return None
    return url.replace("[", "").replace("]", "")

def get_video_id(url):
    """Extracts YouTube video ID from the URL."""
    try:
        if "youtube.com" in url:
            parsed_url = urlparse(url)
            return parse_qs(parsed_url.query).get("v", [None])[0]
        elif "youtu.be" in url:
            parsed_url = urlparse(url)
            return parsed_url.path.lstrip("/")
    except Exception as e:
        logging.error(f"Error extracting video ID from {url}: {e}")
        return None

def create_directory(path):
    """Creates a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
