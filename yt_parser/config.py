import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Input file with YouTube links
YT_LINKS_FILE = os.getenv("YT_LINKS_FILE", "yt_links.txt")

# Output directory for transcripts
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "transcripts/")

# Log file location
LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")