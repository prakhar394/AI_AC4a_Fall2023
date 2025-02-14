import logging
import os
from config import YT_LINKS_FILE, OUTPUT_DIR
from utils import clean_url, create_directory
from transcript_downloader import download_transcript

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def read_youtube_links(file_path):
    """Reads YouTube links from a file and categorizes them."""
    categories = {}
    current_category = None

    if not os.path.exists(file_path):
        logging.error(f"File {file_path} not found.")
        return categories

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "youtube.com" not in line.lower() and "youtu.be" not in line.lower():
                current_category = line
                categories[current_category] = []
            else:
                if current_category:
                    clean_link = clean_url(line)
                    if clean_link:
                        categories[current_category].append(clean_link)

    return categories

def main():
    """Main function to process YouTube transcripts."""
    print("Starting transcript downloader.")

    categories = read_youtube_links(YT_LINKS_FILE)
    if not categories:
        logging.error("No valid categories or YouTube links found.")
        return

    for category, urls in categories.items():
        category_dir = os.path.join(OUTPUT_DIR, category.lower().replace(" ", "_"))
        create_directory(category_dir)

        logging.info(f"Processing category: {category}")
        for url in urls:
            logging.info(f"Processing URL: {url}")
            download_transcript(url, category_dir)

if __name__ == "__main__":
    main()
