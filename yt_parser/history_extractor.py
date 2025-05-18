import sqlite3
import os
import logging
import time
import shutil
import platform
from datetime import datetime, timedelta
import pandas as pd
import config
import requests
import csv
from typing import List, Dict
from transcript_downloader import download_transcript, get_transcript
from video_details import get_video_details

logging.basicConfig(level=logging.INFO)

API_ENDPOINT = config.API_ENDPOINT

def get_history_path(browser):
    system = platform.system().lower()
    if system == 'darwin':  # macOS
        paths = {
            "chrome": os.path.expanduser("~/Library/Application Support/Google/Chrome/Default/History"),
            "edge": os.path.expanduser("~/Library/Application Support/Microsoft Edge/Default/History"),
            "brave": os.path.expanduser("~/Library/Application Support/BraveSoftware/Brave-Browser/Default/History"),
            "safari": os.path.expanduser("~/Library/Safari/History.db")
        }
    else:  # Windows
        paths = {
            "chrome": os.path.join(os.environ['LOCALAPPDATA'], 'Google', 'Chrome', 'User Data', 'Default', 'History'),
            "edge": os.path.join(os.environ['LOCALAPPDATA'], 'Microsoft', 'Edge', 'User Data', 'Default', 'History'),
            "brave": os.path.join(os.environ['LOCALAPPDATA'], 'BraveSoftware', 'Brave-Browser', 'User Data', 'Default', 'History'),
            "safari": None
        }
    return paths.get(browser)

def copy_with_retry(src_path: str, retries=5, delay=1):
    for attempt in range(retries):
        try:
            temp_db = f"temp_history_{attempt}.db"
            shutil.copy2(src_path, temp_db)
            return temp_db
        except Exception as e:
            logging.warning(f"Attempt {attempt+1} failed: {e}")
            time.sleep(delay)
    raise RuntimeError("Copy failed after retries")

def read_history(browser="chrome"):
    temp_db = None
    conn = None
    record_links = []
    try:
        history_path = get_history_path(browser)
        if history_path is None and browser == "safari":
            logging.info("Safari is only available on macOS.")
            return
        if not os.path.exists(history_path):
            logging.error(f"History path does not exist for {browser}: {history_path}")
            return
        
        temp_db = copy_with_retry(history_path)
        
        with sqlite3.connect(f"file:{temp_db}?mode=ro", uri=True) as conn:
            cursor = conn.cursor()

            if browser in ["chrome", "brave", "edge"]:
                query = """
                    SELECT DATE(datetime(last_visit_time/1000000-11644473600, 'unixepoch')) AS visit_date,
                           title, url 
                    FROM urls 
                    WHERE url LIKE '%youtube.com/watch%'
                    ORDER BY visit_date DESC;
                """
            elif browser == "safari":
                query = """
                    SELECT datetime(visit_time/1000000000, 'unixepoch') AS visit_date,
                           title, url 
                    FROM history_items 
                    WHERE url LIKE '%youtube.com/watch%'
                    ORDER BY visit_date DESC;
                """

            cursor.execute(query)
            results = cursor.fetchall()
            logging.info(f"Found {len(results)} YouTube entries in {browser}.")
            
            column_names = [description[0] for description in cursor.description]
            logging.info(f"Columns: {column_names}")
            for row in results:
                record_links.append(row)

    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        if conn:
            conn.close()
        if temp_db and os.path.exists(temp_db):
            for _ in range(5):
                try:
                    os.remove(temp_db)
                    break
                except PermissionError:
                    time.sleep(0.5)
    return record_links

def extract_history():
    browsers = ['chrome', 'edge', 'brave']
    all_history = {}
    for browser in browsers:
        history_browser = read_history(browser)
        all_history[browser] = history_browser

    df = pd.DataFrame(columns=["browser", "date_watched", "video_title", "video_url"])
    for browser in browsers:
        if all_history[browser]:
            for row in all_history[browser]:
                df = pd.concat([pd.DataFrame([[browser, row[0], row[1], row[2]]], columns=df.columns), df], ignore_index=True)

    df_sorted = df.sort_values(by='date_watched', ascending=False)
    return df_sorted

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
        next(reader)
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
    return videos

if __name__ == "__main__":
    extract_history()