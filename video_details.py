from googleapiclient.discovery import build
import re
import logging
from pytube import YouTube
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def is_youtube_url(url):
    if not isinstance(url, str):
        return False
    pattern = re.compile(r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/.*')
    match = pattern.match(url)
    return bool(match)

def getUrlDetails(url, youtube):
    if is_youtube_url(url):
        yt = YouTube(url)
        video_id = yt.video_id
        request = youtube.videos().list(
            part="snippet,contentDetails",
            id=video_id
        )
        response = request.execute()

        category_id = response['items'][0]['snippet']['categoryId']
        duration = response['items'][0]['contentDetails']['duration']

        match = re.match(r'PT(\d+H)?(\d+M)?(\d+S)?', duration)

        hours = int(match.group(1)[:-1] or 0) if match.group(1) else 0
        minutes = int(match.group(2)[:-1] or 0) if match.group(2) else 0
        seconds = int(match.group(3)[:-1] or 0) if match.group(3) else 0

        total_seconds = hours * 3600 + minutes * 60 + seconds

        category_tags_map = config.CATEGORY_TAGS_MAP

        # check cateogry list https://mixedanalytics.com/blog/list-of-youtube-video-category-ids/ to filter for categories, can update number of minutes of videos
        # fetch category_name from category map by category_id fetched via API
        category_name = category_tags_map[category_id]
        return total_seconds, category_name
    else:
        logging.error(f"Invalid Youtube URL to fetch data: {url}")
        return
    

def get_video_details(url):

    #do we have an API key to use? - if not follow the documentation in git repo to get it
    youtube = build('youtube', 'v3', developerKey='AIzaSyBiv6q-m36NUA5LG5DxaLRKVkmdyiPU01M')

    seconds_watched, category_watched = getUrlDetails(url,youtube)

    return seconds_watched, category_watched

