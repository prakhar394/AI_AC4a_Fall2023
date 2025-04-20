import os

# Input file with YouTube links
YT_LINKS_FILE = os.getenv("YT_LINKS_FILE", "yt_links.txt")

# Output directory for transcripts
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "transcripts/")

# Log file location
LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")

# Map of the category id of a yt video to the category it belongs to
CATEGORY_TAGS_MAP = {
            "1"	: "Film & Animation",
            "2"	: "Autos & Vehicles",
            "10" : "Music",
            "15" : "Pets & Animals",
            "17" : "Sports",
            "18" : "Short Movies",
            "19" : "Travel & Events",
            "20" : "Gaming",
            "21" : "Videoblogging",
            "22" : "People & Blogs",
            "23" : "Comedy",
            "24" : "Entertainment",
            "25" : "News & Politics",
            "26" : "Howto & Style",
            "27" : "Education",
            "28" : "Science & Technology",
            "29" : "Nonprofits & Activism",
            "30" : "Movies",
            "31" : "Anime/Animation",
            "32" : "Action/Adventure",
            "33" : "Classics",
            "34" : "Comedy",
            "35" : "Documentary",
            "36" : "Drama",
            "37" : "Family",
            "38" : "Foreign",
            "39" : "Horror",
            "40" : "Sci-Fi/Fantasy",
            "41" : "Thriller",
            "42" : "Shorts",
            "43" : "Shows",
            "44" : "Trailers"
}