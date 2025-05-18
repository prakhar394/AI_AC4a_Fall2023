# AI_AC4a_Fall2025

# 🎬 YouTube Watch History Analyzer

This project provides an end-to-end pipeline to extract a user's YouTube watch history directly from their browser, fetch video transcripts, run emotion analysis models, and store all processed data securely on AWS S3. It's built to integrate with a frontend Chrome Extension, enabling seamless UI-based retrieval and analysis.

---

## 🚀 Features

- 🔍 Extract YouTube watch history from Chrome, Edge, and Brave (macOS & Windows)
- 🧠 Fetch video transcripts using `youtube-transcript-api`
- 🏷️ Analyze transcripts using GoEmotions or Roberta-GoEmotions models
- 📊 Generate Excel files summarizing emotional content
- ☁️ Store and retrieve user data via AWS S3 (`yt-user-history-bucket`)
- 🔌 Built for integration with a Chrome Extension frontend
- ⚡ REST API powered by FastAPI/Flask (`main.py`)

---

## 📁 Project Structure

```bash
.
├── main.py                  # API entry point: handles routes for store/run/download
├── history_extractor.py     # Extracts YouTube video links from browser history
├── transcript_downloader.py # Downloads and parses transcripts
├── video_details.py         # Extracts video metadata
├── models.py                # Emotion classification with transformer models
├── utils.py                 # Common helper functions
├── config.py                # Environment/configuration variables
├── requirements.txt         # Project dependencies
├── yt_links.txt             # Input YouTube URLs (optional)
└── .env                     # API keys and config (not tracked in version control)
```

---

## 🔄 End-to-End Flow

1. **User provides a username** via the Chrome extension frontend.
2. `main.py` triggers the `/store` endpoint:
   - Extracts watch history from browser (`history_extractor.py`)
   - For each video:
     - Downloads transcript (`transcript_downloader.py`)
     - Extracts metadata (`video_details.py`)
   - Uploads results to:
     ```
     s3://yt-user-history-bucket/{username}/watch_history.csv
     s3://yt-user-history-bucket/{username}/transcripts/{video_id}.json
     ```
3. User can trigger `/run` to perform emotion analysis on stored videos.
4. User downloads results via `/download`.

---

## 🔧 Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/your-repo/yt-watch-analyzer.git
cd yt-watch-analyzer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Setup

Create a `.env` file:

```env
YOUTUBE_API_KEY=your_youtube_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=your_region
MODEL_NAME=bhadresh-savani/distilbert-base-uncased-emotion
API_ENDPOINT=https://your-api-endpoint.com
```

---

## 🧪 API Endpoints

| Endpoint        | Method | Description |
|----------------|--------|-------------|
| `/store`       | POST   | Extracts watch history, transcripts, and stores to S3 |
| `/history`     | GET    | Fetches stored watch history for a username |
| `/run`         | POST   | Runs selected emotion model on stored transcripts |
| `/download`    | POST   | Downloads `.xlsx` file with labeled emotions |

---

## ☁️ AWS S3 Structure

```
yt-user-history-bucket/
└── <username>/
    ├── watch_history.csv
    ├── transcripts/
    │   ├── <video_id>.json
    ├── metadata/
    │   ├── <video_id>.json
    └── models/
        └── emotions.csv
```

Use AWS CLI or boto3 to access:

```bash
aws s3 cp s3://yt-user-history-bucket/prakhar/watch_history.csv .
```

---

## 🧠 Models Supported

- 🤖 [`distilbert-base-uncased-emotion`](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion)
- 🤗 [`GoEmotions`](https://huggingface.co/monologg/bert-base-cased-goemotions-original)

Can be extended to:
- Sentiment analysis
- Toxicity detection
- Video summarization

---

## 🛡️ Privacy & Permissions

This app **never uploads video content**—only URLs, transcripts, and metadata. User watch history is stored per-username and can be deleted or encrypted as needed.

---

## 📜 License

MIT License © 2025 Peace Speech Project Team

---

## ✨ Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/)
- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api)
- [FastAPI](https://fastapi.tiangolo.com/)
- [AWS S3](https://aws.amazon.com/s3/)
