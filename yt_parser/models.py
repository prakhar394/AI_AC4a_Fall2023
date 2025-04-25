import nltk
# nltk.data.path.append("/home/ubuntu/nltk_data")  # Add your preferred path
# nltk.download('punkt', download_dir="/home/ubuntu/nltk_data")
from collections import defaultdict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import json
import pandas as pd

def split_into_sentences(paragraph):
    """Splits a paragraph into sentences."""
    return nltk.sent_tokenize(paragraph)

def load_model():
    """Loads the GoEmotions model and tokenizer."""
    model_name = "goemotions"
    token = "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"
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, token=token)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        print("Model and tokenizer loaded successfully!")
        return model, tokenizer
    except OSError as e:
        print(f"Error loading model: {e}")
        return None, None

def classify_emotions(transcript, model_type, model):
    """Processes the transcript and returns emotions as JSON."""
    transcript = transcript.replace("\n", " ").strip()
    sentences = split_into_sentences(transcript)
    
    if model_type == "roberta_go_emotions":
        classifier = pipeline(task="text-classification", 
                             model="SamLowe/roberta-base-go_emotions", 
                             top_k=None)
    elif model_type == "go_emotions":
        classifier = pipeline(task="text-classification", model=model, top_k=None)
    elif model_type == "siebert":
        classifier = pipeline(task="sentiment-analysis",
                             model="siebert/sentiment-roberta-large-english", top_k=None)
    else:
        raise ValueError("Invalid model_type. Use 'roberta_go_emotions' or 'go_emotions'")
    
    model_outputs = []

    for sentence in sentences:
        try:
            output = classifier(sentence[:500])  # Truncate sentence
            model_outputs.append(output)
        except RuntimeError as e:
            print(f"Skipping sentence due to error: {e}")
    
    # Initialize counters
    emotion_totals = defaultdict(float)
    emotion_counts = defaultdict(int)
    
    for sentence_output in model_outputs:
        for emotion in sentence_output[0]:
            label = emotion['label']
            score = emotion['score']
            emotion_totals[label] += score
            emotion_counts[label] += 1

    # Compute and sort average scores
    average_scores_unsorted = {
        label: (emotion_totals[label] / emotion_counts[label]) if emotion_counts[label] > 0 else 0
        for label in emotion_totals
    }
    average_scores = dict(sorted(average_scores_unsorted.items(), key=lambda x: x[1], reverse=True))

    dominant_emotion = max(average_scores, key=average_scores.get, default=None)
    dominant_emotion_score = average_scores.get(dominant_emotion, 0)

    # Respect and contempt categories
    respect_emotions = {"admiration", "approval", "caring"}
    contempt_emotions = {"annoyance", "disapproval", "disgust"}

    # Extract and sort scores for these emotions
    respect_scores_unsorted = {label: average_scores[label] for label in respect_emotions if label in average_scores}
    contempt_scores_unsorted = {label: average_scores[label] for label in contempt_emotions if label in average_scores}

    respect_scores = dict(sorted(respect_scores_unsorted.items(), key=lambda x: x[1], reverse=True))
    contempt_scores = dict(sorted(contempt_scores_unsorted.items(), key=lambda x: x[1], reverse=True))

    # Dominant label across both
    combined_attitudes = {**respect_scores, **contempt_scores}
    if combined_attitudes:
        dominant_attitude_emotion = max(combined_attitudes, key=combined_attitudes.get)
        dominant_attitude_score = combined_attitudes[dominant_attitude_emotion]
    else:
        dominant_attitude_emotion = None
        dominant_attitude_score = 0

    respect_contempt_json = {
        "respect_emotions": respect_scores,
        "contempt_emotions": contempt_scores,
        "dominant_attitude_emotion": dominant_attitude_emotion,
        "dominant_attitude_score": dominant_attitude_score
    }

    # Final result
    result = {
        "average_scores": average_scores,
        "dominant_emotion": dominant_emotion,
        "dominant_emotion_score": dominant_emotion_score,
        "respect_contempt_json": respect_contempt_json
    }

    return result

def result_to_dataframe(result):
    # Flatten nested dicts
    rows = []

    for label, score in result["average_scores"].items():
        rows.append({
            "label": label,
            "avg_score": score,
            "category": (
                "respect" if label in result["respect_contempt_json"]["respect_emotions"]
                else "contempt" if label in result["respect_contempt_json"]["contempt_emotions"]
                else "other"
            ),
            "dominant_attitude": (
                "yes" if label == result["respect_contempt_json"]["dominant_attitude_emotion"] else "no"
            ),
            "dominant_emotion": (
                "yes" if label == result["dominant_emotion"] else "no"
            )
        })
    return pd.DataFrame(rows)


def run_go_emotions(transcript, model_type):
    model, tokenizer = load_model()
    output_json = classify_emotions(transcript, model_type, model)
    f = result_to_dataframe(output_json)
    f.to_csv("emotion_output.csv", index = False)

    return output_json

# Example usage
if __name__ == "__main__":
    run_go_emotions()
    
