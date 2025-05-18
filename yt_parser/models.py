import nltk
from collections import defaultdict, Counter
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import pandas as pd
import sys
import re
import torch

def split_into_paragraphs(text):
    """Robust paragraph splitting with length filtering"""
    paragraphs = re.split(r'\n\s*\n+', text.strip())
    return [p.strip() for p in paragraphs if len(p.strip()) > 25]

def load_model():
    """Load base model with error handling"""
    model_name = "SamLowe/roberta-base-go_emotions"
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Base model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None, None

def classify_emotions_paragraphwise(transcript, model_type, model):
    """Enhanced paragraph analysis with model selection"""
    
    # Clean and prepare text
    cleaned_text = re.sub(r'\s+', ' ', transcript).strip()
    paragraphs = split_into_paragraphs(cleaned_text)
    
    # Initialize aggregators
    emotion_totals = defaultdict(float)
    emotion_counts = defaultdict(int)
    emotion_max = defaultdict(float)
    dominant_emotions = []
    para_weights = []

    # Configure classifier based on model type
    if model_type == "roberta_go_emotions":
        classifier = pipeline(
            task="text-classification",
            model="SamLowe/roberta-base-go_emotions",
            tokenizer=AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions"),
            top_k=None,
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=512
        )
    elif model_type == "go_emotions":
        classifier = pipeline(
            task="text-classification",
            model=model,
            tokenizer=AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions"),
            top_k=None,
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=512
        )
    elif model_type == "siebert":
        classifier = pipeline(
            task="sentiment-analysis",
            model="siebert/sentiment-roberta-large-english",
            tokenizer=AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english"),
            top_k=None,
            device=0 if torch.cuda.is_available() else -1,
            truncation=True,
            max_length=512
        )
    else:
        raise ValueError("Invalid model_type. Choose: 'roberta_go_emotions', 'go_emotions', or 'siebert'")

    for idx, para in enumerate(paragraphs):
        try:
            # Get paragraph scores
            results = classifier(para)
            if not results:
                continue
                
            para_scores = {item['label']: item['score'] for item in results[0]}
            para_length = len(para)
            para_weights.append(para_length)

            # Update aggregators
            for label, score in para_scores.items():
                emotion_totals[label] += score * para_length  # Length-weighted
                emotion_counts[label] += 1
                emotion_max[label] = max(emotion_max[label], score)

            # Track dominant emotion
            if para_scores:
                dominant_emotions.append(max(para_scores, key=para_scores.get))

        except Exception as e:
            print(f"Skipped paragraph {idx+1}: {str(e)}")

    # Calculate weighted averages
    total_weight = sum(para_weights) or 1
    weighted_avg = {
        label: (emotion_totals[label] / total_weight)
        for label in (model.config.id2label.values() if model_type != "siebert" else ["negative", "positive"])
    }

    # Combine with max scores
    combined_scores = {
        label: (weighted_avg.get(label, 0) + emotion_max.get(label, 0)) / 2
        for label in weighted_avg
    }
    sorted_scores = dict(sorted(combined_scores.items(), key=lambda x: x[1], reverse=True))

    # Dominant emotion calculations
    soft_dominant = max(sorted_scores, key=sorted_scores.get) if sorted_scores else None
    hard_dominant = Counter(dominant_emotions).most_common(1)[0][0] if dominant_emotions else None

    # Respect/Contempt analysis (only for Go Emotions models)
    respect_set = {"admiration", "approval", "caring"}
    contempt_set = {"annoyance", "disapproval", "disgust"}
    
    respect_scores = {k: v for k, v in sorted_scores.items() if k in respect_set}
    contempt_scores = {k: v for k, v in sorted_scores.items() if k in contempt_set}
    
    attitude_emotion = max({**respect_scores, **contempt_scores}, 
                          key=sorted_scores.get, default=None)

    return {
        "average_scores": sorted_scores,
        "soft_dominant_emotion": soft_dominant,
        "soft_dominant_emotion_score": sorted_scores.get(soft_dominant, 0),
        "hard_dominant_emotion": hard_dominant,
        "hard_dominant_emotion_confidence": (Counter(dominant_emotions)[hard_dominant]/len(dominant_emotions) if hard_dominant else 0),
        "respect_contempt_json": {
            "respect_emotions": respect_scores,
            "contempt_emotions": contempt_scores,
            "dominant_attitude_emotion": attitude_emotion,
            "dominant_attitude_score": sorted_scores.get(attitude_emotion, 0)
        }
    }

def result_to_dataframe(result):
    """Maintain original output format"""
    all_labels = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring",
        "confusion", "curiosity", "desire", "disappointment", "disapproval",
        "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
        "joy", "love", "nervousness", "optimism", "pride", "realization",
        "relief", "remorse", "sadness", "surprise", "neutral"
    ] + ["positive", "negative"]  # For Siebert compatibility

    rows = []
    for label in all_labels:
        rows.append({
            "label": label,
            "avg_score": result["average_scores"].get(label, 0.0),
            "category": (
                "respect" if label in result["respect_contempt_json"]["respect_emotions"]
                else "contempt" if label in result["respect_contempt_json"]["contempt_emotions"]
                else "other"
            ),
            "hard_dominant_emotion": "yes" if label == result["hard_dominant_emotion"] else "no",
            "soft_dominant_emotion": "yes" if label == result["soft_dominant_emotion"] else "no"
        })

    return pd.DataFrame(rows)

def run_go_emotions(transcript, model_type="roberta_go_emotions"):
    """Main execution function"""
    model, _ = load_model() if model_type in ["roberta_go_emotions", "go_emotions"] else (None, None)
    output_json = classify_emotions_paragraphwise(transcript, model_type, model)
    df = result_to_dataframe(output_json)
    return output_json, df

# Example usage
if __name__ == "__main__":
    run_go_emotions()