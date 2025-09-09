# models/classifier.py

from transformers import pipeline
from utils.preprocessing import clean_text

class TicketClassifier:
    def __init__(self):
        self.topic_labels = [
            "How-to", "Product", "Connector", "Lineage", "API/SDK",
            "SSO", "Glossary", "Best practices", "Sensitive data",
        ]

        self.emotion_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
            truncation=True
        )

        self.topic_pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )

    def classify_topic(self, text: str) -> str:
        text = clean_text(text)
        result = self.topic_pipeline(text, candidate_labels=self.topic_labels, multi_label=False)
        return result["labels"][0]

    def classify_emotion(self, text: str) -> str:
        """
        Classify emotion and map to business-specific categories.
        """
        text = clean_text(text)
        scores = self.emotion_pipeline(text)[0]  # list of dicts
        scores = sorted(scores, key=lambda x: x["score"], reverse=True)
        top = scores[0]["label"].lower()

        # --- Map model labels to your schema ---
        mapping = {
            "anger": "Angry",
            "sadness": "Frustrated",
            "fear": "Frustrated",
            "disgust": "Frustrated",
            "surprise": "Curious",
            "joy": "Curious",
            "neutral": "Neutral",
        }
        return mapping.get(top, "Neutral")

    def classify_priority(self, text: str) -> str:
        """
        Classify ticket priority using zero-shot classification.
        """
        text = clean_text(text)

        candidate_labels = ["P0 (High)", "P1 (Medium)", "P2 (Low)"]

        result = self.topic_pipeline(
            text,
            candidate_labels=candidate_labels,
            multi_label=False
        )
        return result["labels"][0]


    def classify_ticket(self, text: str) -> dict:
        return {
            "Topic": self.classify_topic(text),
            "Sentiment": self.classify_emotion(text),
            "Priority": self.classify_priority(text),
        }