import streamlit as st
import pandas as pd
import json
import os

from models.classifier import TicketClassifier
from models.rag import RAGPipeline
from utils.display import (
    show_ticket_table,
    show_internal_analysis,
    show_final_response,
    show_section_divider,
    show_user_query_form,
)
from utils.preprocessing import clean_text

# ==============================
# Load models and pipelines
# ==============================
st.set_page_config(page_title="Customer Support Copilot", layout="wide")

# Center the title
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">Dummy Helpdesk Application</h1>', unsafe_allow_html=True)


@st.cache_resource
def load_classifier():
    return TicketClassifier()


@st.cache_resource
def load_rag():
    rag = RAGPipeline()
    try:
        with open("kb/knowledge_base.json", "r", encoding="utf-8") as f:
            kb = json.load(f)
        docs = [entry["text"] for entry in kb]
        sources = [entry["source"] for entry in kb]
        rag.build_index(docs, metadata=sources)
        st.success(f"Knowledge base loaded with {len(docs)} documents.")
    except Exception as e:
        st.warning(f"Knowledge base error: {e}")
    return rag


classifier = load_classifier()
rag = load_rag()

# ==============================
# Bulk ticket classification
# ==============================
st.header("Bulk Ticket Classification")

tickets_data = None
default_path = "data/sample_tickets.json"

if os.path.exists(default_path):
    try:
        with open(default_path, "r", encoding="utf-8") as f:
            tickets_data = json.load(f)
        st.info("Processing tickets from default file sample_tickets.json")
    except Exception as e:
        st.error(f"Error loading default file: {e}")
else:
    st.error("File not found 'sample_tickets.json'")

# Process tickets if data is available
if tickets_data:
    if not isinstance(tickets_data, list) or not all("id" in t and "body" in t for t in tickets_data):
        st.error("JSON must be a list of ticket objects with at least 'id' and 'body' fields.")
    else:
        classifications = []
        for t in tickets_data:
            ticket_text = f"{t.get('subject', '')} {t.get('body', '')}"
            result = classifier.classify_ticket(ticket_text)
            result["Ticket ID"] = t["id"]
            result["Subject"] = t.get("subject", "")
            result["Body"] = t.get("body", "")
            classifications.append(result)

        classified_df = pd.DataFrame(classifications)[
            ["Ticket ID", "Subject", "Body", "Topic", "Sentiment", "Priority"]
        ]

        show_ticket_table(classified_df)

show_section_divider()

# ==============================
# Interactive agent
# ==============================
st.header("Interactive AI Agent")

subject, body, submitted = show_user_query_form()

if submitted and (subject or body):
    user_query = f"{subject} {body}"
    analysis = classifier.classify_ticket(user_query)
    show_internal_analysis(analysis)

    topic = analysis["Topic"]

    if topic in ["How-to", "Product", "Best practices", "API/SDK", "SSO"]:
        answer, sources = rag.generate_answer(user_query)
        show_final_response(answer, sources)
    else:
        response = (
            f"This ticket has been classified as a '{topic}' issue "
            f"and routed to the appropriate team."
        )
        show_final_response(response, sources=[])
