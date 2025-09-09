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
# Page Config & Styling
# ==============================
st.set_page_config(page_title="Customer Support Copilot", layout="wide")

st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #2027d3;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        color: #374151;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.5rem;
    }
    .form-container {
        background-color: #f9fafb;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #eff6ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2027d3;
        margin: 1rem 0;
    }
    .stButton > button {
        background-color: #2027d3;
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #62e1fc;
        color: black;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">Customer Support Copilot</h1>', unsafe_allow_html=True)

# ==============================
# Load Models
# ==============================
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

# Initialize routed tickets log
if "routed_tickets" not in st.session_state:
    st.session_state.routed_tickets = []

# ==============================
# Bulk Ticket Classification
# ==============================
st.markdown('<div class="section-title">Bulk Ticket Classification</div>', unsafe_allow_html=True)

tickets_data = None
default_path = "data/sample_tickets.json"

if os.path.exists(default_path):
    try:
        with open(default_path, "r", encoding="utf-8") as f:
            tickets_data = json.load(f)
        st.markdown('<div class="info-box">Processing tickets from default file: sample_tickets.json</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading default file: {e}")
else:
    st.error("Default file not found at 'data/sample_tickets.json'")

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
# Interactive Agent
# ==============================
st.markdown('<div class="section-title">Interactive AI Agent</div>', unsafe_allow_html=True)

st.markdown('<div class="form-container">', unsafe_allow_html=True)
subject, body, submitted = show_user_query_form()
st.markdown('</div>', unsafe_allow_html=True)

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

        # Save routed query to session log
        st.session_state.routed_tickets.append({
            "Subject": subject,
            "Body": body,
            "Topic": topic,
            "Sentiment": analysis["Sentiment"],
            "Priority": analysis["Priority"],
        })

show_section_divider()

# ==============================
# Routed Tickets Log
# ==============================
st.markdown('<div class="section-title">Routed Tickets Log</div>', unsafe_allow_html=True)

if st.session_state.routed_tickets:
    routed_df = pd.DataFrame(st.session_state.routed_tickets)
    st.dataframe(routed_df, use_container_width=True, hide_index=True)
else:
    st.info("No tickets have been routed yet.")
