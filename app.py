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
st.set_page_config(page_title="Atlan Customer Support Copilot", layout="wide")

st.markdown("""
<style>
    /* Dark theme */
    .stApp {
        background-color: #1a1a1a;
        color: #e5e5e5;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main title */
    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 600;
        margin-bottom: 2rem;
        color: #ffffff;
        padding: 1rem 0;
    }
    
    /* Section titles */
    .section-title {
        font-size: 1.3rem;
        font-weight: 500;
        margin: 2rem 0 1rem 0;
        color: #ffffff;
        border-bottom: 1px solid #333333;
        padding-bottom: 0.5rem;
    }
    
    /* Remove form container styling */
    .form-container {
        margin: 1rem 0;
    }
    
    /* Info messages */
    .info-box {
        background-color: #2a2a2a;
        padding: 0.8rem;
        border-radius: 4px;
        border-left: 3px solid #4a9eff;
        margin: 1rem 0;
        color: #e5e5e5;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #333333;
        color: #ffffff;
        border: 1px solid #555555;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-weight: 400;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #4a4a4a;
        border-color: #777777;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #2a2a2a;
        color: #ffffff;
        border: 1px solid #555555;
        border-radius: 4px;
    }
    
    /* DataFrames */
    .stDataFrame {
        background-color: transparent;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1a1a1a;
    }
    
    /* Success/warning/error messages */
    .stSuccess {
        background-color: #1a3d1a;
        border: 1px solid #2d5a2d;
        color: #90ee90;
    }
    
    .stWarning {
        background-color: #3d3d1a;
        border: 1px solid #5a5a2d;
        color: #ffeb3b;
    }
    
    .stError {
        background-color: #3d1a1a;
        border: 1px solid #5a2d2d;
        color: #ff6b6b;
    }
    
    .stInfo {
        background-color: #1a2a3d;
        border: 1px solid #2d4a5a;
        color: #87ceeb;
    }
    
    /* Remove padding/margins for cleaner look */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Dividers */
    hr {
        border: none;
        height: 1px;
        background-color: #333333;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">Customer Support Copilot for Atlan</h1>', unsafe_allow_html=True)

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
        # Just check if DB is accessible
        count = rag.collection.count()
        st.success(f"Knowledge base loaded with {count} documents.")
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
        st.markdown('<div class="info-box">Processing tickets from: sample_tickets.json</div>', unsafe_allow_html=True)
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

st.markdown("<hr>", unsafe_allow_html=True)

# ==============================
# Interactive Agent
# ==============================
st.markdown('<div class="section-title">Interactive AI Agent</div>', unsafe_allow_html=True)

subject, body, submitted = show_user_query_form()

if submitted and (subject or body):
    user_query = f"{subject} {body}"
    analysis = classifier.classify_ticket(user_query)
    show_internal_analysis(analysis)

    topic = analysis["Topic"]

    if topic in ["How-to", "Product", "Best practices", "API/SDK", "SSO"]:
        answer, sources, confidence_val = rag.generate_answer(user_query)
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

st.markdown("<hr>", unsafe_allow_html=True)

# ==============================
# Routed Tickets Log
# ==============================
st.markdown('<div class="section-title">Routed Tickets Log</div>', unsafe_allow_html=True)

if st.session_state.routed_tickets:
    routed_df = pd.DataFrame(st.session_state.routed_tickets)
    st.dataframe(routed_df, width='stretch', hide_index=True)
else:
    st.info("No tickets have been routed yet.")
    