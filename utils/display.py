import streamlit as st
import pandas as pd


def show_ticket_table(tickets: pd.DataFrame):
    """
    Display the bulk classification dashboard for tickets in a styled table.
    """
    st.subheader("Bulk Ticket Classification Dashboard")

    styled_df = tickets.style.set_properties(**{
        "text-align": "left",
        "white-space": "pre-wrap"
    })

    st.dataframe(
        styled_df,
        width='stretch',
        hide_index=True,
    )


def show_internal_analysis(classification: dict):
    """
    Display internal analysis of a single ticket in a 3–4 column layout.
    """
    st.markdown("### Internal Analysis")

    cols = st.columns(4)
    cols[0].metric("Topic", classification.get("Topic", "N/A"))
    cols[1].metric("Sentiment", classification.get("Sentiment", "N/A"))
    cols[2].metric("Priority", classification.get("Priority", "N/A"))
    if "SentimentConfidence" in classification:
        conf = classification.get("SentimentConfidence")
        cols[3].metric("Confidence", f"{conf:.2f}" if conf is not None else "N/A")


def show_final_response(response: str, sources: list):
    """
    Display the AI agent's final response with optional structured sources.

    Args:
        response (str): The generated response for the user.
        sources (list): List of dicts: {"id": int, "url": str, "excerpt": str}
    """
    st.markdown("### Final Response")
    st.write(response)

    if sources:
        st.markdown("#### Sources")
        for s in sources:
            sid = s.get("id")
            url = s.get("url", "")
            excerpt = s.get("excerpt", "")
            # Display numbered source with clickable link
            st.markdown(f"**[{sid}]** [{url}]({url})")
            if excerpt:
                with st.expander(f"View excerpt from source [{sid}]"):
                    st.write(excerpt)


def show_user_query_form():
    """
    Display a form for user to enter subject and body of a query.

    Returns:
        tuple (subject, body, submitted)
    """
    st.markdown("### Submit a New Ticket")

    with st.form("user_ticket_form"):
        subject = st.text_input("Subject", placeholder="Enter the subject of your query")
        body = st.text_area("Body", placeholder="Describe your issue or question in detail")
        submitted = st.form_submit_button("Submit")

    return subject, body, submitted


def show_section_divider():
    """Helper to visually separate sections in the Streamlit UI."""
    st.markdown(
        "<hr style='border:1px solid #2027d3; margin: 1rem 0;'>",
        unsafe_allow_html=True,
    )
