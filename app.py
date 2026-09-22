import streamlit as st
import pandas as pd
import os
from transformers import pipeline

st.set_page_config(page_title="Question Answering App", layout="centered")

if "page" not in st.session_state:
    st.session_state.page = "welcome"

if st.session_state.page == "welcome":
    st.title("📘 Question Answering App")
    st.write("### Welcome!")
    st.write("""
    This application uses a **pretrained Transformer model** (via Hugging Face) 
    to answer questions based on a paragraph you provide.

    **Features:**
    - Paste a paragraph or upload a text file
    - Ask a question and get an instant, AI-extracted answer
    - Confidence scoring for every answer
    - Question history that persists across sessions
    - Export your history as a CSV file
    """)
    st.write("---")
    if st.button("🚀 Get Started", type="primary"):
        st.session_state.page = "app"
        st.rerun()

elif st.session_state.page == "app":

    HISTORY_FILE = "history.csv"

    @st.cache_resource
    def load_model():
        return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    qa_model = load_model()

    if "history" not in st.session_state:
        if os.path.exists(HISTORY_FILE):
            st.session_state.history = pd.read_csv(HISTORY_FILE).to_dict("records")
        else:
            st.session_state.history = []

    if "context_input" not in st.session_state:
        st.session_state.context_input = ""
    if "question_input" not in st.session_state:
        st.session_state.question_input = ""

    if st.button("⬅ Back to Welcome"):
        st.session_state.page = "welcome"
        st.rerun()

    st.title("Question Answering App")
    st.write("Paste a paragraph (or upload a text file), ask a question, and get an instant answer.")

    with st.expander("Try an example"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Eiffel Tower example"):
                st.session_state.context_input = "The Eiffel Tower is located in Paris, France. It was completed in 1889 and stands 330 meters tall."
                st.session_state.question_input = "How tall is the Eiffel Tower?"
        with col2:
            if st.button("Solar System example"):
                st.session_state.context_input = "Jupiter is the largest planet in the solar system. It has 95 known moons and is primarily made of hydrogen and helium."
                st.session_state.question_input = "How many moons does Jupiter have?"

    st.subheader("Your Input")
    uploaded_file = st.file_uploader("Or upload a .txt file as your paragraph", type=["txt"])
    if uploaded_file is not None:
        st.session_state.context_input = uploaded_file.read().decode("utf-8")

    context = st.text_area("Paragraph", value=st.session_state.context_input, height=150)
    question = st.text_input("Question", value=st.session_state.question_input)

    if st.button("Get Answer", type="primary"):
        if context and question:
            result = qa_model(question=question, context=context)
            answer = result["answer"]
            score = result["score"]

            st.subheader("Result")

            if score >= 0.7:
                st.success(f"✅ Answer: {answer}")
                st.write(f"Confidence: {score:.2%} — High")
            elif score >= 0.4:
                st.info(f"ℹ️ Answer: {answer}")
                st.write(f"Confidence: {score:.2%} — Moderate")
            else:
                st.warning(f"⚠️ Answer: {answer}")
                st.write(f"Confidence: {score:.2%} — Low, may be unreliable")

            st.session_state.history.append({
                "Question": question,
                "Answer": answer,
                "Confidence": f"{score:.2%}"
            })
            pd.DataFrame(st.session_state.history).to_csv(HISTORY_FILE, index=False)
        else:
            st.warning("Please enter both a paragraph and a question.")

    if len(st.session_state.history) > 0:
        st.subheader("Question History")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear History"):
                st.session_state.history = []
                if os.path.exists(HISTORY_FILE):
                    os.remove(HISTORY_FILE)
                st.rerun()
        with col2:
            csv_data = history_df.to_csv(index=False)
            st.download_button(
                label="Download History as CSV",
                data=csv_data,
                file_name="question_history.csv",
                mime="text/csv"
            )