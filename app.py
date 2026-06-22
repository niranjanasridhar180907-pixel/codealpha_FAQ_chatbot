import streamlit as st  # type: ignore
import pandas as pd  # type: ignore

from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

from datetime import datetime 

st.set_page_config(
    page_title="AI FAQ Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Load CSS
with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# Load HTML
with open("template.html") as f:
    st.markdown(
        f.read(),
        unsafe_allow_html=True
    )

# Sidebar

with st.sidebar:

    st.markdown("""
    ## 📌 About Project

    ### Features

    ✅ NLP Chatbot

    ✅ FAQ Database

    ✅ Confidence Score

    ✅ Chat History

    ✅ Download History

    ✅ Real-Time Answers

    ---

    ### Developer

    👩‍💻 Niranjana Sridhar

    B.Tech AI & DS

    SVCE
    """)

# Dataset

faq = pd.read_csv("faq.csv")

questions = faq["Question"]

answers = faq["Answer"]

vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform(
    questions
)

# Session State

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown(
'<div class="glass-card">',
unsafe_allow_html=True
)

user_input = st.text_input(
    "Ask a Question"
)

if "current_answer" not in st.session_state:
    st.session_state.current_answer = ""

if st.button("Send"):

    if user_input.strip():

        user_vector = vectorizer.transform(
            [user_input]
        )

        similarity = cosine_similarity(
            user_vector,
            vectors
        )

        score = similarity.max()

        index = similarity.argmax()

        if score >= 0.50:

            response = answers.iloc[index]

        elif score >= 0.30:

            response = f"""
Related Answer:

{answers.iloc[index]}
"""

        else:

            response = """
Sorry, I couldn't find an exact answer.
Try rephrasing your question.
"""

        st.session_state.current_answer = response

        st.session_state.history.append(
            {
                "question": user_input,
                "answer": response,
                "confidence": round(score*100,2)
            }
        )

st.markdown("""
<h2 style="
color:white;
margin-top:20px;">
💡 Answer
</h2>
""", unsafe_allow_html=True)

st.markdown(
f"""
<div class='answer-box'>
{st.session_state.current_answer}
</div>
""",
unsafe_allow_html=True
)


st.markdown(
'</div>',
unsafe_allow_html=True
)

st.subheader("💬 Chat History")

for chat in reversed(st.session_state.history):
    st.markdown(
        f"""
        <div class='user-message'>
        👤 {chat['question']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class='bot-message'>
        🤖 {chat['answer']}
        <br><br>
        Confidence:
        {chat['confidence']}%
        </div>
        """,
        unsafe_allow_html=True,
    )

# Download Chat
chat_text=""

for item in st.session_state.history:

    chat_text += f"""
Question: {item['question']}
Answer: {item['answer']}
Confidence: {item['confidence']}%

------------------------------------
"""

if chat_text:

    st.download_button(
        "📥 Download Chat History",
        chat_text,
        file_name="chat_history.txt"
    )

# Dashboard

col1,col2,col3 = st.columns(3)

with col1:
    st.metric(
        "FAQs",
        len(faq)
    )

with col2:
    st.metric(
        "Questions",
        len(st.session_state.history)
    )

with col3:
    st.metric(
        "Status",
        "Active"
    )

st.markdown("""
<div class="footer">

<hr>

<h3>
🚀 Developed using Python,
Streamlit,
Scikit-Learn
</h3>

<p>
CodeAlpha Artificial Intelligence Internship
</p>

<p>
Made by Niranjana Sridhar
</p>

</div>
""", unsafe_allow_html=True)
