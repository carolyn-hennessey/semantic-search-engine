import streamlit as st
from transformers import pipeline

from preprocessing import generate_chunks, load_chroma, load_text

LLM_MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

st.set_page_config(page_title="Microservice Q&A", page_icon="🌈", layout="centered")

# ---------- Styling: bright, happy, handwritten, lots of white space ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Patrick+Hand&display=swap');

    html, body, [class*="css"], .stApp, input, textarea, button {
        font-family: 'Patrick Hand', 'Comic Sans MS', cursive !important;
    }
    .stApp { background: #FFFDF5; }
    .block-container { padding-top: 6rem; padding-bottom: 6rem; max-width: 720px; }

    h1 {
        color: #FF6B6B;
        text-align: center;
        font-size: 3.2rem !important;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #4ECDC4;
        font-size: 1.5rem;
        margin-bottom: 4rem;
    }
    label p { color: #7B61FF; font-size: 1.4rem !important; }

    .stTextInput input {
        background: #FFF3B0;
        border: 3px solid #FFB703 !important;
        border-radius: 18px;
        font-size: 1.4rem;
        padding: 0.9rem 1rem;
        color: #333;
    }
    .answer-box {
        margin-top: 4rem;
        padding: 2rem;
        min-height: 12rem;
        background: #E0F7F4;
        border: 3px dashed #4ECDC4;
        border-radius: 24px;
        font-size: 1.4rem;
        line-height: 1.6;
        color: #333;
    }
    .answer-box.empty { color: #9aa; text-align: center; padding-top: 4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Resources (loaded once) ----------
@st.cache_resource(show_spinner="Getting the books ready... 📚")
def get_collection():
    chunks = generate_chunks(load_text())
    return load_chroma(chunks)


@st.cache_resource(show_spinner="Waking up the little robot... 🤖")
def get_llm():
    return pipeline("text-generation", model=LLM_MODEL_NAME)


def answer_question(question: str) -> str:
    collection = get_collection()
    results = collection.query(query_texts=[question], n_results=1)
    context = " ".join(results["documents"][0])

    messages = [
        {
            "role": "system",
            "content": "You are a helpful teacher. Answer the question using only "
            "the context provided. If the context doesn't help, say you don't know. Make sure you also reference the context that you discovered so the user knows you found something.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]
    output = get_llm()(
        messages, max_new_tokens=256, do_sample=False, return_full_text=False
    )
    return (output[0]["generated_text"].strip(), context)


# ---------- UI ----------
st.title("🌈 Ask Me Anything!")
st.markdown(
    '<div class="subtitle">Curious about microservices? Let\'s find out together ✏️</div>',
    unsafe_allow_html=True,
)

question = st.text_input(
    "Ask a question about microservice architecture",
    placeholder="e.g. What are microservices?",
)

if question:
    with st.spinner("Thinking really hard... 💭"):
        answer, context = answer_question(question)
    st.markdown(f'<div class="answer-box"><p>{answer}</p><div><h3>References:</h3><p>{context}</p></div></div>', unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="answer-box empty">Your answer will show up here! ⭐</div>',
        unsafe_allow_html=True,
    )
