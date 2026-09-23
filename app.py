import re

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

st.set_page_config(page_title="Transcript RAG", page_icon="Q", layout="centered")

st.markdown(
    """
    <style>
    .stApp { background: #1a1a1a; }
    .block-container { max-width: 820px; padding-top: 3rem; }
    h1 { color: blue; letter-spacing: -0.03em; }
    .lede { color: lightblue; font-size: 1.05rem; margin-bottom: 2rem; }
    .stButton > button { background: #1d5c4a; color: white; border: 0; }
    .stButton > button:hover { background: #154637; color: white; }
    [data-testid="stChatMessage"] { border: 1px sorag.ipynblid #d9ded7; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Ask the transcript")
st.markdown(
    '<p class="lede">Drop in a YouTube video and ask focused questions about what it says.</p>',
    unsafe_allow_html=True,
)


def extract_video_id(value):
    value = value.strip()
    match = re.search(r"(?:v=|youtu\.be/|youtube\.com/embed/)([\w-]{11})", value)
    return match.group(1) if match else value if re.fullmatch(r"[\w-]{11}", value) else None


@st.cache_resource(show_spinner=False)
def build_retriever(video_id):
    transcript = " ".join(
        snippet.text for snippet in YouTubeTranscriptApi().fetch(video_id)
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4}), len(chunks)


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)


prompt = PromptTemplate(
    template="""You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}""",
    input_variables=["context", "question"],
)

with st.sidebar:rag.ipynb
    if not video_id:
        st.sidebar.error("Enter a valid 11-character YouTube ID or URL.")
    else:
        with st.spinner("Fetching transcript and building index..."):
            try:
                retriever, chunk_count = build_retriever(video_id)
                st.session_state.video_id = video_id
                st.session_state.retriever = retriever
                st.session_state.messages = []
                st.sidebar.success(f"Ready: {chunk_count} transcript chunks")
            except Exception as error:
                st.sidebar.error(f"Could not load video: {error}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask something about the video...")
if question:
    if "retriever" not in st.session_state:
        st.warning("Load a YouTube transcript first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the transcript..."):
                documents = st.session_state.retriever.invoke(question)
                context = "\n\n".join(document.page_content for document in documents)
                response = get_llm().invoke(prompt.invoke({"context": context, "question": question}))
                answer = response.content
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
