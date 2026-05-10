import re
import json
import psycopg2  # pyright: ignore[reportMissingModuleSource]
import streamlit as st  # pyright: ignore[reportMissingImports]
import streamlit.components.v1 as components  # pyright: ignore[reportMissingImports]
from langchain_community.embeddings import HuggingFaceEmbeddings  # pyright: ignore[reportMissingImports]
from langchain_community.vectorstores import FAISS  # pyright: ignore[reportMissingImports]
from langchain.text_splitter import CharacterTextSplitter  # pyright: ignore[reportMissingImports]
from langchain.schema import Document  # pyright: ignore[reportMissingImports]
from youtube_transcript_api import (  # pyright: ignore[reportMissingImports]
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
)

# ---------------- UI ----------------
st.set_page_config(page_title="AI Transcript Tutor", layout="centered")
st.title("🎓 AI Transcript Power Tutor App")
st.write("Ask questions from YouTube lectures")

# ---------------- SESSION INIT ----------------
if "saved" not in st.session_state:
    st.session_state.saved = False

# ---------------- DATABASE ----------------
def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="postgres",
        user="postgres",
        password="Database",  
        port="5432"
    )

def save_to_db(video_url, question, answer):
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
        INSERT INTO transcript_history (video_url, question, answer)
        VALUES (%s, %s, %s)
        """, (video_url, question, answer))

        conn.commit()
        cur.close()
        conn.close()

        st.success("✅ Saved to database!")

    except Exception as e:
        st.error(f"❌ DB Error: {e}")

# ---------------- TEXT TO SPEECH ----------------
def speak_browser(text):
    safe_text = json.dumps(text)
    components.html(f"""
    <script>
    var msg = new SpeechSynthesisUtterance({safe_text});
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(msg);
    </script>
    """, height=0)

def stop_speaking():
    components.html("""
    <script>
    window.speechSynthesis.cancel();
    </script>
    """, height=0)

# ---------------- VOICE INPUT ----------------
def voice_input():
    components.html("""
    <script>
    var recognition = new webkitSpeechRecognition();
    recognition.lang = 'en-US';
    recognition.start();

    recognition.onresult = function(event) {
        const text = event.results[0][0].transcript;
        const inputs = window.parent.document.querySelectorAll('input[type="text"]');
        inputs[inputs.length - 1].value = text;
        inputs[inputs.length - 1].dispatchEvent(new Event('input', { bubbles: true }));
    };
    </script>
    """, height=0)

# ---------------- YOUTUBE ID ----------------
_YOUTUBE_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_-]{11})"
)

def extract_youtube_video_id(url):
    if not url:
        return None
    match = _YOUTUBE_ID_RE.search(url)
    return match.group(1) if match else None

# ---------------- GET TRANSCRIPT ----------------
def get_youtube_transcript(url):
    video_id = extract_youtube_video_id(url)

    if not video_id:
        st.error("❌ Invalid YouTube URL")
        return None

    api = YouTubeTranscriptApi()
    preferred_langs = ("en", "en-US", "en-GB", "en-IN")

    try:
        fetched = api.fetch(video_id, languages=preferred_langs)
        return " ".join([t.text for t in fetched])

    except NoTranscriptFound:
        try:
            tlist = api.list(video_id)
            transcripts = list(tlist)

            if not transcripts:
                st.error("❌ No captions available")
                return None

            fetched = transcripts[0].fetch()
            st.info(f"Using {fetched.language} transcript")
            return " ".join([t.text for t in fetched])

        except (CouldNotRetrieveTranscript, VideoUnavailable) as e:
            st.error(f"Error: {e}")
            return None

    except TranscriptsDisabled:
        st.error("❌ Captions disabled")
    except VideoUnavailable:
        st.error("❌ Video unavailable")
    except Exception as e:
        st.error(f"Error: {e}")

    return None

# ---------------- INPUT ----------------
video_url = st.text_input("📺 Enter YouTube Video URL")

# ---------------- PROCESS ----------------
if st.button("🚀 Process Video"):
    st.session_state.saved = False  # reset save flag for new video

    if not video_url.strip():
        st.warning("⚠️ Please enter a URL")
    else:
        transcript = get_youtube_transcript(video_url)

        if transcript:
            documents = [Document(page_content=transcript)]

            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )

            vectorstore = FAISS.from_documents(docs, embeddings)
            st.session_state.retriever = vectorstore.as_retriever()

            st.success("✅ Video processed! Ask your question 👇")

# ---------------- QUESTION ----------------
if "retriever" in st.session_state:

    col1, col2 = st.columns([4,1])

    with col1:
        user_question = st.text_input("💬 Ask a question", key="question")
        

    if user_question:
        docs = st.session_state.retriever.get_relevant_documents(user_question)

        if docs:
            answer = " ".join([doc.page_content for doc in docs[:3]])
            st.session_state.answer = answer

            # ✅ SAVE TO DB (ONCE PER QUESTION)
            if not st.session_state.saved:
                save_to_db(video_url, user_question, answer)
                st.session_state.saved = True

        else:
            st.warning("No relevant answer found")

# ---------------- SHOW ANSWER ----------------
if "answer" in st.session_state:
    st.write("### 📘 Answer:")
    st.write(st.session_state.answer)

    col1, col2, col3 = st.columns([1,1,4])

    with col1:
        if st.button("▶️ Read"):
            speak_browser(st.session_state.answer)

    with col2:
        if st.button("⛔ Stop"):
            stop_speaking()

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp {
    background-color: #696969;
}
</style>
""", unsafe_allow_html=True)