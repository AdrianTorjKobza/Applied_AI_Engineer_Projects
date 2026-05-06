import os
import httpx
import requests
from requests.sessions import Session
import urllib3
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# SSL BYPASS (sometimes needed)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_original_request = Session.request

def _patched_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return _original_request(self, method, url, **kwargs)

Session.request = _patched_request

load_dotenv()

st.set_page_config(page_title="YouTube Researcher", page_icon="📺")

class YouTubeAssistant:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            http_client=httpx.Client(verify=False) 
        )

    def extract_video_id(self, url):
        if "v=" in url:
            return url.split("v=")[1].split("&")[0]
        elif "be/" in url:
            return url.split("be/")[1]
        return None

    def get_transcript(self, video_id):
        try:
            api_instance = YouTubeTranscriptApi()
            
            if hasattr(api_instance, 'fetch'):
                return api_instance.fetch(video_id)
            else:
                transcript_list = api_instance.list(video_id)
                return transcript_list.find_transcript(['en']).fetch()
                
        except Exception as e:
            st.error(f"❌ Transcript Error: {e}")
            return None

# --- UI STATE ---
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = None

# --- UI LAYOUT ---
st.title("📺 YouTube Research Assistant")

with st.sidebar:
    st.header("Step 1: Video Link")
    url = st.text_input("YouTube URL:", placeholder="Paste link here...")
    extract_btn = st.button("📥 Extract Transcript")

# --- MAIN LOGIC ---
if url:
    assistant = YouTubeAssistant()
    video_id = assistant.extract_video_id(url)
    
    if video_id:
        if extract_btn:
            with st.spinner("📥 Bypassing Firewall & Fetching Transcript..."):
                data = assistant.get_transcript(video_id)
                if data:
                    text_chunks = []
                    for item in data:
                        if isinstance(item, dict):
                            text_chunks.append(item.get('text', ''))
                        else:
                            text_chunks.append(getattr(item, 'text', str(item)))
                            
                    st.session_state.transcript_text = " ".join(text_chunks)
                    st.success("✅ Transcript extracted successfully!")

        if st.session_state.transcript_text:
            with st.expander("📄 View Extracted Transcript (first 3000 characters)"):
                st.write(st.session_state.transcript_text[:3000] + " ...")

            st.header("Step 2: Ask a Question")
            query = st.text_input("What do you want to find out from this video?")
            
            if st.button("🚀 Run AI Analysis"):
                if query:
                    with st.spinner("🔍 Groq AI is analyzing..."):
                        context = st.session_state.transcript_text[:8000]
                        prompt = f"Transcript:\n{context}\n\nQuestion:\n{query}"
                        
                        messages = [
                            SystemMessage(content="You are a professional research assistant. Answer based ONLY on the transcript provided."),
                            HumanMessage(content=prompt)
                        ]
                        
                        try:
                            response = assistant.llm.invoke(messages)
                            st.markdown("---")
                            st.markdown("### 📝 AI Insights")
                            st.write(response.content)
                        except Exception as e:
                            st.error(f"Groq API Error: {e}")
    else:
        st.error("Invalid URL. Please check the YouTube link.")
else:
    st.info("👈 Please enter a YouTube URL in the sidebar to begin.")