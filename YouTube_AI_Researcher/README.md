# YouTube AI Research Assistant

AI tool that extracts transcripts from YouTube videos and uses Groq-powered Llama 3 models to query the content.

## Key Features
- **Instant Extraction:** Fetches full transcripts from YouTube URLs (Supports Standard and Short links).
- **SSL Bypass:** Implements a custom "Monkey Patch" for the `requests` library to bypass SSL certificate errors (common on Windows/Zscaler environments).
- **AI-Powered Insights:** Uses Groq's inference engine to summarize, search, and analyze video content.
- **Session Caching:** Remembers your transcript so you can ask follow-up questions without re-downloading data.

## Tech Stack
- **UI Framework:** [Streamlit](https://streamlit.io/) (Web-based dashboard)
- **Inference Engine:** [Groq Cloud API](https://console.groq.com/)
- **Large Language Model:** Llama 3.3 70B (State-of-the-art open-source model)
- **Orchestration:** [LangChain Core](https://python.langchain.com/)
- **Data Source:** [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api)
- **Networking:** `httpx`, `requests`, and `urllib3`

## Requirements
The following packages to be installed:
- `streamlit`
- `langchain-groq`
- `langchain-core`
- `youtube-transcript-api`
- `python-dotenv`
- `requests`
- `httpx`