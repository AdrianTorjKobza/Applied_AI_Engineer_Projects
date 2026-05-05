import streamlit as st # Framework used to build the web interface.
from query import ask

st.set_page_config(page_title="My Knowledge Bot", page_icon="📚")
st.title("Ask my notes")

# Allow the app to "remember" your chat history.
if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask anything from your docs...")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your docs..."):
            answer, sources = ask(question)
        st.write(answer)

        with st.expander(f"Sources ({len(sources)} chunks retrieved)"):
            for s in sources:
                st.markdown(f"**{s['source']}** — relevance: `{s['score']}`")
                st.caption(s["text"][:200] + "...")

    st.session_state.history.append({"role": "assistant", "content": answer})