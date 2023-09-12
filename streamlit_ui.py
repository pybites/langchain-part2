# for an intro to streamlit see https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
import streamlit as st

from ai_lib import create_conversation_chain, load_youtube_video, query_qa, query_video

st.markdown("# Pybites AI")

youtube_url = st.text_input("Insert Youtube link:")


refresh_session = st.button("Reset Session")

# Initialize chat history
if refresh_session or "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.reload_youtube_video = True
    st.session_state.openai_lc_client = None
    st.session_state.result = None
else:
    st.session_state.reload_youtube_video = False


if youtube_url:
    # if st.session_state.reload_youtube_video:
    st.session_state.openai_lc_client, st.session_state.result = load_youtube_video(youtube_url)


    result = st.session_state.result
    openai_lc_client = st.session_state.openai_lc_client

    st.markdown(f"Title: {result[0].metadata['title']}")
    st.markdown(f"Duration: {result[0].metadata['length'] / 60:.2f} minutes")
    st.markdown(f"Author: {result[0].metadata['author']}")

# Accept user input
if prompt := st.chat_input("What do you want to know about the youtube video?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    image = "human.png" if message["role"] == "user" else "bot.png"
    with st.chat_message(message["role"], avatar=image):
        st.markdown(message["content"])

if len(st.session_state.messages) != 0 and st.session_state.messages[-1]["role"] != "assistant":

    with st.chat_message("assistant", avatar="bot.png"):
        with st.spinner("Thinking..."):
            qa = create_conversation_chain(db=openai_lc_client)

            full_response = query_qa(qa, prompt)
            # full_response = query_video(openai_lc_client, prompt)
            answer = full_response["answer"]
            st.session_state.messages.append({"role": "assistant", "content": answer})
        st.write(answer)


