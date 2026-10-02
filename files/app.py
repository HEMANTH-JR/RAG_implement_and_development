import streamlit as st

from rag import answer, index_folder

st.set_page_config(page_title="RAG Chat", page_icon="📚")
st.title("📚 Chat with your documents")

with st.sidebar:
    st.header("Documents")
    folder = st.text_input("Folder path", "docs")
    if st.button("Index documents"):
        with st.spinner("Indexing..."):
            n = index_folder(folder)
        st.success(f"Indexed {n} chunks")
    k = st.slider("Passages to retrieve (k)", 1, 10, 4)

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            text, sources = answer(question, k)
        st.markdown(text)
        with st.expander("Sources"):
            for i, (src, snippet) in enumerate(sources, 1):
                st.markdown(f"**[{i}] {src}**\n\n{snippet}")
    st.session_state.messages.append({"role": "assistant", "content": text})
