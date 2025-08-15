import streamlit as st
import rag_core # Import our RAG core functions

st.set_page_config(page_title="RAG Chatbot", layout="centered")

st.title("📄 Document Q&A with RAG")
st.markdown("""
Welcome! Ask a question about the documents in our knowledge base.
""")

# Initialize chat history in Streamlit session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about the documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from RAG system
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Call the ask_rag_system function from rag_core.py
                response = rag_core.ask_rag_system(prompt)
                llm_answer = response["answer"]
                retrieved_context = response["context"]

                st.markdown(llm_answer) # Display the LLM's answer

                # Optionally display retrieved context (useful for debugging/transparency)
                if retrieved_context:
                    st.subheader("📚 Retrieved Context")
                    for i, doc in enumerate(retrieved_context):
                        source_info = f"Source: {doc.metadata.get('source', 'N/A')}"
                        page_info = f"Page: {doc.metadata.get('page', 'N/A')}"
                        st.expander(f"Document {i+1} ({source_info}, {page_info})").markdown(
                            f"```\n{doc.page_content}\n```"
                        )
                else:
                    st.info("No specific documents were retrieved for this query.")

                st.session_state.messages.append({"role": "assistant", "content": llm_answer})
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})

