import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.vectorstores import Chroma
import os


def load_docs_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                documents.append(content)
    return documents


def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.create_documents(documents)
    return chunks


def create_vectorstore(persist_directory, text_chunks, user_question):
    embeddings = OpenAIEmbeddings()
    
    # Check if the persistent vector database already exists
    if os.path.exists(persist_directory):
        final_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        vector_store = Chroma.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vector_store.persist()
        final_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        final_db.similarity_search_with_score(user_question,k=3)
    return final_db
    # vector_store.similarity_search_with_score(user_question, k=3)
    # return vector_store


def create_conversation_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_user_input(user_question):
    if callable(st.session_state.conversation):
        response = st.session_state.conversation({'question': user_question})
        if st.session_state.chat_history is None:
            st.session_state.chat_history = []
        st.session_state.chat_history = response['chat_history'] + st.session_state.chat_history
        for i, message in list(enumerate(st.session_state.chat_history)):
            if i % 2 != 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.error("Conversation chain is not initialized. Please process documents first.")


def list_files(directory):
    files = []
    for filename in os.listdir(directory):
        files.append(os.path.join(directory, filename))
    return files


def main():
    os.environ["OPENAI_API_KEY"] = "YOUR OPENAI_API_KEY"
    st.set_page_config(page_title="Chat with multiple Files",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple files :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question)

    directory_path = "D:\\Python\\Visual Studio\\OPENAI\\contents"
    
    if os.path.isdir(directory_path):
        documents = load_docs_from_folder(directory_path)
        if documents:
            persist_directory = "chroma_db"
            text_chunks = get_text_chunks(documents)
            vector_store = create_vectorstore(
                persist_directory, text_chunks, user_question)
            st.session_state.conversation = create_conversation_chain(
                vector_store)
        else:
            st.warning("No text files found in the folder.")
    else:
        st.error("Invalid folder path. Please enter a valid path.")


if __name__ == '__main__':
    main()
