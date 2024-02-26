import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.vectorstores import Chroma
import os


def load_docs(text_files):
    documents = []
    for text_file in text_files:
        content = text_file.getvalue().decode("utf-8")
        documents.append(content)
    return documents


def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.create_documents(documents)
    return chunks


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    

    for i, message in reversed(list(enumerate(st.session_state.chat_history))):
        if i % 2 != 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        


def main():
    os.environ["OPENAI_API_KEY"]= "Your OPENAI_API_KEY"
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
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        text_file = st.file_uploader(
            "Upload your text files here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):

                documents = load_docs(text_file)
 
                # Create OpenAIEmbeddings object
                embeddings = OpenAIEmbeddings()

                # Persist directory for Chroma database
                persist_directory = "chroma_db3"

                # get the text chunks
                text_chunks = get_text_chunks(documents)

                # Create Chroma database
                vectordb = Chroma.from_documents(documents=text_chunks, embedding=embeddings, persist_directory=persist_directory)
                vectordb.persist()
                final_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
                matching_final= final_db.similarity_search_with_score(user_question,k=3)

                # create vector store
                #vectorstore = get_vectorstore(persist_directory)


                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    final_db)


if __name__ == '__main__':
    main()
