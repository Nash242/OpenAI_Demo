import streamlit as st
import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
from streamlit_chat import message


def load_docs(folder_path):
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith('.docx') or file.endswith('.doc'):
            doc_path = os.path.join(folder_path, file)
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())
    return documents

def split_docs(documents, chunk_size=1000, chunk_overlap=50):
    text_split = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_split.split_documents(documents)
    return docs

def create_vectors(persist_directory, directory, docs):
    embeddings = OpenAIEmbeddings()
    if os.path.exists(persist_directory):
        processed_files = st.session_state.get('processed_files', [])
        current_files = os.listdir(directory)
        new_files = [file for file in current_files if file not in processed_files]
        if new_files:
            documents = load_docs(directory)
            docs = split_docs(documents)

            embeddings = OpenAIEmbeddings()

            # Create or fetch Chroma vector database
            vectors = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
            vectors.persist()
            vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            return vectordb
        else:
            vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            return vectordb
    else:
        os.makedirs(persist_directory)
        # Create or fetch Chroma vector database
        vectors = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
        vectors.persist()
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return vectordb


def main():
    os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"
    st.set_page_config(page_title="Chat with multiple Files", page_icon=":books:")
    st.title("Document QA Bot")
    st.header("Chat with multiple files :books:")
    
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    
    if 'past' not in st.session_state:
        st.session_state['past'] = []

    directory = "D:\\Python\\Visual Studio\\OPENAI\\contents\\"
    persist_directory = "chroma_db"    

    # Load documents
    documents = load_docs(directory)
    docs = split_docs(documents)

    vectordb = create_vectors(persist_directory, directory, docs)
    llm_model = "gpt-3.5-turbo"
    llm = ChatOpenAI(temperature=0.1, model=llm_model)
    qa_chain = load_qa_chain(llm, chain_type="stuff")

    question = st.text_input("Enter your question:")
    if question:
        matching_docs = vectordb.similarity_search(question)
        input_data = {'question': question, 'input_documents': matching_docs}
        answer = qa_chain.invoke(input=input_data)
        st.session_state.past.append(question)
        st.session_state.generated.append(answer['output_text'])
        unanswered_phrases = ["I don't know.", "I don't have much information", "I don't have any information", "I'm unable to provide an answer"]  # Add more phrases if needed
        if any(phrase in answer.get('output_text', '') for phrase in unanswered_phrases):
            unanswered_question = f"Unanswered question: {question}\n"
            log_file_path = "log.txt"
            with open(log_file_path, "a") as log_file:
                log_file.write(unanswered_question)
            st.error("Sorry, I'm unable to provide an answer to this question at the moment")
        else:
            for i in range(len(st.session_state['generated'])- 1, -1, -1):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                message(st.session_state['generated'][i], key=str(i))
                
                for document in answer['input_documents']:
                    if 'source' in document.metadata:
                        st.write(f"Source: {document.metadata['source']}")
                        break

if __name__ == "__main__":
    main()
