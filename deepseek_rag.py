# a streamlit rag app using deepseek r1 through ollama, 
import streamlit as st
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import tempfile
import os


st.set_page_config(page_title= 'Rag app', page_icon= ":computer:", ) #layout= 'wide' )

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #3A3A3A !important;
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A !important;
        color: #E0E0E0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040 !important;
        color: #F0F0F0 !important;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    
    /* Text Color Fix */
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important;
    }
    
    .stFileUploader {
        background-color: #1E1E1E;
        border: 1px solid #3A3A3A;
        border-radius: 5px;
        padding: 15px;
    }
    
    h1, h2, h3,h6 {
        color: #FFFFFF !important;
    }
            
     div[data-testid="stTextInput"] label {     
        color: #FFFFFF !important;  /* White Text */
        font-size: 18px; /* Increase font size */
        font-weight: bold; /* Make text bold */
    }
    </style>

    """, unsafe_allow_html=True)





def load_pdf_documents(uploaded_file):
    # Create a temporary file to handle the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        try:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
            
            document_loader = PyPDFLoader(tmp_file_path)
            docs = document_loader.load()
            return docs
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
            return None
        finally:
            # Clean up the temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass


def create_chunk_documents(raw_documents):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
    )
    document = text_splitter.split_documents(raw_documents)
    return document


def create_vector_store(document):
    
    # create embeddings and vector store
    embeddings = OllamaEmbeddings(model = "deepseek-r1:1.5b")

    try:
        db = FAISS.from_documents(document, embeddings)
        st.write("Vector store created successfully.")
    except Exception as e:
        st.write(f"Error while creating vector store: {e}")

    return db


# UI Configuration
def main():
    st.title(":mag: DocSearch ")
    st.markdown("### Your Document Assistant")
    st.markdown("###### Quickly find and retrieve your documents with AI-powered search.")
    st.markdown("---")

    # Initialize session state for vector store
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None

    uploaded_file = st.file_uploader(
        "Upload PDF Document ",
        type="pdf",
        help="Select a PDF document for analysis",
        accept_multiple_files=False
    )

    if uploaded_file and st.session_state.vector_store is None:
        with st.spinner('Processing document...'):
            # load the document
            raw_documents = load_pdf_documents(uploaded_file)
            if raw_documents:
                documents = create_chunk_documents(raw_documents)
                st.session_state.vector_store = create_vector_store(documents)
                if st.session_state.vector_store:
                    st.write("Document successfully loaded and indexed.")
                    st.write("You can now start searching for your documents.")
    if st.session_state.vector_store:
        # Add search functionality
        query = st.text_input("Enter your search query:")
        if query:
            with st.spinner('Searching...'):
                try:
                    # Search the vector store
                    docs = st.session_state.vector_store.similarity_search(query, k=3)
                    
                    # Create context from retrieved documents
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    # Create prompt template
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a helpful assistant. Use the following context to answer the question.\n\nContext: {context}"),
                        ("user", "{question}")
                    ])
                    
                    # Get response from LLM
                    response = llm_engine.invoke(
                        prompt.format_messages(context=context, question=query)
                    )
                    
                    # Display response
                    st.write("Answer:", response.content)
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")

    return None



# using explicit base url
llm_engine = ChatOllama(
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0.3
)



# response_content = response.content  # Extract content string

# # Remove the "thinking" part and keep only the actual reply
# cleaned_response = re.sub(r"<think>.*?</think>\s*", "", response_content, flags=re.DOTALL)

# st.write(cleaned_response)  # Display the cleaned reply

if __name__ == "__main__":
    main()

