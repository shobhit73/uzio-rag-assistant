
import os
import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# Load documents
@st.cache_resource
def load_documents(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            loader = UnstructuredPDFLoader(path)
        elif filename.endswith(".txt") or filename.endswith(".md"):
            loader = TextLoader(path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            continue
        docs.extend(loader.load())
    return docs

# Build RAG pipeline
@st.cache_resource
def create_rag_pipeline():
    docs = load_documents("docs")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.5, "max_length": 500}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

# Streamlit App
st.set_page_config(page_title="UZIO Assistant", layout="wide")
st.title("🤖 UZIO Help Center Assistant (Cloud Version)")

query = st.text_input("Ask a question about UZIO Help Center:")
if query:
    with st.spinner("Searching and generating answer..."):
        qa_chain = create_rag_pipeline()
        result = qa_chain({"query": query})
        st.success("Answer:")
        st.write(result["result"])

        with st.expander("🔍 Source Chunks"):
            for i, doc in enumerate(result["source_documents"]):
                st.markdown(f"**Source {i+1}:**")
                st.write(doc.page_content)
