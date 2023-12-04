import os
import sys
from typing import List
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings


def create_index(file_path: str) -> None:

    documents = load_documents(file_path)
    if not documents:
        print("No documents found or unable to load documents.")
        return
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=128
    )

    splits = text_splitter.split_documents(documents)

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    persist_directory = 'db'

    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    vectordb.persist()


def load_documents(path_or_url: str) -> List[str]:
    """
    path_or_url: path to the PDF file or directory, or a web URL.
    return: document contents.
    """
    if os.path.isfile(path_or_url) and path_or_url.lower().endswith('.pdf'):
        # single PDF file
        return PyPDFLoader(path_or_url).load()

    elif os.path.isdir(path_or_url):
        # multi-pdfs
        return [doc for f in os.listdir(path_or_url) if f.endswith('.pdf')
                for doc in PyPDFLoader(os.path.join(path_or_url, f)).load()]

    elif path_or_url.lower().startswith(('http://', 'https://')):
        # web URL
        return WebBaseLoader(path_or_url).load()

    else:
        print("Invalid input. Please provide a valid PDF file, directory, or web URL.")
        return []


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python index.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]
    create_index(directory_path)
