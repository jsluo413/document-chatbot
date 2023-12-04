import os
from langchain import hub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
load_dotenv()


def create_conversation() -> ConversationalRetrievalChain:

    persist_directory = 'db'

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=False
    )
    prompt = hub.pull("rlm/rag-prompt")

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_type="mmr"),
        memory=memory,
        get_chat_history=lambda h: h,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    return qa
