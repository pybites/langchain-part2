import chromadb
from dotenv import dotenv_values
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

settings = dotenv_values(".env")


def load_youtube_video(youtube_url):

    loader = YoutubeLoader.from_youtube_url(youtube_url=youtube_url, add_video_info=True)

    result = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)

    docs = text_splitter.split_documents(result)


    embeddings = OpenAIEmbeddings()
    new_client = chromadb.EphemeralClient()
    openai_lc_client = Chroma.from_documents(
        docs, embeddings, client=new_client, collection_name="openai_collection"
    )
    return openai_lc_client, result


def query_video(openai_lc_client, query):
    llm = OpenAI(temperature=0, openai_api_key=settings["OPENAI_API_KEY"], verbose=True)

    docs = openai_lc_client.similarity_search(query, k=3)

    chain = load_qa_chain(llm=llm, chain_type="stuff")

    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=query)
        print(cb)

    return response


def create_conversation_chain(db):

    # TODO: reactivate persistence
    # embedding = OpenAIEmbeddings()
    # persist_directory = 'docs/chroma/'
    # db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    retriever=db.as_retriever()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    llm = ChatOpenAI(temperature=0)

    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )

    return qa


def query_qa(qa, query):
    result = qa({"question": query})

    return result
