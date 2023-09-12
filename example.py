from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=2uaTPmNvH0I", add_video_info=True)

result = loader.load()

# print(type(result))
# print(f"Found video from {result[0].metadata['author']} that is {result[0].metadata['length']} seconds long")
# print("")
# print(result)

llm = OpenAI(temperature=0, openai_api_key="")

# chain = load_summarize_chain(llm, chain_type="stuff", verbose=False)
# chain.run(result)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)

texts = text_splitter.split_documents(result)
# print(len(texts))

chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
result = chain.run(texts[:4])

# print(result)


import os

import pinecone
from langchain.vectorstores import Pinecone

pinecone.init(
    api_key="35a5406a-c2da-489f-ac4a-4adc17b92951",  
    environment="gcp-starter"
)

from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model_name="ada", openai_api_key="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")

index_name = "example"
search = Pinecone.from_documents(texts, embeddings, index_name=index_name)
