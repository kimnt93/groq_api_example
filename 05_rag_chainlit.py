"""
This script demonstrates how to build and serve a standard retriever as a SaaS model to end customers using a chatbot interface.
We utilize the Chainlit library, which is built on top of Streamlit—a popular Python app framework for quick Data Science and ML demos.
Chainlit provides a user interface similar to ChatGPT.

To install the required libraries, run the following command:
```
pip install -q chainlit
```

After installing the requirements, you can run the application with:
```
chainlit run 05_rag_chainlit.py
```
"""
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.core.callbacks.base import CallbackManager
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.node_parser import SentenceSplitter

import os

import chainlit as cl


@cl.on_chat_start
async def factory():
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    llm = Groq(model="llama3-8b-8192", api_key=os.environ["GROQ_API_KEY"])

    service_context = ServiceContext.from_defaults(
        embed_model=embed_model,
        llm=llm,
        text_splitter=SentenceSplitter(chunk_size=1024),
        chunk_size=512,
        chunk_overlap=20,
        transformations=[SentenceSplitter(chunk_size=1024)],
        context_window=3500,
        num_output=512,
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
    )

    documents = SimpleWebPageReader(html_to_text=True).load_data([
        "https://docs.llamaindex.ai/en/stable/", 
        "https://docs.llamaindex.ai/en/stable/getting_started/concepts/"
    ])
    index = VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=True)

    chat_engine = index.as_chat_engine(
        service_context=service_context,
        similarity_top_k=5
    )

    cl.user_session.set("chat_engine", chat_engine)


@cl.on_message
async def main(message: cl.Message):
    chat_engine = cl.user_session.get("chat_engine")  
    response = await cl.make_async(chat_engine.query)(message.content)

    response_message = cl.Message(content="")

    for token in response.response:
        await response_message.stream_token(token=token)

    await response_message.send()
