import os
from typing import Optional
import dotenv
from chromadb.api.client import Client as ClientCreator
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE
from chromadb.config import Settings, System

dotenv.load_dotenv()


def openai():
    from openai import OpenAI

    # 100 api calls per day, free tier ..................NO!!!! ❄️
    client = OpenAI(
        base_url="https://ai.gitee.com/v1",
        api_key=os.getenv("GITEE_API_KEY"),
        # default_headers={"X-Failover-Enabled":"true"},
    )

    response = client.embeddings.create(
        input="Today is a sunny day and I will get some ice cream.",
        model="Qwen3-Embedding-8B",
        dimensions=1024,
    )
    print(response)


class CustomClientCreator(ClientCreator):
    # region Initialization
    def __init__(
        self,
        tenant: Optional[str] = DEFAULT_TENANT,
        database: Optional[str] = DEFAULT_DATABASE,
        settings: Settings = Settings(),
    ) -> None:
        super().__init__(tenant=tenant, database=database, settings=settings)

    def get_max_batch_size(self) -> int:
        return 10  # some server api limit batch size


def openai_embedding():
    from langchain_openai import OpenAIEmbeddings

    os.environ["USER_AGENT"] = "achatbot-demo"
    ### from langchain_cohere import CohereEmbeddings

    # 100 api calls per day, free tier ..................NO!!!! ❄️
    # TODO:use open source embedding model to replace it with vllm/sglang
    # Set embeddings
    embd = OpenAIEmbeddings(
        base_url="https://ai.gitee.com/v1",
        model="Qwen3-Embedding-8B",
        api_key=os.getenv("GITEE_API_KEY"),
        dimensions=1024,
        check_embedding_ctx_length=False,
        chunk_size=1000,
    )

    ### Build Index
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.vectorstores import Chroma

    # Docs to index
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    # Load
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    # Split
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    print(f"Document splits: {len(doc_splits)}")

    # Add to vectorstore
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        persist_directory="./rag_chroma_db",  # sqlite(row) file path or duckdb(column) file path
        embedding=embd,
        client=CustomClientCreator(),  # use custom client with batch utils
    )
    retriever = vectorstore.as_retriever()
    print(retriever)
    vectorstore.persist()


def retrieval():
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    os.environ["USER_AGENT"] = "achatbot-demo"
    ### from langchain_cohere import CohereEmbeddings

    # 100 api calls per day, free tier ..................NO!!!! ❄️
    # TODO:use open source embedding model to replace it with vllm/sglang
    # Set embeddings
    embd = OpenAIEmbeddings(
        base_url="https://ai.gitee.com/v1",
        model="Qwen3-Embedding-8B",
        api_key=os.getenv("GITEE_API_KEY"),
        dimensions=1024,
        check_embedding_ctx_length=False,
        chunk_size=1000,
    )

    vectorstore = Chroma(
        collection_name="rag-chroma",
        embedding_function=embd,
        persist_directory="./rag_chroma_db",  # sqlite(row) file path or duckdb(column) file path
        # client=CustomClientCreator(),  # use custom client with batch utils
    )
    retriever = vectorstore.as_retriever()
    print(retriever)


def inmemory():
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_openai import OpenAIEmbeddings

    ### Build Index
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import WebBaseLoader

    # Docs to index
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        # "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        # "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    # Load
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    # Split
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    print(f"Document splits: {len(doc_splits)}")

    os.environ["USER_AGENT"] = "achatbot-demo"
    ### from langchain_cohere import CohereEmbeddings

    # 100 api calls per day, free tier ..................NO!!!! ❄️
    # TODO:use open source embedding model to replace it with vllm/sglang
    # Set embeddings
    embd = OpenAIEmbeddings(
        base_url="https://ai.gitee.com/v1",
        model="Qwen3-Embedding-8B",
        api_key=os.getenv("GITEE_API_KEY"),
        dimensions=1024,
        check_embedding_ctx_length=False,
        chunk_size=1000,
    )

    vectorstore = InMemoryVectorStore(embedding=embd)
    while len(doc_splits) > 0:
        documents = doc_splits[:10]
        print(f"{len(documents)=}")
        # vectorstore.add_documents(documents=documents)
        doc_splits = doc_splits[10:]
    retriever = vectorstore.as_retriever()


if __name__ == "__main__":
    # openai()
    # openai_embedding()
    inmemory()
