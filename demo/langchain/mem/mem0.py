import os

import dotenv
from mem0 import Memory, MemoryClient
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore, Chroma

dotenv.load_dotenv()

# zhipu GLM
llm = ChatOpenAI(
    base_url="https://open.bigmodel.cn/api/paas/v4",
    model="glm-4.5-flash",
    max_tokens=32768,
    temperature=0.2,
    api_key=os.environ["ZHIPU_API_KEY"],
)


# Set embeddings
# https://ai.gitee.com/serverless-api#embedding-rerank
# 100 api calls per day, free tier
embd = OpenAIEmbeddings(
    base_url="https://ai.gitee.com/v1",
    model="Qwen3-Embedding-8B",  # 4096
    api_key=os.environ["GITEE_API_KEY"],
    dimensions=1536,
    check_embedding_ctx_length=False,
    chunk_size=1000,
)

# vector_store = Chroma(
#    persist_directory="./rag_chroma_db",
#    embedding_function=embd,
#    collection_name="mem0",  # Required collection name
# )

vector_store = InMemoryVectorStore(embedding=embd)


def langchain_mem0():
    # Pass the initialized model to the config
    # RAG
    config = {
        "llm": {"provider": "langchain", "config": {"model": llm}},
        "embedder": {"provider": "langchain", "config": {"model": embd}},
        # "vector_store": {
        #    "provider": "langchain",
        #    "config": {"client": vector_store},
        # },  # default qdrant
        # "graph_store": None,
        # "ranker": None,
    }

    m = Memory.from_config(config)
    messages = [
        {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
        {"role": "assistant", "content": "How about thriller movies? They can be quite engaging."},
        {
            "role": "user",
            "content": "I'm not a big fan of thriller movies but I love sci-fi movies.",
        },
        {
            "role": "assistant",
            "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future.",
        },
    ]
    m.add(
        messages,
        user_id="alice",
        agent_id="achatbot-agent1",
        run_id="bot1",
        metadata={"category": "movies"},
    )
    res = m.search(
        query="Can you suggest some good sci-fi movies?",
        user_id="alice",
        agent_id="achatbot-agent1",
        run_id="bot1",
        limit=3,
    )
    print(res)


def remote_api_mem0():
    m = MemoryClient(
        api_key=os.environ["MEM0_API_KEY"],
        host=None,  # https://api.mem0.ai
    )

    messages = [
        {"role": "user", "content": "I'm planning to watch a movie tonight. Any recommendations?"},
        {"role": "assistant", "content": "How about thriller movies? They can be quite engaging."},
        {
            "role": "user",
            "content": "I'm not a big fan of thriller movies but I love sci-fi movies.",
        },
        {
            "role": "assistant",
            "content": "Got it! I'll avoid thriller recommendations and suggest sci-fi movies in the future.",
        },
    ]
    m.add(
        messages,
        user_id="alice",
        agent_id="achatbot-agent1",
        run_id="bot1",
        metadata={"category": "movies"},
    )

    filters = {
        "OR": [
            {"user_id": "alice"},
            {"agent_id": "achatbot-agent1"},
            {"run_id": "bot1"},
        ]
    }
    res = m.search(
        query="Can you suggest some good sci-fi movies?",
        filters=filters,
        version="v2",
        top_k=3,
        threshold=0.1,
        output_format="v1.1",
    )
    print(res)


"""
MOD=local python -m demo.langchain.mem.mem0
MOD=api python -m demo.langchain.mem.mem0
"""
if __name__ == "__main__":
    mod = os.getenv("MOD", "local")
    if mod == "api":
        remote_api_mem0()
    else:
        langchain_mem0()
 