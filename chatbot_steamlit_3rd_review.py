import streamlit as st
import redis
import pickle
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# ----------------------------- PAGE SETTINGS -----------------------------
st.set_page_config(page_title="PDF QA Chatbot", page_icon="📚", layout="wide")

st.title("📚 PDF Question-Answering Chatbot")
st.markdown("Ask questions about your uploaded PDF document.")

# ----------------------------- SESSION STATE -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "system_ready" not in st.session_state:
    st.session_state.system_ready = False


# ----------------------------- COSINE SIMILARITY -----------------------------
def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# ----------------------------- SEMANTIC CACHE -----------------------------
def find_semantic_cache(query, embedding_model, redis_client, threshold=0.85):
    query_emb = embedding_model.embed_query(query)

    max_sim = 0
    best_query = None
    best_answer = None

    for key in redis_client.keys("query:*"):
        if key.endswith(b":embedding"):
            continue

        q = key.decode().replace("query:", "")
        emb_key = f"query:{q}:embedding".encode()

        emb_data = redis_client.get(emb_key)
        if not emb_data:
            continue

        cached_emb = pickle.loads(emb_data)
        sim = cosine_similarity(query_emb, cached_emb)

        if sim > max_sim:
            max_sim = sim
            best_query = q
            best_answer = redis_client.get(key).decode()

    if max_sim >= threshold:
        return best_query, best_answer, max_sim

    return None, None, 0


# ----------------------------- SYSTEM INITIALIZATION -----------------------------
@st.cache_resource
def init_system():
    try:
        # Embeddings
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # Load FAISS index
        vectordb = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        # Prompt
        prompt = ChatPromptTemplate.from_template(
            """
You are a PDF question-answering assistant. Use the provided context to answer.

Context:
{context}

Question:
{question}

Answer:
"""
        )

        # Retrieval Chain (new LC format)
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
        )

        # Redis caching
        redis_client = redis.StrictRedis(
            host="localhost",
            port=6379,
            db=0,
            decode_responses=False
        )

        return embeddings, chain, redis_client, True

    except Exception as e:
        st.error(f"Initialization error: {e}")
        return None, None, None, False


# ----------------------------- LOAD SYSTEM -----------------------------
with st.spinner("Loading system..."):
    embeddings, chain, redis_client, ok = init_system()
    if ok:
        st.session_state.system_ready = True
        st.success("System ready!")


# ----------------------------- SIDEBAR -----------------------------
with st.sidebar:
    st.header("ℹ️ About this chatbot")
    st.write("Uses FAISS, SentenceTransformer, OpenAI GPT, Redis semantic cache, LangChain 0.3")

    st.metric("Messages", len(st.session_state.messages))

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("Clear Cache"):
        redis_client.flushdb()
        st.success("Cache cleared!")


# ----------------------------- DISPLAY CHAT -----------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# ----------------------------- MAIN CHAT FUNCTION -----------------------------
if user_query := st.chat_input("Ask anything from your PDF..."):

    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # Exact match cache
            key = f"query:{user_query}".encode()
            cached = redis_client.get(key)

            if cached:
                answer = cached.decode()
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.stop()

            # Semantic match
            sq, sa, sim = find_semantic_cache(user_query, embeddings, redis_client)
            if sa:
                st.write(sa)
                st.caption(f"Retrieved from semantic cache (similarity {sim:.2%})")
                st.session_state.messages.append({"role": "assistant", "content": sa})
                st.stop()

            # Generate fresh answer
            ai_response = chain.invoke(user_query)
            answer = ai_response.content

            st.write(answer)

            # Cache answer + embedding
            emb = embeddings.embed_query(user_query)
            redis_client.setex(key, 604800, answer.encode())
            redis_client.setex(f"query:{user_query}:embedding".encode(), 604800, pickle.dumps(emb))

            st.session_state.messages.append({"role": "assistant", "content": answer})


st.divider()
st.caption("💡 Uses FAISS + semantic caching + latest LangChain")
