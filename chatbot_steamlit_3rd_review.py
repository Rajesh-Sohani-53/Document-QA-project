import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import redis
import pickle
import numpy as np

# Page configuration
st.set_page_config(
    page_title="PDF QA Chatbot",
    page_icon="📚",
    layout="wide"
)

# Title
st.title("📚 PDF Question-Answering Chatbot")
st.markdown("Ask questions about your uploaded PDF document")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if 'system_ready' not in st.session_state:
    st.session_state.system_ready = False


# Helper function: Cosine similarity
def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0


# Helper function: Find semantically similar cached query
def find_similar_cached_query(query, embedding_model, redis_client, threshold=0.85):
    """
    Search for a semantically similar cached query.
    Returns (cached_query, cached_answer, similarity_score) or (None, None, 0)
    """
    query_embedding = embedding_model.embed_query(query)

    max_similarity = 0
    best_cached_query = None
    best_cached_answer = None

    all_keys = redis_client.keys("query:*")

    for key in all_keys:
        if key.endswith(b":embedding"):
            continue

        cached_query = key.decode('utf-8').replace("query:", "", 1)

        embedding_key = f"query:{cached_query}:embedding".encode('utf-8')
        cached_embedding_data = redis_client.get(embedding_key)

        if cached_embedding_data:
            cached_embedding = pickle.loads(cached_embedding_data)
            similarity = cosine_similarity(query_embedding, cached_embedding)

            if similarity > max_similarity:
                max_similarity = similarity
                best_cached_query = cached_query
                best_cached_answer = redis_client.get(key).decode('utf-8')

    if max_similarity >= threshold:
        return best_cached_query, best_cached_answer, max_similarity
    else:
        return None, None, 0


# Initialize system
@st.cache_resource
def initialize_system():
    """Initialize the QA system components"""
    try:
        # Load embedding model
        embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # Load vector database
        vectordb = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)

        # Create retriever
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # Initialize LLM
        llm = ChatOpenAI(model_name="gpt-3.5-turbo")

        # Define prompt
        prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the following context to answer the question.
If not found in the document, say "I don't know."

Previous conversation:
{chat_history}

Context:
{context}

Question:
{input}

Answer:
""")

        # Build the chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        qa_chain = create_retrieval_chain(retriever, document_chain)

        # Connect to Redis
        r = redis.StrictRedis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=False
        )

        return embedding_model, qa_chain, r, True

    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        return None, None, None, False


# Load system
with st.spinner("🔄 Loading system..."):
    embedding_model, qa_chain, redis_client, success = initialize_system()
    if success:
        st.session_state.system_ready = True
        st.success("✅ System loaded successfully!")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This chatbot uses:
    - **FAISS** for vector storage
    - **SentenceTransformer** for embeddings
    - **OpenAI GPT-3.5** for answers
    - **Redis** for semantic caching
    - **LangChain** for orchestration
    """)

    st.divider()

    st.header("📊 Statistics")
    if st.session_state.system_ready:
        try:
            cache_size = len(redis_client.keys("query:*")) // 2  # Divide by 2 (answer + embedding)
            st.metric("Cached Queries", cache_size)
        except:
            st.metric("Cached Queries", "N/A")

    st.metric("Messages", len(st.session_state.messages))

    st.divider()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.rerun()

    if st.button("🔄 Clear Cache"):
        if st.session_state.system_ready:
            try:
                redis_client.flushdb()
                st.success("Cache cleared!")
            except Exception as e:
                st.error(f"Error clearing cache: {str(e)}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metadata" in message:
            with st.expander("ℹ️ Details"):
                st.caption(message["metadata"])

# ✅ CHAT INPUT SECTION (ADD THIS PART)
if prompt := st.chat_input("Ask a question about your document..."):
    if not st.session_state.system_ready:
        st.error("⚠️ System not ready. Please check if faiss_index exists and Redis is running.")
    else:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                # Step 1: Check exact match
                exact_cache_key = f"query:{prompt}".encode('utf-8')
                cached_answer = redis_client.get(exact_cache_key)

                metadata = ""

                if cached_answer:
                    answer = cached_answer.decode('utf-8')
                    metadata = "✅ Retrieved from exact match cache"
                    st.markdown(answer)
                else:
                    # Step 2: Check semantic similarity
                    similar_query, similar_answer, similarity_score = find_similar_cached_query(
                        prompt, embedding_model, redis_client, threshold=0.85
                    )

                    if similar_query:
                        answer = similar_answer
                        metadata = f"✅ Retrieved from semantic cache\n\nSimilar query: '{similar_query}'\n\nSimilarity: {similarity_score:.2%}"
                        st.markdown(answer)
                    else:
                        # Step 3: Generate new answer
                        result = qa_chain.invoke({
                            "input": prompt,
                            "chat_history": st.session_state.memory.chat_memory.messages
                        })

                        answer = result["answer"]

                        # Cache the answer
                        query_embedding = embedding_model.embed_query(prompt)
                        redis_client.setex(exact_cache_key, 604800, answer.encode('utf-8'))
                        embedding_key = f"query:{prompt}:embedding".encode('utf-8')
                        redis_client.setex(embedding_key, 604800, pickle.dumps(query_embedding))

                        metadata = "🔍 Generated from document retrieval + LLM\n\n💾 Cached for future use"
                        st.markdown(answer)

                        # Update memory
                        st.session_state.memory.chat_memory.add_user_message(prompt)
                        st.session_state.memory.chat_memory.add_ai_message(answer)

                # Show metadata
                with st.expander("ℹ️ Details"):
                    st.caption(metadata)

        # Add assistant message to chat
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "metadata": metadata
        })

# Footer
st.divider()
st.caption("💡 Tip: Ask questions in different ways - semantic caching will find similar answers!")
