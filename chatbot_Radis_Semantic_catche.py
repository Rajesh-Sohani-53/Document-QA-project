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

# Load embedding model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Load vector database
vectordb = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)
print("✅ Database loaded successfully")

# Create retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
print("✅ Retriever is created")

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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
    decode_responses=False  # Changed to False to store binary data (pickled embeddings)
)
print("✅ Connected to Redis cache\n")


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
def find_similar_cached_query(query, threshold=0.85):
    """
    Search for a semantically similar cached query.
    Returns (cached_query, cached_answer, similarity_score) or (None, None, 0)
    """
    # Get embedding for incoming query
    query_embedding = embedding_model.embed_query(query)

    max_similarity = 0
    best_cached_query = None
    best_cached_answer = None

    # Get all cached query keys
    all_keys = r.keys("query:*")

    for key in all_keys:
        if key.endswith(b":embedding"):
            continue  # Skip embedding keys

        cached_query = key.decode('utf-8').replace("query:", "", 1)

        # Get cached embedding
        embedding_key = f"query:{cached_query}:embedding".encode('utf-8')
        cached_embedding_data = r.get(embedding_key)

        if cached_embedding_data:
            cached_embedding = pickle.loads(cached_embedding_data)

            # Calculate similarity
            similarity = cosine_similarity(query_embedding, cached_embedding)

            if similarity > max_similarity:
                max_similarity = similarity
                best_cached_query = cached_query
                best_cached_answer = r.get(key).decode('utf-8')

    if max_similarity >= threshold:
        return best_cached_query, best_cached_answer, max_similarity
    else:
        return None, None, 0


# Chat loop
while True:
    query = input("You: ")
    if query.lower() == "exit":
        print("Goodbye!")
        break

    # Step 1: Check for exact match in Redis cache
    exact_cache_key = f"query:{query}".encode('utf-8')
    cached_answer = r.get(exact_cache_key)

    if cached_answer:
        print(f"\n🤖 (exact match from cache): {cached_answer.decode('utf-8')}\n")
        continue

    # Step 2: Check for semantically similar cached query
    print("🔍 Checking for similar cached queries...")
    similar_query, similar_answer, similarity_score = find_similar_cached_query(query, threshold=0.85)

    if similar_query:
        print(f"✅ Found similar query: '{similar_query}' (similarity: {similarity_score:.2f})")
        print(f"\n🤖 (from semantic cache): {similar_answer}\n")
        continue

    # Step 3: No cache hit → Run retrieval + LLM
    print("🔍 Not in cache, running retrieval + LLM...")
    result = qa_chain.invoke({"input": query, "chat_history": memory.chat_memory.messages})

    answer = result["answer"]

    # Step 4: Store in Redis cache with embedding
    query_embedding = embedding_model.embed_query(query)

    # Store answer
    r.setex(exact_cache_key, 604800, answer.encode('utf-8'))  # 7 days

    # Store embedding
    embedding_key = f"query:{query}:embedding".encode('utf-8')
    r.setex(embedding_key, 604800, pickle.dumps(query_embedding))  # 7 days

    print(f"💾 Answer and embedding cached in Redis")

    # Update conversation memory
    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(answer)

    print(f"\n🤖: {answer}\n")
