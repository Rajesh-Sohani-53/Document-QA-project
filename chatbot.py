from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb=FAISS.load_local("faiss_index",embeddings=embedding_model,allow_dangerous_deserialization=True)
print("Database load sucessfully ")
#create the retriver
retirver=vectordb.as_retriever(search_kwargs={"k": 3})
print("retirver is created ")

# initilization of llm
llm=ChatOpenAI(model_name="gpt-3.5-turbo")
# initilization of memory
memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
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

# build the chain
document_chain=create_stuff_documents_chain(llm,prompt)
qa_chain=create_retrieval_chain(retirver,document_chain)
# chat loop
while True:
    query = input("You: ")
    if query.lower() == "exit":
        print("Goodbye!")
        break

    result = qa_chain.invoke({"input": query, "chat_history": memory.chat_memory.messages})
    # update the memory
    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(result["answer"])
    print(f"\n🤖: {result['answer']}\n")