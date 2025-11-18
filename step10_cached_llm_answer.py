from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
#adding the new function
from langchain.memory import ConversationBufferMemory
from langchain.globals import set_llm_cache,get_llm_cache
from langchain.cache import InMemoryCache
import pickle

file=open("pdf_embeddings.pkl","rb")
data=pickle.load(file)
file.close()

chunks=data["chunks"]

print("file loaded sucessfully with the chunk size is ",len(chunks))
#creating the embedding model
embedding_model=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#creatign the vectore store
vectordb=FAISS.from_texts(texts=chunks,embedding=embedding_model)
#creating the retruiver
retriever=vectordb.as_retriever(search_kwargs={"k":3})
print(" the retriever is ready")

# now the llm
llm=ChatOpenAI(model_name="gpt-3.5-turbo")

memory=ConversationBufferMemory(memory_key="chat_history",return_masage=True)
# 6️⃣ Define prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the following context to answer the question.
If not found in the document, say "I don’t know."

Previous conversation:
{chat_history}

Context:
{context}

Question:
{input}

Answer:
""")

# build chat with retriver memory
document_chain=create_stuff_documents_chain(llm,prompt)
qa_chain=create_retrieval_chain(retriever,document_chain)
print("\n💬 Chatbot ready! Type 'exit' to quit.\n")
# simple loop
while True :
    query=input("you :")
    if query.lower()=="exit":
        print("Goodbuy")
        break

    result=qa_chain.invoke({"input":query,"chat_history":memory.chat_memory.messages})
    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(result["Answer"])
    print("\n🤖:", result["answer"], "\n")