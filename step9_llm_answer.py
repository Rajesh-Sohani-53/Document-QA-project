# step9_llm_answer.py
# ✅ WORKS WITH LATEST (2025) LANGCHAIN VERSION
# ✅ Works for LangChain 2025+ (community + openai modules)

# ✅ FINAL STABLE IMPORTS for LangChain 0.2.16 setup

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
import pickle

file=open("pdf_embeddings.pkl","rb")
data=pickle.load(file)
file.close()

chunks=data["chunks"]
print("Data loader sucessfuly with chunk size",len(chunks))

# embedding model
Embedding_model=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# fassi vector database
vectordb=FAISS.from_texts(chunks,embedding=Embedding_model)
#retriver
retriver=vectordb.as_retriever(search_kwargs={"k": 3})

print("Retriver is ready ")
# gpt model
llm=ChatOpenAI(model_name="gpt-3.5-turbo")
prompt=ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the following context to answer the question clearly and briefly.
If the answer is not found in the document, say "Sorry, I couldn’t find this in the document."

Context:
{context}

Question:
{input}

Answer:
""")
# now we combine the retriver with reg means llm
document_chain=create_stuff_documents_chain(llm,prompt)
qa_chain=create_retrieval_chain(retriver,document_chain)
query=input("\n Enter the query")
result=qa_chain.invoke({"input":query})
# 9️⃣ Show final answer
print("\n🤖 Answer:\n")
print(result["answer"])


