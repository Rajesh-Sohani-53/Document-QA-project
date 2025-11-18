# step 9:
# 1. we will be load chunks with embeddings data
# 2. then create a FAISS vectore store
# 3. We use the langchain to make it easy to search
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

import pickle

from setuptools._distutils.command.check import check

# 1 open the file
file=open("pdf_embeddings.pkl","rb")
data=pickle.load(file)
file.close()


chunks=data["chunks"]
embeddings=data["embeddings"]
print("✅ File loaded successfully with", len(chunks), "chunks.")
# we create the embedding model
embedding_model=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# also we need a FAISS database
#it store the embeddings,chunks and   embedding model
vectordb = FAISS.from_texts(texts=chunks, embedding=embedding_model)
print("🧠 Vector database created successfully!")
# at here the database is created sucesfully
# now we need a retrival
# this is our search tool
retriver=vectordb.as_retriever(search_kwargs={"k": 3})
query=input("\n Ask the question")
# get the result
results = retriver.invoke(query)
print("\n🔍 Top results:\n")
for i, doc in enumerate(results, start=1):
    print(f"{i}. {doc.page_content[:400]}...")
    print("\n------------------------\n")

