import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
file=open("pdf_embeddings.pkl","rb")
data=pickle.load(file)
file.close()

chunks=data["chunks"]
print("✅ Chunk found and the length of chunk is ",len(chunks))
embedding_model=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
print("✅ embedding_model created ")
vectordb=FAISS.from_texts(texts=chunks,embedding=embedding_model)
vectordb.save_local("faiss_index")
print("✅ Vector Database created ")
print("✅ Vector store saved to 'faiss_index' folder")