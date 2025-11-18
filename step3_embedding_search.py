# step1: import important libraries
import pickle
import numpy
import numpy as np
from sentence_transformers import SentenceTransformer

# step 2: open the file
file=open("pdf_embeddings.pkl","rb")
data=pickle.load(file)
file.close()

print("kyes in the file ",list(data.keys()))
#print("chunks",data["chunks"])
#print("lenght of the chunks",len(data["chunks"]))
# step 3 convert the embedding to the numpy array
embeddings=np.array(data["embeddings"])
chunks=np.array(data["chunks"])


#print("Shape of the embeddings ",embeddings.shape)
# step 4: load the sentence tranformer
# we are using the sentence transformer for converting the user query to the embedding so we need sentence transformer
model=SentenceTransformer("all-MiniLM-L6-v2",device="cpu")
# after making the sentence transformer we need the query
# step:5 create the query vector
query="what is the bias-variance tradeoff ?"
q_vec=model.encode(query)
#print("Lenght of Query vector is ",len(q_vec))
#print("First 6 number of vector is ",q_vec[:6])
# step 6: now we find the cosine similarty to
# compute how two vectors are close
def cosine_sim(a,b):
    return  np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

# print("Self-similarity (should be ~1.0):", cosine_sim(q_vec, q_vec))
# step 7:
# now we compare the query vector with each of the chunk vector in the embedding
similarities=[]
for i,emb in enumerate(embeddings):
    sim=cosine_sim(q_vec,emb)
    similarities.append((i,sim))

print("Sample similarities (first 5):", [s for _, s in similarities[:5]])

# sort by score descending
similarities.sort(key=lambda x: x[1], reverse=True)
#step *: Goal: find the top N most similar chunks and display them.
# show top 3
for rank in range(3):
    idx, score = similarities[rank]
    print(f"\nRank {rank+1} — chunk {idx+1} — score {score:.4f}\n")
    print(chunks[idx][:400])  # show first 400 characters
    print("\n" + "-"*40 + "\n")

