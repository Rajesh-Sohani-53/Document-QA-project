import chunk
import pickle
import numpy as numpy
file = open("pdf_embeddings.pkl", "rb")

data=pickle.load(file)
# close the file after reading
file.close()

# now the data is like the box
# that has two things insede : chunks and embeddings
chunks=data["chunks"]
embeddings=data["embeddings"]
# print someting to check
print("file open sucessfully")
print("how many chunks ",len(chunks))
print("\n here is one small example text :\n")
print(chunks[0])