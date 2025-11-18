import  pickle
# open the file
import re
file=open("pdf_embeddings.pkl","rb")
data=pickle.load(file)
file.close()
# get the chunk from the file
chunks=data["chunks"]
print("file loaded sucessfully ")
print("total length of the chunk is ",len(chunks))

# now ask the use what they want to search
word=input("\n Enter the word do you want to search in the document ").lower()
# Step 3: clean the word (remove symbols)
word = re.sub(r'[^a-z0-9\s-]', '', word)
print("searching")
found=False
for i,text in enumerate(chunks):
    clean_text = re.sub(r'[^a-z0-9\s-]', '', text.lower())
    if word in text.lower():
        found=True
        print(f"👉 Found in chunk {i + 1}:")
        print(text[:400])  # show only first 400 characters
        print("\n-----------------------------\n")

# Step 5: If nothing found
if not found:
    print("⚠️ Word not found in any chunk.")

