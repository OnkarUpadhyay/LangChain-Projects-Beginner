import os
import time

# 1. FORCE PYTHON TO USE A MIRROR BYPASSING ISP BLOCKS
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. NOW IMPORT LANGCHAIN
from langchain_huggingface import HuggingFaceEmbeddings

def main():
    print("==================================================")
    print("Initiating Local Embedding Model via Mirror...")
    print("Model: sentence-transformers/all-MiniLM-L6-v2")
    print("==================================================")
    
    start_time = time.time()

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        end_time = time.time()
        duration = round(end_time - start_time, 2)
        
        print("\n✅ SUCCESS!")
        print(f"The model was successfully loaded in {duration} seconds.")
        print("You can now remove the mirror line from app.py if you want.")
        
    except Exception as e:
        print("\n❌ ERROR OCCURRED:")
        print(e)

if __name__ == "__main__":
    main()