from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# ---------- CONFIG ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "db", "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------- LOAD EMBEDDINGS ----------
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

# ---------- LOAD VECTOR STORE ----------
def load_vectorstore(embeddings):
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )

# ---------- MAIN ----------
if __name__ == "__main__":
    # 1. Load embeddings
    embeddings = load_embeddings()

    # 2. Load vectorstore
    vectorstore = load_vectorstore(embeddings)

    # 3. Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # 4. Query
    query = "What was NVIDIA's first graphics accelerator called?"

    docs = retriever.invoke(query)

    print("\nUser Query:", query)
    print("\n--- Retrieved Documents ---\n")

    for i, doc in enumerate(docs, 1):
        print(f"Document {i}:")
        print(doc.page_content)
        print("-" * 50)


# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"