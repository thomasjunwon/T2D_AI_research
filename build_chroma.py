import json
import chromadb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
JSON_PATH = BASE_DIR / "dia_guidelines.json"
CHROMA_PATH = BASE_DIR / "chroma_dia"


def build_ada_chroma(json_path=JSON_PATH, persist_path=CHROMA_PATH):
    client = chromadb.PersistentClient(path=persist_path)
    
    try:
        client.delete_collection("dia_guidelines")
        print("Deleted existing collection.")
    except Exception:
        print("No existing collection to delete.")
        
    collection = client.get_or_create_collection(name="dia_guidelines")

    with open(json_path, "r", encoding="utf-8") as f:
        guidelines = json.load(f)

    ids = [g["id"] for g in guidelines]
    documents = [g["text"] for g in guidelines]
    metadatas = [
        {
            "variable": g["variable"],
            "reference": g["reference"]
        }
        for g in guidelines
    ]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    print("CHROMA PATH:", CHROMA_PATH)
    print("COLLECTION COUNT:", collection.count())
    print(f"Added {len(ids)} guideline chunks.")


if __name__ == "__main__":
    build_ada_chroma()