# build_chroma.py

import json
import chromadb


def build_ada_chroma(json_path="dia_guidelines.json", persist_path="./chroma_dia"):
    client = chromadb.PersistentClient(path=persist_path)

    collection = client.get_or_create_collection(
        name="dia_guidelines"
    )

    with open(json_path, "r", encoding="utf-8") as f:
        guidelines = json.load(f)

    ids = [g["id"] for g in guidelines]
    documents = [g["text"] for g in guidelines]
    metadatas = [
        {
            "variable_group": g["variable_group"],
            "reference": g["reference"]
        }
        for g in guidelines
    ]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    print(f"Added {len(ids)} guideline chunks.")


if __name__ == "__main__":
    build_ada_chroma()