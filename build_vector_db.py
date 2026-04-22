import os
import json
import chromadb
import google.generativeai as genai

API_KEY = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=API_KEY)

EMBEDDING_MODEL = "models/gemini-embedding-001"
DB_DIR = r"c:\Users\SEOP\Desktop\Embedding_Image\chroma_db"
METADATA_FILE = r"c:\Users\SEOP\Desktop\Embedding_Image\optimized_metadata.json"

def get_embedding(text):
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_document"
    )
    return result['embedding']

def main():
    if not os.path.exists(METADATA_FILE):
        print(f"Metadata file not found: {METADATA_FILE}")
        return

    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)

    if not metadata_list:
        print("Empty metadata list.")
        return

    print("Initializing ChromaDB client...")
    client = chromadb.PersistentClient(path=DB_DIR)
    
    collection = client.get_or_create_collection(
        name="architecture_images",
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Loaded {len(metadata_list)} items from JSON.")
    
    ids = []
    embeddings = []
    metadatas = []
    
    for idx, item in enumerate(metadata_list):
        img_id = f"img_{idx:04d}"
        print(f"[{idx+1}/{len(metadata_list)}] Embedding {img_id}...")
        
        mat_str = ", ".join(item.get("materiality", [])) if isinstance(item.get("materiality"), list) else item.get("materiality", "")
        kw_str = ", ".join(item.get("style_keywords", [])) if isinstance(item.get("style_keywords"), list) else item.get("style_keywords", "")
        
        meta = {
            "project_name": item.get("project_name", ""),
            "project_usage": item.get("project_usage", ""),
            "camera_angle": item.get("camera_angle", ""),
            "massing_and_form": item.get("massing_and_form", ""),
            "materiality": mat_str,
            "lighting_and_atmosphere": item.get("lighting_and_atmosphere", ""),
            "surroundings": item.get("surroundings", ""),
            "style_keywords": kw_str,
            "image_path": item.get("image_path", ""),
            "embedding_text": item.get("embedding_text", ""),
            "thumbnail_b64": item.get("thumbnail_b64", "")
        }
        
        meta = {k: ("" if v is None else str(v)) for k, v in meta.items()}
        
        try:
            emb = get_embedding(item["embedding_text"])
            ids.append(img_id)
            embeddings.append(emb)
            metadatas.append(meta)
        except Exception as e:
            print(f"Error embedding {img_id}: {e}")
            
    if ids:
        print("Upserting vectors into ChromaDB...")
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"Successfully added {len(ids)} items to Vector DB.")
    else:
        print("No items to upsert.")

if __name__ == "__main__":
    main()
