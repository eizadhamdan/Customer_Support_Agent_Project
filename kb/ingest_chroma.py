import os
import json
import chromadb
import re
from typing import List, Dict
from preprocessing import clean_text
from embeddings import EmbeddingGenerator
from tqdm import tqdm


def smart_chunk_document(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Intelligently chunk documents by paragraphs and sentences
    """
    # First try to split by double newlines (paragraphs)
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If paragraph is small enough, add it to current chunk
        if len(current_chunk) + len(paragraph) <= chunk_size:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        else:
            # Save current chunk if it exists
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If paragraph itself is too long, split by sentences
            if len(paragraph) > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                temp_chunk = ""
                
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) <= chunk_size:
                        temp_chunk += " " + sentence if temp_chunk else sentence
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = sentence
                
                if temp_chunk:
                    current_chunk = temp_chunk
                else:
                    current_chunk = ""
            else:
                current_chunk = paragraph
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Add overlapping context to chunks (except the first one)
    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0 and len(chunks[i-1]) > overlap:
            # Add some context from previous chunk
            prev_context = chunks[i-1][-overlap:].strip()
            overlapped_chunk = f"{prev_context} [...] {chunk}"
        else:
            overlapped_chunk = chunk
        overlapped_chunks.append(overlapped_chunk)
    
    return overlapped_chunks


def extract_metadata(entry: Dict) -> Dict:
    """
    Extract rich metadata from document entries
    """
    metadata = {"source": entry.get("source", "")}
    
    # Add document type based on source URL
    source = metadata["source"]
    if "docs." in source:
        metadata["type"] = "documentation"
    elif "blog" in source or "post" in source:
        metadata["type"] = "blog"
    elif "api" in source:
        metadata["type"] = "api_reference"
    else:
        metadata["type"] = "general"
    
    # Extract domain
    if source.startswith("http"):
        domain = source.split("/")[2]
        metadata["domain"] = domain
    
    # Add other metadata if available
    for key in ["title", "category", "tags", "date"]:
        if key in entry:
            metadata[key] = entry[key]
    
    return metadata


def ingest_improved_kb(
    kb_path: str = "knowledge_base.json", 
    persist_dir: str = "chroma_db", 
    batch_size: int = 50,
    chunk_size: int = 500,
    overlap: int = 100
):
    """
    Improved knowledge base ingestion with smart chunking and metadata
    """
    if not os.path.exists(kb_path):
        raise FileNotFoundError(f"Knowledge base file not found: {kb_path}")

    with open(kb_path, "r", encoding="utf-8") as f:
        kb = json.load(f)

    print(f"Loaded {len(kb)} documents from {kb_path}")

    # Initialize Chroma with persistence
    client = chromadb.PersistentClient(path=persist_dir)
    
    # Clear existing collection if it exists
    try:
        client.delete_collection("knowledge_base")
        print("Cleared existing collection")
    except:
        pass
    
    collection = client.get_or_create_collection("knowledge_base")
    
    # Initialize embedder
    embedder = EmbeddingGenerator()

    # Process all documents into chunks
    all_chunks = []
    all_metadata = []
    
    print("Chunking documents...")
    for doc_idx, entry in enumerate(tqdm(kb, desc="Processing documents")):
        text = clean_text(entry["text"])
        metadata = extract_metadata(entry)
        
        # Smart chunking
        chunks = smart_chunk_document(text, chunk_size, overlap)
        
        for chunk_idx, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:  # Skip very short chunks
                continue
                
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "doc_id": doc_idx,
                "chunk_id": chunk_idx,
                "chunk_count": len(chunks),
                "char_length": len(chunk)
            })
            
            all_chunks.append(chunk)
            all_metadata.append(chunk_metadata)
    
    print(f"Created {len(all_chunks)} chunks from {len(kb)} documents")
    print(f"Average chunks per document: {len(all_chunks) / len(kb):.1f}")

    # Batch ingestion
    print("Ingesting chunks...")
    for start in tqdm(range(0, len(all_chunks), batch_size), desc="Ingesting chunks"):
        end = min(start + batch_size, len(all_chunks))
        batch_chunks = all_chunks[start:end]
        batch_metadata = all_metadata[start:end]

        # Generate embeddings
        embeddings = embedder.encode(batch_chunks)

        # Unique IDs
        ids = [f"chunk-{i}" for i in range(start, start + len(batch_chunks))]

        # Insert into Chroma
        collection.upsert(
            documents=batch_chunks,
            embeddings=embeddings.tolist(),
            metadatas=batch_metadata,
            ids=ids,
        )

    print(f"Successfully ingested {len(all_chunks)} chunks into ChromaDB at '{persist_dir}'")
    
    # Print some statistics
    avg_chunk_length = sum(len(chunk) for chunk in all_chunks) / len(all_chunks)
    print(f"Average chunk length: {avg_chunk_length:.0f} characters")
    
    doc_types = {}
    for meta in all_metadata:
        doc_type = meta.get("type", "unknown")
        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
    
    print(f"Document types: {doc_types}")


if __name__ == "__main__":
    ingest_improved_kb(
        chunk_size=600,
        overlap=150
    )