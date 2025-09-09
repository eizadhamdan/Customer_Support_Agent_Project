import os
import numpy as np
from typing import List, Tuple, Dict
from transformers import pipeline
import faiss
import textwrap

from utils.embeddings import EmbeddingGenerator
from utils.preprocessing import clean_text


class RAGPipeline:
    """
    Improved Retrieval-Augmented Generation pipeline
    """

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Embedding model
        self.embedder = EmbeddingGenerator(model_name=embedding_model)

        # Initialize FAISS index (cosine similarity via inner product)
        self.index = None
        self.docs = []   # text chunks
        self.meta = []   # metadata (e.g. URLs)

        # Better generative model options (choose one):
        # Option 1: Larger T5 model
        self.generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",  # 250M params - much better
            tokenizer="google/flan-t5-base",
            device=0 if os.getenv('CUDA_AVAILABLE') else -1  # GPU if available
        )
        
        # Option 2: Use a more capable model (uncomment if you want this instead)
        # self.generator = pipeline(
        #     "text-generation",
        #     model="microsoft/DialoGPT-medium",
        #     tokenizer="microsoft/DialoGPT-medium",
        #     device=0 if os.getenv('CUDA_AVAILABLE') else -1
        # )

    def build_index(self, documents: List[str], metadata: List[str] = None):
        """
        Build FAISS index from list of documents.
        """
        print(f"Building index for {len(documents)} documents...")
        
        clean_docs = [clean_text(d) for d in documents]
        embeddings = self.embedder.encode(clean_docs, normalize=True)
        embeddings = np.array(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity
        self.index.add(embeddings)

        self.docs = documents
        self.meta = metadata if metadata else [""] * len(documents)
        print(f"Index built successfully with {self.index.ntotal} documents")

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """
        Retrieve top-k most relevant documents with scores.
        """
        if self.index is None:
            raise ValueError("Index is empty. Call build_index() first.")

        q_emb = self.embedder.encode([clean_text(query)], normalize=True)
        q_emb = np.array(q_emb, dtype="float32")

        scores, indices = self.index.search(q_emb, top_k)
        results = [(self.docs[i], self.meta[i], scores[0][j]) for j, i in enumerate(indices[0])]
        return results

    def generate_answer(self, query: str, top_k: int = 3, max_new_tokens: int = 150) -> Tuple[str, List[Dict]]:
        """
        Generate answer with improved prompting and error handling.
        """
        if self.index is None:
            return ("Knowledge base not loaded. Please try again later.", [])

        try:
            retrieved = self.retrieve(query, top_k=top_k)

            if not retrieved:
                return ("I could not find relevant information in the knowledge base.", [])

            # Filter by relevance score (optional - adjust threshold as needed)
            filtered_docs = [(doc, meta, score) for doc, meta, score in retrieved if score > 0.3]
            
            if not filtered_docs:
                return ("No sufficiently relevant information found for your question.", [])

            sources = []
            context_parts = []
            
            for i, (doc_text, meta, score) in enumerate(filtered_docs, start=1):
                # Truncate documents to prevent context overflow
                words = doc_text.strip().split()
                excerpt = " ".join(words[:100])  # Limit to 100 words per doc
                
                sources.append({
                    "id": i, 
                    "url": meta or "unknown", 
                    "excerpt": excerpt,
                    "relevance_score": float(score)
                })
                context_parts.append(f"Source {i}: {excerpt}")

            # Simplified, clearer prompt
            context_str = "\n\n".join(context_parts)
            
            prompt = f"""Based on the following information, answer the question clearly and concisely.

Information:
{context_str}

Question: {query}

Answer:"""

            # Generate with better parameters
            gen_output = self.generator(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.generator.tokenizer.eos_token_id
            )

            # Better output parsing
            if isinstance(gen_output, list) and len(gen_output) > 0:
                generated_text = gen_output[0].get("generated_text", "")
            else:
                generated_text = str(gen_output)
            
            # Clean up the response
            if prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()
                
            if not generated_text:
                generated_text = "I apologize, but I couldn't generate a proper response based on the available information."
                
            return generated_text, sources
            
        except Exception as e:
            print(f"Error in generate_answer: {e}")
            return (f"Sorry, I encountered an error while processing your question: {str(e)}", [])

    def debug_retrieval(self, query: str, top_k: int = 5):
        """
        Debug function to see what documents are being retrieved.
        """
        if self.index is None:
            print("Index not built yet.")
            return
            
        retrieved = self.retrieve(query, top_k=top_k)
        print(f"\nQuery: {query}")
        print("=" * 50)
        
        for i, (doc, meta, score) in enumerate(retrieved, 1):
            print(f"\nRank {i} (Score: {score:.3f}):")
            print(f"Source: {meta}")
            print(f"Content: {doc[:200]}...")
            print("-" * 30)
    
    def get_cache_info(self) -> Dict:
        """Get information about cached models."""
        cache_info = {
            "cache_directory": self.cache_dir,
            "total_size_mb": 0,
            "models": []
        }
        
        if not os.path.exists(self.cache_dir):
            return cache_info
            
        for item in os.listdir(self.cache_dir):
            item_path = os.path.join(self.cache_dir, item)
            if os.path.isdir(item_path):
                # Calculate directory size
                size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(item_path)
                    for filename in filenames
                ) / (1024 * 1024)  # Convert to MB
                
                cache_info["models"].append({
                    "name": item.replace('_', '/'),
                    "size_mb": round(size, 2),
                    "path": item_path
                })
                cache_info["total_size_mb"] += size
        
        cache_info["total_size_mb"] = round(cache_info["total_size_mb"], 2)
        return cache_info
    
    def clear_cache(self, model_name: str = None):
        """Clear cached models."""
        import shutil
        
        if model_name:
            # Clear specific model
            model_cache_path = os.path.join(self.cache_dir, model_name.replace('/', '_'))
            if os.path.exists(model_cache_path):
                shutil.rmtree(model_cache_path)
                print(f"Cleared cache for model: {model_name}")
            else:
                print(f"No cache found for model: {model_name}")
        else:
            # Clear all cache
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
                print("Cleared all model cache")
            else:
                print("No cache directory found")
                