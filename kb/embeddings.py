import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
import hashlib
import pickle
import os


class ImprovedEmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", cache_dir: str = "embedding_cache"):
        """
        Initialize default model and caching
    
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
        # Set up caching
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # In-memory cache for frequent queries
        self.memory_cache = {}
        self.max_memory_cache = 1000
    
    def _get_cache_key(self, texts: List[str], normalize: bool) -> str:
        """Generate cache key for texts"""
        # Create hash from texts and parameters
        content = f"{self.model_name}_{normalize}_{str(texts)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> np.ndarray:
        """Load embeddings from disk cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                # If cache is corrupted, remove it
                os.remove(cache_file)
        return None
    
    def _save_to_cache(self, cache_key: str, embeddings: np.ndarray):
        """Save embeddings to disk cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
        except:
            pass  # Silently fail cache saves
    
    def encode(self, texts: Union[str, List[str]], normalize: bool = True, batch_size: int = 32) -> np.ndarray:
        """
        Encode texts with caching and batching
        """
        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        # Check memory cache first
        cache_key = self._get_cache_key(texts, normalize)
        if cache_key in self.memory_cache:
            embeddings = self.memory_cache[cache_key]
            return embeddings[0:1] if single_text else embeddings
        
        # Check disk cache
        cached_embeddings = self._load_from_cache(cache_key)
        if cached_embeddings is not None:
            # Add to memory cache
            if len(self.memory_cache) < self.max_memory_cache:
                self.memory_cache[cache_key] = cached_embeddings
            return cached_embeddings[0:1] if single_text else cached_embeddings
        
        # Generate embeddings in batches for efficiency
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch_texts, 
                convert_to_numpy=True,
                show_progress_bar=False
            )
            all_embeddings.append(batch_embeddings)
        
        # Combine all batches
        embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
        
        # Normalize if requested
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, a_min=1e-12, a_max=None)
        
        # Cache the results
        self._save_to_cache(cache_key, embeddings)
        
        # Add to memory cache if space available
        if len(self.memory_cache) < self.max_memory_cache:
            self.memory_cache[cache_key] = embeddings
        
        # Return single embedding or all embeddings
        return embeddings[0:1] if single_text else embeddings
    
    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Special method for encoding queries with potential preprocessing
        """
        # You could add query-specific preprocessing here
        # For now, just use regular encoding
        return self.encode([query], normalize=normalize)
    
    def encode_documents(self, docs: List[str], normalize: bool = True, batch_size: int = 16) -> np.ndarray:
        """
        Special method for encoding documents (potentially larger batch size)
        """
        return self.encode(docs, normalize=normalize, batch_size=batch_size)
    
    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between embeddings
        """
        # Ensure embeddings are normalized
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        embeddings1_norm = embeddings1 / np.clip(norm1, a_min=1e-12, a_max=None)
        embeddings2_norm = embeddings2 / np.clip(norm2, a_min=1e-12, a_max=None)
        
        return np.dot(embeddings1_norm, embeddings2_norm.T)
    
    def clear_cache(self):
        """Clear both memory and disk cache"""
        self.memory_cache.clear()
        
        # Clear disk cache
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        disk_cache_files = 0
        disk_cache_size = 0
        
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    disk_cache_files += 1
                    disk_cache_size += os.path.getsize(os.path.join(self.cache_dir, filename))
        
        return {
            "memory_cache_entries": len(self.memory_cache),
            "disk_cache_files": disk_cache_files,
            "disk_cache_size_mb": disk_cache_size / (1024 * 1024),
            "model_name": self.model_name
        }



class EmbeddingGenerator(ImprovedEmbeddingGenerator):
    """Backward compatible embedding generator"""
    pass