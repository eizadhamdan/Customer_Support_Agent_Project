import chromadb
import re
from typing import List, Tuple, Dict, Optional
from utils.preprocessing import clean_text
from utils.embeddings import EmbeddingGenerator
from transformers import pipeline
import numpy as np


class RAGPipeline:
    def __init__(self, persist_dir="kb/chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection("knowledge_base")
        self.embedder = EmbeddingGenerator(model_name="sentence-transformers/all-mpnet-base-v2")

        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            tokenizer="deepset/roberta-base-squad2"
        )
        
        # Context management
        self.max_context_length = 2000

    def expand_query(self, query: str) -> str:
        """
        Simple query expansion to improve retrieval
        """
        # Add common variations and synonyms
        query_lower = query.lower()
        expansions = []
        
        # Add original query
        expansions.append(query)
        
        # Add variations for common terms
        if "how to" in query_lower:
            expansions.append(query.replace("how to", "steps to"))
            expansions.append(query.replace("how to", "guide"))
        
        if "what is" in query_lower:
            expansions.append(query.replace("what is", "definition of"))
            expansions.append(query.replace("what is", ""))
        
        if "error" in query_lower or "issue" in query_lower:
            expansions.append(query + " troubleshooting")
            expansions.append(query + " fix")
        
        return " ".join(expansions)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Enhanced retrieval with query expansion and better ranking
        """
        # Expand query for better matching
        expanded_query = self.expand_query(query)
        
        # Get embeddings for expanded query
        q_emb = self.embedder.encode([clean_text(expanded_query)]).tolist()[0]
        
        # Retrieve more candidates initially
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=min(top_k * 2, 10)  # Get more candidates
        )
        
        if not results["documents"][0]:
            return []
        
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results.get("distances", [None] * len(docs))[0]
        
        # Convert distances to similarity scores (lower distance = higher similarity)
        similarities = []
        for dist in distances:
            if dist is not None:
                # Convert distance to similarity (assuming cosine distance)
                similarity = 1 - dist if dist is not None else 0.0
            else:
                similarity = 0.0
            similarities.append(similarity)
        
        # Simple reranking based on query term overlap
        reranked_results = []
        query_terms = set(clean_text(query).lower().split())
        
        for doc, meta, sim in zip(docs, metas, similarities):
            # Calculate term overlap boost
            doc_terms = set(clean_text(doc).lower().split())
            overlap_ratio = len(query_terms.intersection(doc_terms)) / len(query_terms) if query_terms else 0
            
            # Boost score based on term overlap
            boosted_score = sim + (overlap_ratio * 0.1)  # Small boost for term overlap
            
            reranked_results.append((doc, meta.get("source", ""), boosted_score))
        
        # Sort by boosted score and return top_k
        reranked_results.sort(key=lambda x: x[2], reverse=True)
        return reranked_results[:top_k]

    def create_structured_context(self, retrieved_docs: List[Tuple[str, str, float]], query: str) -> str:
        """
        Create well-structured context instead of simple concatenation
        """
        if not retrieved_docs:
            return ""
        
        context_parts = []
        total_length = 0
        
        for i, (doc, source, score) in enumerate(retrieved_docs):
            doc_clean = doc.strip()
            
            # Skip very short or repetitive documents
            if len(doc_clean) < 30:
                continue
                
            # Add document with clear separation
            doc_header = f"Document {i+1}:"
            if source:
                doc_header += f" (Source: {source})"
            
            formatted_doc = f"{doc_header}\n{doc_clean}"
            
            # Check length limits
            if total_length + len(formatted_doc) > self.max_context_length:
                remaining_space = self.max_context_length - total_length - len(doc_header) - 10
                if remaining_space > 100:
                    truncated_doc = f"{doc_header}\n{doc_clean[:remaining_space]}..."
                    context_parts.append(truncated_doc)
                break
            
            context_parts.append(formatted_doc)
            total_length += len(formatted_doc)
        
        return "\n\n---\n\n".join(context_parts)

    def validate_answer(self, answer: str, query: str, context: str) -> Tuple[str, float]:
        """
        Simple answer validation and confidence estimation
        """
        if not answer or answer.lower().strip() in ["", "no answer found.", "no answer"]:
            return "I couldn't find a specific answer to your question in the knowledge base.", 0.0
        
        # Clean up common QA model artifacts
        answer = answer.strip()
        
        # Remove answers that are just repetitions of the question
        if answer.lower() in query.lower():
            return "I couldn't find a specific answer to your question in the knowledge base.", 0.0
        
        # Simple confidence based on answer length and content
        confidence = 0.5  # Base confidence
        
        # Boost confidence for longer, more detailed answers
        if len(answer) > 20:
            confidence += 0.2
        if len(answer) > 50:
            confidence += 0.1
        
        # Reduce confidence for very short answers
        if len(answer) < 10:
            confidence -= 0.2
        
        # Check if answer seems to be from context (simple heuristic)
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        if answer_words.intersection(context_words):
            confidence += 0.1
        
        confidence = max(0.0, min(1.0, confidence))
        
        return answer, confidence

    def generate_answer(self, query: str, top_k: int = 3) -> Tuple[str, List[Dict], float]:
        """
        Generate answer with retrieval and validation
        """
        # Retrieve documents
        retrieved = self.retrieve(query, top_k=top_k)
        
        if not retrieved:
            return "I couldn't find relevant information in the knowledge base.", [], 0.0
        
        # Create structured context
        context = self.create_structured_context(retrieved, query)
        
        # Prepare sources
        sources = []
        for idx, (doc, src, score) in enumerate(retrieved):
            doc_words = doc.split()
            if len(doc_words) > 30:
                excerpt = " ".join(doc_words[:30]) + "..."
            else:
                excerpt = doc
                
            sources.append({
                "id": f"doc-{idx+1}",
                "url": src,
                "excerpt": excerpt,
                "relevance_score": round(float(score), 3)
            })
        
        if not context.strip():
            return "I couldn't find relevant information in the knowledge base.", sources, 0.0
        
        try:
            # Generate answer
            result = self.qa_pipeline({
                "question": query,
                "context": context
            })
            
            raw_answer = result.get("answer", "No answer found.")
            qa_confidence = result.get("score", 0.0)
            
            # Validate and clean up answer
            final_answer, validation_confidence = self.validate_answer(raw_answer, query, context)
            
            # Combine confidences
            combined_confidence = (qa_confidence + validation_confidence) / 2
            
            # Add confidence indicator for very uncertain answers
            if combined_confidence < 0.3:
                final_answer = f"Based on limited information: {final_answer}"
            
            return final_answer, sources, combined_confidence
            
        except Exception as e:
            print(f"Error in QA pipeline: {e}")
            return f"I encountered an error while processing your question: {str(e)}", sources, 0.0
    
    def get_similar_questions(self, query: str, top_k: int = 3) -> List[str]:
        """
        Find similar questions that might help users refine their query
        """
        
        variations = []
        query_lower = query.lower()
        
        if "how" in query_lower:
            variations.append(query.replace("how", "what are the steps"))
            variations.append(query.replace("how", "guide"))
        
        if "what" in query_lower:
            variations.append(query.replace("what", "how"))
            variations.append(query + " examples")
        
        if "error" in query_lower:
            variations.append(query.replace("error", "issue"))
            variations.append(query + " solution")
        
        return variations[:top_k]