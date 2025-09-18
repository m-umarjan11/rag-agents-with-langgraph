"""
Embeddings service using Google Gemini
"""
import asyncio
from typing import List, Optional
import numpy as np

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from core.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class EmbeddingsService:
    def __init__(self):
        self._initialize_gemini()

    def _initialize_gemini(self):
        """Initialize Google Gemini API"""
        if genai is None:
            raise ImportError(
                "Google GenerativeAI library is required. Install with: pip install google-generativeai"
            )

        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required for embeddings service")

        genai.configure(api_key=settings.GOOGLE_API_KEY)
        logger.info("Gemini API initialized successfully")

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using Gemini
        """
        try:
            # Run the synchronous API call in a thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: genai.embed_content(
                    model="models/embedding-001",  # Gemini embedding model
                    content=text,
                    task_type="retrieval_document"
                )
            )

            embedding = result['embedding']
            logger.debug(f"Generated embedding with dimension: {len(embedding)}")
            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise ValueError(f"Failed to generate embedding: {str(e)}")

    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        """
        try:
            embeddings = []

            # Process in batches to avoid rate limits
            batch_size = 10
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = await asyncio.gather(
                    *[self.generate_embedding(text) for text in batch],
                    return_exceptions=True
                )

                # Handle any exceptions in the batch
                for j, embedding in enumerate(batch_embeddings):
                    if isinstance(embedding, Exception):
                        logger.error(f"Error in batch embedding {i+j}: {str(embedding)}")
                        # Use zero vector as fallback
                        embedding = [0.0] * 768  # Default Gemini embedding dimension
                    embeddings.append(embedding)

                # Add delay between batches to respect rate limits
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)

            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise

    async def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query (may use different task type)
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: genai.embed_content(
                    model="models/embedding-001",  # Gemini embedding model
                    content=query,
                    task_type="retrieval_query"
                )
            )

            embedding = result['embedding']
            logger.debug(f"Generated query embedding with dimension: {len(embedding)}")
            return embedding

        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise ValueError(f"Failed to generate query embedding: {str(e)}")

    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0

    async def find_similar_embeddings(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar embeddings to query
        Returns list of (index, similarity_score) tuples
        """
        try:
            similarities = []

            for i, candidate in enumerate(candidate_embeddings):
                similarity = self.cosine_similarity(query_embedding, candidate)
                similarities.append((i, similarity))

            # Sort by similarity score (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Return top k results
            return similarities[:top_k]

        except Exception as e:
            logger.error(f"Error finding similar embeddings: {str(e)}")
            return []