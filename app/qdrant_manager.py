"""
qdrant_manager.py

Contains QdrantManager for managing Qdrant collections and vector upserts.
"""

import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

logger = logging.getLogger(__name__)

class QdrantManager:
    """
    Manages Qdrant collection creation, deletion, and point upserts.
    """

    def __init__(self, client: QdrantClient, embedding_model):
        self.client = client
        self.model = embedding_model

    def ensure_collection_exists(self, project_id: str):
        """
        Checks if a Qdrant collection with the given project_id exists.
        If it doesn't, it creates one.
        """
        try:
            self.client.get_collection(project_id)
        except Exception:
            logger.info(f"Creating new collection for project '{project_id}'")
            self.client.create_collection(
                collection_name=project_id,
                vectors_config=VectorParams(
                    size=self.model.get_sentence_embedding_dimension(),
                    distance=Distance.COSINE
                )
            )

    def create_collection(self, project_id: str):
        """
        Creates a new Qdrant collection for the given project_id.
        """
        self.client.create_collection(
            collection_name=project_id,
            vectors_config=VectorParams(
                size=self.model.get_sentence_embedding_dimension(),
                distance=Distance.COSINE
            )
        )

    def delete_collection(self, project_id: str):
        """
        Deletes the specified Qdrant collection.
        """
        self.client.delete_collection(collection_name=project_id)

    def get_collections(self):
        """
        Returns a list of all existing Qdrant collections.
        """
        return self.client.get_collections().collections

    def upsert_points(self, project_id: str, points: list, batch_size: int = 100):
        """
        Upserts the given points to the specified Qdrant collection in batches.
        """
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                self.client.upsert(collection_name=project_id, points=batch)
            except Exception as e:
                error_message = f"Error upserting batch {i // batch_size + 1} in '{project_id}': {str(e)}"
                logger.error(error_message)
                raise
