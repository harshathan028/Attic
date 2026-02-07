"""
Vector Store - ChromaDB integration for semantic memory.

This module provides vector storage capabilities using ChromaDB,
enabling semantic search and retrieval of research findings.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional
import uuid

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document in the vector store."""
    id: str
    content: str
    metadata: dict
    score: Optional[float] = None


class VectorStore:
    """
    ChromaDB-based vector store for document storage and retrieval.
    
    Provides semantic search capabilities for the agent pipeline,
    allowing agents to store and retrieve contextually relevant information.
    """

    def __init__(
        self,
        collection_name: str = "content_factory",
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Directory to persist the database.
                If None, uses in-memory storage.
        """
        self.collection_name = collection_name

        if persist_directory:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
            logger.info(f"Initialized persistent vector store at {persist_directory}")
        else:
            self.client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False),
            )
            logger.info("Initialized in-memory vector store")

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "AI Content Factory knowledge base"},
        )
        logger.info(f"Using collection: {collection_name}")

    def store_document(
        self,
        content: str,
        metadata: Optional[dict] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """
        Store a document in the vector store.

        Args:
            content: The text content to store.
            metadata: Optional metadata dictionary.
            doc_id: Optional document ID. Generated if not provided.

        Returns:
            The document ID.
        """
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        metadata = metadata or {}
        metadata["content_length"] = len(content)

        try:
            self.collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[doc_id],
            )
            logger.info(f"Stored document {doc_id} ({len(content)} chars)")
            return doc_id

        except Exception as e:
            logger.error(f"Failed to store document: {e}")
            raise

    def store_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Store multiple documents in the vector store.

        Args:
            documents: List of text contents to store.
            metadatas: Optional list of metadata dictionaries.
            ids: Optional list of document IDs.

        Returns:
            List of document IDs.
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        if metadatas is None:
            metadatas = [{"content_length": len(doc)} for doc in documents]
        else:
            for i, md in enumerate(metadatas):
                md["content_length"] = len(documents[i])

        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info(f"Stored {len(documents)} documents")
            return ids

        except Exception as e:
            logger.error(f"Failed to store documents: {e}")
            raise

    def retrieve_context(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> List[Document]:
        """
        Retrieve relevant documents based on semantic similarity.

        Args:
            query: The search query.
            n_results: Maximum number of results to return.
            where: Optional filter conditions.

        Returns:
            List of Document objects with relevance scores.
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
            )

            documents = []
            if results and results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    doc_id = results["ids"][0][i] if results["ids"] else str(i)
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results.get("distances") else None
                    
                    # Convert distance to similarity score (0-1, higher is better)
                    score = 1 / (1 + distance) if distance is not None else None

                    documents.append(Document(
                        id=doc_id,
                        content=doc,
                        metadata=metadata,
                        score=score,
                    ))

            logger.info(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
            return documents

        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            raise

    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents as a context string.

        Args:
            documents: List of Document objects.

        Returns:
            Formatted context string for LLM input.
        """
        if not documents:
            return "No relevant context found."

        formatted = ["=== Retrieved Context ===\n"]
        for i, doc in enumerate(documents, 1):
            score_str = f" (relevance: {doc.score:.2f})" if doc.score else ""
            formatted.append(f"[Source {i}]{score_str}:\n{doc.content}\n")

        return "\n".join(formatted)

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the vector store.

        Args:
            doc_id: The document ID to delete.

        Returns:
            True if successful.
        """
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted document {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            raise

    def clear(self) -> None:
        """Clear all documents from the collection."""
        try:
            # Delete and recreate the collection
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "AI Content Factory knowledge base"},
            )
            logger.info("Cleared vector store")
        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")
            raise

    def count(self) -> int:
        """
        Get the number of documents in the store.

        Returns:
            Document count.
        """
        return self.collection.count()
