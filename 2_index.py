#!/usr/bin/env python
"""
2_index.py
==========
Chunks both corpora and embeds them into ChromaDB (cosine HNSW). Each page's text is split into lines and packed
greedily into chunks of at most 1,500 characters with 250 characters of overlap (a single line longer than the
limit is kept whole), and every chunk is prefixed with "[Page: <title>]". The BM25 index is built in memory from
the C_pre collection when a retriever starts (retrieval.py).

    python 2_index.py
"""
import argparse
import json
import os
import re
from typing import Any, Dict, List

import chromadb

from config import (CHROMA_DB_PATH, CHUNK_OVERLAP, EMBED_MODEL, MAX_CHUNK_SIZE, TEST_COLLECTION_NAME,
                    TEST_JSON_FILE, TRAIN_COLLECTION_NAME, TRAIN_JSON_FILE)
from retrieval import get_embedding_function


def text_chunker(content_list: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Chunks a list of strings (paragraphs/lines) into overlapping blocks of text.
    Uses a rolling window approach, prioritizing complete logical units (sentences/lines).
    """
    if not content_list: return []
    chunks, current_chunk = [], ""

    for line in content_list:
        # If adding the current line exceeds the max size, finalize the current chunk
        if len(current_chunk) + len(line) + 1 > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Start the new chunk using the overlap portion of the previous chunk
            if len(current_chunk) > chunk_overlap:
                # Find a sensible point to start the overlap (e.g., after the first
                # line that overlaps, or just a character slice)
                overlap_start = max(0, len(current_chunk) - chunk_overlap)
                current_chunk = current_chunk[overlap_start:].strip() + " " + line
            else:
                # If the previous chunk was too small for meaningful overlap, start fresh
                current_chunk = line
        else:
            # Append the current line to the running chunk
            if current_chunk:
                current_chunk += " " + line
            else:
                current_chunk = line

    # Add the final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def clean_chunk_text(text: str) -> str:
    """Removes common wiki noise like edit sections and excessive whitespace."""
    # Remove the section edit links
    cleaned_text = re.sub(r'\[\s*edit\s*\|\s*edit source\s*\]', '', text, flags=re.IGNORECASE).strip()
    # Remove footnote markers (e.g., [1], [3])
    cleaned_text = re.sub(r'\[\d+\]', '', cleaned_text).strip()
    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()
    return cleaned_text


def build_database(json_file_path: str, db_path: str, collection_name: str, embedding_function: chromadb.EmbeddingFunction):
    """
    Reads the JSON data, chunks it with overlap, and loads it into a fresh ChromaDB.
    """
    if not os.path.exists(json_file_path):
        print(f"Error: Source JSON file not found at {json_file_path}")
        return

    print(f"Loading data from {json_file_path}...")
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Initialize ChromaDB Client and Collection
    client = chromadb.PersistentClient(path=db_path)

    print(f"Preparing to build collection: '{collection_name}'")
    # Delete the old collection if it exists for a clean build
    try:
        if collection_name in [c.name for c in client.list_collections()]:
            client.delete_collection(name=collection_name)
            print(f"Successfully deleted old collection.")
    except Exception as e:
        print(f"An error occurred while deleting the collection: {e}")
        return None

    print(f"Creating new collection '{collection_name}' using '{EMBED_MODEL}'...")
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function
    )

    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []
    document_counter = 0

    for page_key, doc_data in data.items():
        # Get the actual URL from the doc_data dictionary
        source_url = doc_data.get("url", "")
        if not source_url:
            print(f"Skipping entry with missing URL: {page_key}")
            continue

        # Clean and prepare title
        title = source_url
        if '/w/' in title:
            title = title.split('/w/')[-1].replace('_', ' ')
        title = title.strip()

        # Get the other metadata fields, providing default values
        oldid = doc_data.get("oldid", "N/A")
        revision_date = doc_data.get("date", "N/A")
        content = doc_data.get("content", "")

        # Split the page into lines for the chunker.
        if not content:
            print(f"Skipping empty content for: {title}")
            continue

        content_list = [line.strip() for line in content.split('\n') if line.strip()]

        # Generate overlapping chunks for the document's content
        chunks = text_chunker(content_list, MAX_CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk_text in enumerate(chunks):
            cleaned_chunk = clean_chunk_text(chunk_text)
            if not cleaned_chunk:
                continue

            # Contextual retrieval: prepend page title context to ensure referents are preserved
            contextual_chunk = f"[Page: {title}]\n{cleaned_chunk}"
            documents.append(contextual_chunk)
            # 4. Store the complete metadata for each chunk
            metadatas.append({
                "title": title,
                "source_url": source_url,
                "oldid": oldid,
                "revision_date": revision_date,
                "chunk_index": i
            })

            ids.append(f"doc_{document_counter}")
            document_counter += 1

            # Batch loading into ChromaDB
            # The embedding function will be called automatically by .add()
            if len(documents) >= 500:
                print(f"Adding batch of {len(documents)} chunks to ChromaDB...")
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                documents, metadatas, ids = [], [], []

    # Add final batch of remaining documents
    if documents:
        print(f"Adding final batch of {len(documents)} chunks to ChromaDB...")
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    print("\n--- Database Initialization Complete ---")
    print(f"Total chunks indexed: {document_counter}")
    print(f"DB saved at: {os.path.abspath(db_path)}")
    return client


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Chunk and embed C_pre and C_post into ChromaDB.")
    ap.add_argument("--db", default=CHROMA_DB_PATH)
    args = ap.parse_args()
    ef = get_embedding_function()
    build_database(TRAIN_JSON_FILE, args.db, TRAIN_COLLECTION_NAME, ef)
    build_database(TEST_JSON_FILE, args.db, TEST_COLLECTION_NAME, ef)
