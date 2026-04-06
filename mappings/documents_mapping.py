mapping = {
    "mappings": {
        "properties": {
            "chunk_id": {
                "type": "keyword"
            },
            "doc_type": {
                "type": "text"
            },
            "doc_name": {
                "type": "text"
            },
            "source_url": {
                "type": "keyword"
            },
            "topic": {
                "type": "text"
            },
            "page_start": {
                "type": "integer"
            },
            "page_end": {
                "type": "integer"
            },
            "token_count": {
                "type": "integer"
            },
            "content": {
                "type": "text"
            },
            "semantic_vector": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
}