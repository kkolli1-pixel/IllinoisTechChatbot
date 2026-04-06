mapping = {
    "mappings": {
        
        "properties": {

            "doc_id": {
                "type": "keyword"
            },

            "school": {
                "type": "keyword"
            },

            "chunk_text": {
                "type": "text"
            },

            "content": {
                "type": "text"
            },

            "level": {
                "type": "keyword"
            },

            "section": {
                "type": "keyword"
            },

            "fee_name": {
                "type": "keyword"
            },

            "fee_kind": {
                "type": "keyword"
            },

            "billing_period": {
                "type": "keyword"
            },

            "academic_year": {
                "type": "keyword"
            },

            "term": {
                "type": "keyword"
            },

            "enrollment": {
                "type": "keyword"
            },

            "program": {
                "type": "keyword"
            },

            "unit": {
                "type": "keyword"
            },

            "amount_value": {
                "type": "float"
            },

            "source_url": {
                "type": "keyword"
            },

            "semantic_text": {
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