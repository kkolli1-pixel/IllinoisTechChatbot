mapping = {
    "mappings": {
        "properties": {
            "name": {
                "type": "text",
                "fields": {
                    "keyword": {
                        "type": "keyword"
                    }
                }
            },
            "department": {
                "type": "text",
                "fields": {
                    "keyword": {
                        "type": "keyword"
                    }
                }
            },

            "category": {
                "type": "keyword"
            },

            "description": {
                "type": "text"
            },

            "phone": {
                "type": "keyword"
            },

            "fax": {
                "type": "keyword"
            },

            "email": {
                "type": "keyword"
            },

            "building": {
                "type": "text"
            },

            "address": {
                "type": "text"
            },

            "city": {
                "type": "keyword"
            },

            "state": {
                "type": "keyword"
            },

            "zip": {
                "type": "keyword"
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