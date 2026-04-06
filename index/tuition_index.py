import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_root))

from elasticsearch import helpers
from elasticsearch.helpers import bulk, BulkIndexError

from common.embedding_model import model_large
from common.es_client import es
from mappings import tuition_mapping
from common.tuition_fee_kind import derive_fee_kind
import pandas as pd

def create_index(index_name):
    try:
        if es.indices.exists(index = index_name):
            es.indices.delete(index = index_name)

        mapping = tuition_mapping.mapping["mappings"]
        es.indices.create(index = index_name, mappings = mapping)
        print(f"Index {index_name} created successfully.")
    except Exception as e:
        raise Exception(f"Failed to create index {index_name}: {str(e)}")

# Indexing
index_name = "iit_tuition"

if __name__ == "__main__":
    data = pd.read_json(_root / "data" / "tuition_data.json")

    actions = []
    for i, row in data.iterrows():
        semantic_text = row["chunk_text"]
        semantic_vector = model_large.encode(
            f"passage: {semantic_text}",
            normalize_embeddings=True
        ).tolist()

        actions.append({
            "_index": index_name,
            "_source": {
                "doc_id": row["doc_id"],
                "chunk_text": row["chunk_text"],
                "content": row["content"],
                "school": row["school"],
                "level": row["level"],
                "section": row["section"],
                "fee_name": row["fee_name"],
                "fee_kind": derive_fee_kind(row["fee_name"]),
                "billing_period": row["billing_period"],
                "academic_year": row["academic_year"],
                "term": row["term"],
                "enrollment": row["enrollment"],
                "program": row["program"],
                "unit": row["unit"],
                "amount_value": row["amount_value"],
                "source_url": row["source_url"],
                "semantic_text": semantic_text,
                "semantic_vector": semantic_vector,
            },
        })

    create_index(index_name)

    try:
        helpers.bulk(es, actions)
        n = len(actions)
        print(f"Indexed {n} documents to {index_name}.")
    except Exception as e:
        print(f"Failed to index data: {str(e)}")