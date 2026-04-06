import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_root))

from elasticsearch import helpers
from elasticsearch.helpers import bulk, BulkIndexError

from common.embedding_model import model_large
from common.es_client import es
from mappings import contacts_mapping
import pandas as pd

def create_index(index_name):
    try:
        if es.indices.exists(index = index_name):
            es.indices.delete(index = index_name)

        mapping = contacts_mapping.mapping["mappings"]
        es.indices.create(index = index_name, mappings = mapping)
        print(f"Index {index_name} created successfully.")
    except Exception as e:
        raise Exception(f"Failed to create index {index_name}: {str(e)}")

# Build Semantic Text
def build_semantic_text(row):
    """
    Build a natural-language passage optimised for dense retrieval.

    Phrasing mirrors how a student would ask about this contact:
    "how do I contact", "who handles", "where is the office for".
    Avoids generic filler like "belongs to" or "is a unit" which add
    noise without retrieval value.
    """
    parts = []

    name     = row["Name"].strip()
    dept     = row["Department"].strip()
    category = row["Category"].strip()
    desc     = row["Description"].strip()
    phone    = row["Phone"].strip()
    email    = row["Email"].strip()
    building = row["Building"].strip()
    address  = row["Address"].strip()
    city     = row["City"].strip()
    state    = row["State"].strip()
    zipcode  = row["Zip"].strip()

    # Who/what this entry is
    if name and dept and name.lower() != dept.lower():
        parts.append(f"{name} works in {dept}.")
    elif name:
        parts.append(f"{name} is a {category} at Illinois Tech.")

    # Role / function — critical for queries like "who handles transcripts"
    if desc and desc.lower() not in (name.lower(), dept.lower()):
        parts.append(f"Their role is: {desc}.")

    # Contact — phrased to match student queries
    if phone:
        parts.append(f"You can reach them by phone at {phone}.")
    if email:
        parts.append(f"You can contact them by email at {email}.")

    # Location
    if building:
        parts.append(f"They are located in {building}.")
    if address:
        loc = address
        if city:
            loc += f", {city}"
        if state:
            loc += f", {state}"
        if zipcode:
            loc += f" {zipcode}"
        parts.append(f"Their office address is {loc}.")

    return " ".join(parts)

# Indexing
index_name = "iit_contacts"

if __name__ == "__main__":

    data = pd.read_csv(_root / "data" / "Contacts data.csv", dtype=str)
    data = data.fillna("")
    data = data.replace("\xa0", " ", regex=True)

    actions = []

    for i, row in data.iterrows():
        semantic_text = build_semantic_text(row)
        semantic_vector = model_large.encode(
            f"passage: {semantic_text}",
            normalize_embeddings=True
        ).tolist()

        actions.append({
            "_index": index_name,
            "_source": {
                "name": row["Name"],
                "department": row["Department"],
                "category": row["Category"],
                "description": row["Description"],
                "phone": row["Phone"],
                "fax": row["Fax"],
                "email": row["Email"],
                "building": row["Building"],
                "address": row["Address"],
                "city": row["City"],
                "state": row["State"],
                "zip": row["Zip"],
                "source_url": row["Source_url"],
                "semantic_text": semantic_text,
                "semantic_vector": semantic_vector,
            },
        })

    create_index(index_name)

    try:
        success, failed = helpers.bulk(
            es,
            actions,
            raise_on_error=False
        )

        print(f"Successfully indexed: {success}")

        if failed:
            print("Some documents failed:")
            for item in failed:
                print(item)

    except BulkIndexError as e:
        print("Bulk indexing error:")
        for err in e.errors:
            print(err)