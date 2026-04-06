mapping = {
  "mappings": {
    "properties": {

      "term": {
        "type": "text"
      },

      "event_name": {
        "type": "text"
      },

      "start_date": {
        "type": "date"
      },

      "end_date": {
        "type": "date"
      },

      "source_urls": {
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

# mapping = {
#   "mappings": {
#     "properties": {

#       "term": {
#         "analyzer": "standard",
#         "fields": {
#           "keyword": {
#             "type": "keyword"
#           }
#         },
#         "type": "text"
#       },

#       "academic_year": {
#         "fields": {
#           "date": {
#             "type": "date"
#             "format": "yyyy-MM-dd"
#           }
#         },
#         "type": "keyword"
#       },

#       "event_date": {
#         "type": "date"
#       },

#       "event_name": {
#         "type": "text"
#       },

#       "source_url": {
#         "type": "keyword"
#       }
#     }
#   }
# }
