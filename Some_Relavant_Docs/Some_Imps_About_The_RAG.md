# Embedding Dimensions
> If the dimensions of objects in the weaviate cluster donot match then it fails to retrieve objects and throws an exception
- OpenAI = 1536
- Sentence Transformers = 768

# Two similar docs are treated as two in the weaviate database

# Ingestion IDs
> When a doc is ingested, the ID that is generated is random, is in no particular order. Hence no neighboring docs can be extracted from the VDB itself.