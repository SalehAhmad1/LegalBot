- Query Translation
- Re-Rank/Ranking of relavant documents


# By Maaz
- focus on RAPTOR, and Multi Vector Reterival
- we can insire a new idea from all these approaches or from the last section (Future of RAG) tou wo bhi soch lena

- We have to add a guardrail to make sure it is relavant to uk (all of uk)
    - We make separate DBs for the countires and then we use logical routing via LLMs to do that


- Query Translation
    - Multi Query i.e. different versions of the same question but no ranking
    - RAG Fusion i.e. different versions of the same question but with ranking 
    - Decomposition i.e. decompose to sub problems and solve sequentially. i.e. generate similar queries. Solve first. Pass as doc to 2nd query and so on untill the very end. or you can use the answers to form documents to send to the LLM for the final answer.
    - Step Back Prompting i.e. modifying the prompt to be able to get better results. more like simplify the query.
    - Hyde i.e.  take question/prompt and generate a hypothetical document from the original documents and prompt. 

    Which can we use? 
    - Multi Query -> Yes
    - RAG Fusion -> Yes


- Routing
    - Logical
    Routing to right source. It uses an LLM as a classifier with structured output to route to the required source.
    Possible Scenarios:
        - UK/Whales/Scotland/Nothern Ireland
        - Some Other Country
        - None but in UK/Whales/Scotland/Nothern Ireland
        - None but not in UK/Whales/Scotland/NOthern Ireland
    - Semantic
    Based on the question, select a relavant prompt to answer from. 
    RN i donot need this but based on the user I could. I'll ask the client.

## Indexing
- Chunk Optimization
    Types:
    - Characters
    - Semantics
    - Delimiters
    - Sections

- Multi Representation Embedding
    - We donot need it. We need the chunks to be as is since it is formal data.

- RAPTOR
    - Make clusters from documents. Summarize each cluster. This way we have less number of consise, crisp documents. Use those to make a final summary and then it will be the context.
    - In the vector store you feed
        - Raw doc
        - Mid level summaries (intermediate abstract)
        - Final summary (More abstract)

- Use finetuned LLM's embeddings to embed the docs. ColBERT 