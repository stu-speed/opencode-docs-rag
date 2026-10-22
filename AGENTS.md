## Subject Selection (Stateless)

RAG runs ONLY when the user explicitly selects subjects:

- "docs subject X"
- "docs subject X,Y"

This call directly invokes docs_rag_query with those subjects.

## Query Routing

A prompt triggers docs_rag_query ONLY when:
1. The user explicitly names one or more subjects, OR
2. The user uses the phrase “docs subject …”, OR
3. The user uses “docs for X”, “docs on X”, “documentation for X”.

If the prompt does NOT name a subject:
- Do NOT call docs_rag_query.
- Treat it as normal conversation.

## Examples

"docs subject xplane" → triggers RAG for xplane  
"docs subject fshd,xplane" → triggers RAG for both  
"search docs for xplane METAR injection" → triggers  
"how do I inject METAR" → does NOT trigger unless subject is named  
"what time is it" → normal conversation

