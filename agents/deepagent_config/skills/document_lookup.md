# Skill: Document Lookup

Use this skill when the user asks about policy terms, coverage details, exclusions,
claim procedures, regulatory requirements, or any document-based question.

## Steps
1. Call `classify_user_intent` to confirm this is a `document_lookup` query
2. Call `resolve_data_assets` to find the document VS index
3. Call `search_policy_documents` with the resolved index name
4. Cite document titles and relevant content sections in your answer
5. If results are thin, broaden the search terms and try again

## Tips
- Include the product type (motor, health, life, property) in the search query for better results
- Exclusion questions often need the specific coverage type mentioned
- For regulatory questions, look for documents with category "regulatory" or "compliance"
- Always cite the document title when quoting policy content
