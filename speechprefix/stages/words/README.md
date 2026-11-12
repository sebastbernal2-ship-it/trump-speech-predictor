# Words stage (reserved)

**Status:** scaffold only; no word/token model or task is implemented.

This stage owns lexical sequence prediction downstream of a frozen sentence artifact (or a separately specified direct-context task). Before implementation, version the tokenization contract: what counts as a word, how punctuation/whitespace are represented, how contractions and numbers are handled, and how token IDs map back to surface text.

The stage must not change sentence boundaries or idea coverage implicitly. Report word/token-level metrics separately from sentence and idea metrics, retain source/output provenance, and evaluate both teacher-forced and end-to-end inputs if conditioned on predicted sentences.