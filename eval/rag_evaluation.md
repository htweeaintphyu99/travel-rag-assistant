# RAG Answer Evaluation

This evaluation measures the quality of the complete travel assistant RAG pipeline by checking whether generated answers are relevant to user questions.

The evaluation uses the same ground-truth dataset and covers three stages:

- **Retrieval**: Hybrid search retrieves relevant travel chunks from Elasticsearch.

- **Generation**: Gemini generates an answer using the retrieved context.

- **Evaluation**: Gemini judges answer relevance.

Each answer is classified as:
- `RELEVANT`
- `PARTLY_RELEVANT`
- `NON_RELEVANT`

## Run the Evaluation
Run:

```bash
docker compose exec streamlit python eval/evaluate_rag.py
```

## What the Script Does

Running [evaluate_rag.py](evaluate_rag.py) performs the following steps:

1. Initialize the RAG system and recreates the Elasticsearch index from `data/chunks.json`.

2. Load questions from `eval/ground_truth.json`.

3. Run each question through the RAG pipeline using hybrid search and Gemini generation.

4. Evaluate each generated answer using a separate Gemini evaluation prompt.

5. Runs the evaluation using two Gemini models:
   - `gemini-3.1-flash-lite`
   - `gemini-3.5-flash`

6. Saves evaluation results as CSV files.


## Evaluation Results
Comparison of evaluating RAG answer results for gemini-3.1-flash-lite and gemini-3.5-flash is as follows.

| Model name | Relevant count | Partly relevant count | Non-relevant count |
|---|---:|---:|---:|
| `gemini-3.1-flash-lite` | 45 | 4 | 0 |
| `gemini-3.5-flash` | 49 | 0 | 1 |

While running the evaluation script, Gemini 3.1 Flash Lite sometimes returned malformed JSON responses, causing some samples to be skipped. As a result, the total number of evaluated samples differs between the two models.

Based on the results, **Gemini 3.5 Flash** produced more relevant answers, so it was chosen for the RAG pipeline.