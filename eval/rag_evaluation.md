# RAG Answer Evaluation

`evaluate_rag.py` evaluates the complete travel-assistant RAG flow, rather than retrieval alone. For every question in `eval/ground_truth.json`, it retrieves travel chunks, generates an answer with Gemini, and asks Gemini to judge whether that answer is relevant to the question.

## What the script does

Running `evaluate_rag.py` performs the following sequence:

1. Calls `initialize()` from `travel_assistant.rag_pipeline`.
   This recreates the `travel-chunks` Elasticsearch index from `data/chunks.json` and creates the search engine.
2. Loads questions from `eval/ground_truth.json` using `load_ground_truth()`.
3. For every ground-truth question, calls `rag(question, model)`.
   The RAG pipeline retrieves the top five hybrid-search results, builds a context prompt, and asks the selected Gemini model to generate the travel answer.
4. Builds a separate evaluator prompt containing the question and generated answer. Gemini returns a JSON verdict:
   `RELEVANT`, `PARTLY_RELEVANT`, or `NON_RELEVANT`, with a brief explanation.
5. Saves the accumulated results every ten questions as a CSV file in output directory.


## Prerequisites

- A running Elasticsearch instance at `http://localhost:9200`.
- `data/chunks.json`, which is used to recreate the `travel-chunks` index.
- `eval/ground_truth.json`, containing the evaluation questions.
- A `GEMINI_API_KEY` value in the environment or `.env` file.
- The `sentence-transformers/all-MiniLM-L6-v2` embedding model available for indexing and vector retrieval.

## Run the evaluation

Run the script from the project root:

```bash
python eval/evaluate_rag.py
```

The active model is selected in `main()`:

```python
evaluate_rag(model="gemini-3.5-flash")
```

To compare models, change the model argument, run the script again, and keep each generated CSV. The output filename includes the model name.


For example, evaluating `gemini-3.5-flash` writes `data/rag-eval-gemini-3.5-flash.csv`.

Each row in output csv file contains:

| Column | Description |
|---|---|
| `id` | Ground-truth source chunk ID associated with the question. |
| `question` | User question sent to the RAG pipeline. |
| `answer` | RAG output and its metadata, including token usage, latency, cost, and the pipeline's own relevance assessment. |
| `relevance` | Verdict produced by the evaluator call: `RELEVANT`, `PARTLY_RELEVANT`, or `NON_RELEVANT`. |
| `explanation` | Brief explanation produced by the evaluator call. |

## Interpreting results

- **RELEVANT** means the answer addresses the traveller's question using appropriate information.
- **PARTLY_RELEVANT** means the answer has useful information but is incomplete, too general, or only partially answers the question.
- **NON_RELEVANT** means the answer does not address the question sufficiently.

Review the `relevance` and `explanation` columns to identify recurring failure patterns, such as weak retrieval context, unsupported answers, or incomplete responses. Because Gemini is used as the evaluator, treat the labels as an automated quality signal and manually inspect a sample of results before making major system changes.

## Evaluation Results
Comparison of evaluating RAG answer results for gemini-3.1-flash-lite and gemini-3.5-flash is as follows.

| Model name | Relevant count | Partly relevant count | Non-relevant count |
|---|---:|---:|---:|
| `gemini-3.1-flash-lite` | 45 | 4 | 0 |
| `gemini-3.5-flash` | 49 | 0 | 1 |

While running the evaluation script, Gemini 3.1 Flash Lite sometimes returned malformed JSON responses, causing some samples to be skipped. As a result, the total number of evaluated samples differs between the two models.

Based on the results, **Gemini 3.5 Flash** produced more relevant answers, so it was chosen for the RAG pipeline.