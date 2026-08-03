# RAG Retrieval Evaluation

This evaluation measures the performance of the retrieval component in the travel assistant RAG pipeline by checking whether the relevant document chunk is returned for a user's question.

It compares two retrieval strategies using the same ground-truth dataset:

- **Text search**: Elasticsearch BM25 multi-match search across `page_title`, `section`, and `text`.

- **Vector search**: k-nearest-neighbour (kNN) search over the `embedding` field using `sentence-transformers/all-MiniLM-L6-v2`.

The top five retrieved results from each strategy are evaluated using retrieval metrics.

## Generate `ground_truth.json`

The ground truth dataset has already been generated in this directory. It samples 30 chunks from `chunks.json` and uses Gemini to generate 3 natural travel questions for each chunk.

The final dataset contains fewer than the expected 90 questions because some Gemini responses returned invalid JSON.

To reproduce (or) create a new ground truth dataset, run:

```bash
docker compose exec streamlit python eval/generate_ground_truth.py --input="data/chunks.json" --samples=30
```

## Ground-truth dataset
Every record in `ground_truth.json` contains source chunk and realistic user questions.

```json
{
    "id": "wikivoyage:busan:on_foot",
    "city": "Busan",
    "source": "wikivoyage",
    "page_title": "Busan",
    "section": "On foot",
    "text": "Due to the mountains and valleys,...",
    "questions": [
    "Is Busan practical to explore on foot?"
  ]
}
```

## Run the evaluation
Run this in the terminal:
```bash
docker compose exec streamlit python eval/evaluate_retrieval.py
```

The script loads eval/ground_truth.json and evaluates each question using:

* SearchEngine.text_search (BM25)
* SearchEngine.vector_search (kNN)

It reports retrieval metrics for each method and also evaluates hybrid search using Reciprocal Rank Fusion (RRF).

## Metrics

| Metric | Meaning |
|---|---|
| Hit Rate | Percentage of questions where the expected chunk appears in the top five retrieved results. Higher is better. |
| MRR (Mean Reciprocal Rank) | Average reciprocal rank of the first relevant result. Higher values indicate that relevant chunks appear closer to the top of the ranking. |

## Retrieval Results
| Retrieval method | Boost dict | Hit Rate@5 | MRR@5 | Result |
|------------------|------------|-----------:|------:|--------|
| Text search (BM25) | `{"page_title": 1, "section": 1, "text": 1}` | 0.8039 | 0.6092 | Baseline keyword retrieval |
| Text search (BM25) | `{"page_title": 3, "section": 2, "text": 1}` | 0.6863 | 0.5085 | BM25 with field boosting |
| Vector search (kNN) | N/A | 0.8235 | 0.65 | Semantic retrieval using MiniLM embeddings |

The default BM25 configuration performed better than manual field boosting, showing that additional weighting of **page_title** and **section** did not improve retrieval quality for this dataset.


## Hybrid search with Reciprocal Rank Fusion (RRF)

RRF combines BM25 and vector search rankings by assigning scores based on document positions in each result list. This allows documents that perform well in either retrieval method to appear higher in the final ranking.

The evaluation tested different rrf_k values using the same ground-truth dataset and measured performance with MRR@5.



| RRF configuration | MRR@5 | Result |
|---|---:|---|
| Hybrid RRF (`rrf_k=1`) | 0.6569 | Gives stronger preference to top-ranked results |
| Hybrid RRF (`rrf_k=50`) | 0.6650 | Balanced rank weighting |
| Hybrid RRF (`rrf_k=100`) | 0.6650 | Similar performance with more smoothing |
| Hybrid RRF (`rrf_k=200`) | 0.6650 | Similar performance with the most smoothing |

Based on these results, rrf_k=50 was selected for the final inference pipeline because it achieved the best MRR while maintaining a balanced combination of keyword and semantic retrieval.


# RAG Final Evaluation Result

The evaluation showed that **vector search** outperformed **BM25 text search** as a single retrieval method. However, **hybrid retrieval with Reciprocal Rank Fusion (RRF)** achieved the highest overall retrieval performance by combining both keyword-based and semantic search.

Therefore, the final RAG system uses **hybrid search with RRF (`rrf_k=50`)** during inference to leverage the strengths of both retrieval approaches.

