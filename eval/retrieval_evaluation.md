# RAG Retrieval Evaluation

This evaluation measures how well the retrieval component of the travel-planning RAG flow returns the document chunk that is relevant to a user's question. It compares two retrieval strategies against the same ground-truth dataset:

- **Text search**: Elasticsearch BM25 multi-match search over `page_title`, `section`, and `text`.
- **Vector search**: k-nearest-neighbour search over the `embedding` field using `sentence-transformers/all-MiniLM-L6-v2`.

The top five results returned by each strategy are evaluated. The generation step of the RAG flow is not evaluated here; this is a retrieval-only evaluation.

## Ground-truth dataset

`ground_truth.json` contains source chunks and realistic traveller questions that should be answered by each chunk. Every record has this form:

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

The `id` is the expected relevant document ID. During evaluation, each question is sent to a search method and its returned IDs are converted into a relevance list: `1` when the expected ID appears, otherwise `0`.

## Generate `ground_truth.json`

`generate_ground_truth.py` samples chunks from `chunks.json` and asks Gemini to create three natural travel questions for every sampled chunk. 

Set `GEMINI_API_KEY` and run this command from the project root:

```bash
export GEMINI_API_KEY="your-api-key"
python eval/generate_ground_truth.py \
  --input chunks.json \
  --output eval/ground_truth.json \
  --samples 30
```

This generated dataset was used for evaluation result comparison of text search and vector search. 

The ground truth JSON file contains fewer than the intended 90 questions because Gemini occasionally returned invalid JSON responses and the free-tier rate limits interrupted the generation process.

## Run the evaluation

Before evaluating, start Elasticsearch, index the same chunks that were used to create the dataset, and ensure the embedding model is available locally. The evaluator expects the Elasticsearch index name `travel-chunks` at `http://localhost:9200`.

```bash
python index.py --chunks chunks.json --index travel-chunks --recreate
python eval/evaluate_retrieval.py
```

`evaluate_retrieval.py` loads `eval/ground_truth.json`, runs every question through `SearchEngine.text_search` and `SearchEngine.vector_search`, then prints the metrics. It also evaluates hybrid reciprocal-rank-fusion variants; those results are separate from the text-versus-vector comparison below.

## Metrics

| Metric | Meaning |
|---|---|
| Hit Rate | Fraction of questions for which the expected chunk appears anywhere in the top five results. Higher is better. |
| MRR (Mean Reciprocal Rank) | Average of `1 / rank` for the first occurrence of the expected chunk; zero if it is absent. Higher is better, and it rewards results found nearer rank 1. |

## Results
| Retrieval method | Boost dict | Hit Rate@5 | MRR@5 | Result |
|------------------|------------|-----------:|------:|--------|
| Text search (BM25) | `{"page_title": 1, "section": 1, "text": 1}` | 0.8039 | 0.6092 | Baseline keyword retrieval |
| Text search (BM25) | `{"page_title": 3, "section": 2, "text": 1}` | 0.6863 | 0.5085 | BM25 with field boosting |
| Vector search (kNN) | N/A | 0.8235 | 0.65 | Semantic retrieval using MiniLM embeddings |

The default equal weighting ({"page_title": 1, "section": 1, "text": 1}) achieved better retrieval performance than the manually boosted configuration, indicating that additional field weighting did not improve BM25 for this dataset.


## Hybrid search with Reciprocal Rank Fusion (RRF)

RRF combines the ranked result lists from BM25 text search and vector search. Each document earns a score from every list in which it appears, and the scores are added; documents ranked well by either retrieval method can therefore rise in the final hybrid ranking.

`evaluate_retrieval.py` tests `rrf_k` values of `1`, `50`, `100`, and `200`. A smaller value gives more weight to highly ranked results, whereas a larger value smooths rank differences. Every RRF configuration uses the same ground-truth questions and is evaluated with **MRR@5**.

| RRF configuration | MRR@5 | Result |
|---|---:|---|
| Hybrid RRF (`rrf_k=1`) | 0.6569 | Strongly prioritises top-ranked documents |
| Hybrid RRF (`rrf_k=50`) | 0.6650 | Moderate rank smoothing |
| Hybrid RRF (`rrf_k=100`) | 0.6650 | Greater rank smoothing |
| Hybrid RRF (`rrf_k=200`) | 0.6650 | Least sensitivity to rank differences |



# RAG Final Evaluation Result
Although **vector search** outperformed **BM25 text search**, **hybrid retrieval using Reciprocal Rank Fusion (RRF)** achieved the best retrieval performance.   
Therefore, the final RAG system uses **hybrid search with RRF (rrf_k=50)** during inference, as it combines the strengths of keyword and semantic retrieval.

