import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer
from travel_assistant.search_engine import SearchEngine
from evaluate_utils import *

RRF_K_VALUES = [1, 50, 100, 200]


def evaluate_rrf_k_values(ground_truth, search_engine, k_values=RRF_K_VALUES):
    for k in k_values:
        search_function = lambda query, num_results=5, k=k: (
            search_engine.hybrid_search_eval(
                query,
                num_results=num_results,
                rrf_k=k,
            )
        )
        metrics = evaluate(ground_truth, search_function)
        print(f"k={k}, MRR={metrics['mrr']}")


def main():
    engine = SearchEngine(host="http://localhost:9200", index_name="travel-chunks")
    gt_dict = load_ground_truth()

    print("Evaluating Text Search...")
    text_search_metrics = evaluate(gt_dict, engine.text_search)
    print(
        f"Text Search Metrics: Hit Rate={text_search_metrics['hit_rate']}, MRR={text_search_metrics['mrr']}"
    )

    vector_search_metrics = evaluate(gt_dict, engine.vector_search)
    print(
        f"Vector Search Metrics: Hit Rate={vector_search_metrics['hit_rate']}, MRR={vector_search_metrics['mrr']}"
    )

    evaluate_rrf_k_values(gt_dict, engine)


if __name__ == "__main__":
    main()
