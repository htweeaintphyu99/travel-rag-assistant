import json
import tqdm 

def load_ground_truth(gt_json: str = "eval/ground_truth.json") -> dict:
    gt_dict = {}
    with open(gt_json, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    for record in ground_truth:
        gt_dict[record["id"]] = record["questions"]

    return gt_dict

def compute_relevance(doc_id, q, search_function):
    results = search_function(query=q)

    relevance = []
    for d in results:
        relevance.append(int(d["id"] == doc_id))
    return relevance

def compute_relevance_total(ground_truth_dict, search_function):
    relevance_total = []

    for id, question_list in tqdm.tqdm(ground_truth_dict.items()):
        for q in question_list:
            relevance = compute_relevance(id, q, search_function)
            relevance_total.append(relevance)

    return relevance_total

def hit_rate(relevance):
    cnt = 0

    for line in relevance:
        if 1 in line:
            cnt = cnt + 1

    return cnt / len(relevance)

def mrr(relevance):
    total_score = 0.0

    for line in relevance:
        for rank in range(len(line)):
            if line[rank] == 1:
                score = 1 / (rank + 1)
                total_score = total_score + score
                break

    return total_score / len(relevance)

def evaluate(ground_truth, search_function):
    relevance_total = compute_relevance_total(ground_truth, search_function)
    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total),
    }
    
def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["id"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]