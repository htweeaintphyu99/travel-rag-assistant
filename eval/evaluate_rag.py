import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm.auto import tqdm
from evaluate_utils import load_ground_truth
from travel_assistant.rag_pipeline import llm, rag, initialize, EVAL_PROMPT_TEMPLATE
from travel_assistant.search_engine import SearchEngine

prompt_template = """
You are an expert evaluator for a RAG system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as 'NON_RELEVANT', 'PARTLY_RELEVANT', or 'RELEVANT'.

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  'Relevance': 'NON_RELEVANT' | 'PARTLY_RELEVANT' | 'RELEVANT',
  'Explanation': '[Provide a brief explanation for your evaluation]'
}}
""".strip()

search_engine = SearchEngine(
        host="http://localhost:9200",
        index_name="travel-chunks",
    )

def evaluate_rag(model: str):
  gt_dict = load_ground_truth()
  evaluations = []

  save_every = 10
  count = 0

  for doc_id, questions in gt_dict.items():
      for question in questions:
          answer_data = rag(search_engine, EVAL_PROMPT_TEMPLATE, question, model)

          prompt = prompt_template.format(
              question=question,
              answer_llm=answer_data["answer"],
          )

          try:
              evaluation, token_stat = llm(prompt, model)
              evaluation = json.loads(evaluation)

              record = {
                  "id": doc_id,
                  "question": question,
              }

              evaluations.append((record, answer_data, evaluation))

          except Exception as e:
              print(f"Error: {e}")

          count += 1

          if count % save_every == 0:
              save_results(evaluations, model)

              print(f"Saved {count} evaluations.")


def save_results(evaluations, model, output_path="eval/results"):
    df = pd.DataFrame(
        evaluations,
        columns=["record", "answer", "evaluation"],
    )

    df["id"] = df.record.apply(lambda d: d["id"])
    df["question"] = df.record.apply(lambda d: d["question"])
    df["relevance"] = df.evaluation.apply(lambda d: d["Relevance"])
    df["explanation"] = df.evaluation.apply(lambda d: d["Explanation"])

    os.makedirs(output_path, exist_ok=True)
    df.to_csv(f"{output_path}/rag-eval-{model}.csv", index=False)


def main():
    initialize()
    # evaluate RAG using gemini-3.1-flash-lite model
    evaluate_rag(model="gemini-3.1-flash-lite")

    # evaluate RAG using gemini-3.5-flash
    evaluate_rag(model="gemini-3.5-flash")


if __name__ == "__main__":
    main()
