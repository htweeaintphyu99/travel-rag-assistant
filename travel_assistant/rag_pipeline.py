from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from time import time
import subprocess

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
from .search_engine import SearchEngine
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EVAL_PROMPT_TEMPLATE = """
You're a travel assistant. Answer the QUESTION based on the CONTEXT from our travel chunks database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

NATURAL_PROMPT_TEMPLATE ="""
You are a friendly and knowledgeable travel assistant.
Answer the user's QUESTION using the provided CONTEXT.
The CONTEXT contains factual information retrieved from a travel knowledge base.

Instructions:
- Give a direct answer first.
- Write naturally, as if you were talking to a traveler who is curious about the country they plan to visit.
- Combine information from multiple context passages when needed.
- Do not copy sentences from the context verbatim.
- Include only information supported by the context.
- If the context does not contain enough information, say so instead of making up facts.
- Use bullet points only when they improve readability.

QUESTION:
{question}

CONTEXT:
{context}
"""

evaluation_prompt_template = """
You are an expert evaluator for a RAG system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


entry_template = """
city: {city}
content: {text}
""".strip()


def initialize():

    subprocess.run(
        [sys.executable, "index.py", "--recreate"],
        check=True,
    )

    global engine

    engine = SearchEngine(
        host="http://localhost:9200",
        index_name="travel-chunks",
    )


def search(engine, query, rrf_k=50):
    results = engine.hybrid_search(query=query, num_results=5, rrf_k=rrf_k)
    return results


def build_prompt(prompt, query, search_results):
    context = ""

    for doc in search_results:
        context = context + entry_template.format(**doc) + "\n\n"

    prompt = prompt.format(question=query, context=context).strip()
    return prompt


def llm(prompt: str, model: str = "gemini-3.5-flash"):
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    answer = response.text
    usage = response.usage_metadata

    token_stats = {
        "prompt_tokens": usage.prompt_token_count,
        "completion_tokens": usage.candidates_token_count,
        "total_tokens": usage.total_token_count,
    }

    return answer, token_stats


def evaluate_relevance(question, answer):
    prompt = evaluation_prompt_template.format(question=question, answer=answer)
    evaluation, tokens = llm(prompt)

    try:
        json_eval = json.loads(evaluation)
        return json_eval, tokens
    except json.JSONDecodeError:
        result = {"Relevance": "UNKNOWN", "Explanation": "Failed to parse evaluation"}
        return result, tokens


def calculate_gemini_cost(model, tokens):
    gemini_cost = 0

    if model == "gemini-3.5-flash":
        # Price per 1M tokens
        input_price = 1.5  # $1.5 / 1M input tokens
        output_price = 9.0  # $9.0 / 1M output tokens

        gemini_cost = (
            tokens["prompt_tokens"] * input_price
            + tokens["completion_tokens"] * output_price
        ) / 1_000_000

    elif model == "gemini-3.1-flash-lite":
        input_price = 0.25  
        output_price = 1.5 

        gemini_cost = (
            tokens["prompt_tokens"] * input_price
            + tokens["completion_tokens"] * output_price
        ) / 1_000_000
    else:
        print("Model not recognized. Gemini cost calculation failed.")

    return gemini_cost


def rag(search_engine, prompt, query, model="gemini-3.5-flash"):
    t0 = time()

    search_results = search(search_engine, query)
    prompt = build_prompt(prompt, query, search_results)
    answer, token_stats = llm(prompt, model=model)

    relevance, rel_token_stats = evaluate_relevance(query, answer)

    t1 = time()
    took = t1 - t0

    gemini_cost_rag = calculate_gemini_cost(model, token_stats)
    gemini_cost_eval = calculate_gemini_cost(model, rel_token_stats)

    gemini_cost = gemini_cost_rag + gemini_cost_eval

    answer_data = {
        "answer": answer,
        "model_used": model,
        "response_time": took,
        "relevance": relevance.get("Relevance", "UNKNOWN"),
        "relevance_explanation": relevance.get(
            "Explanation", "Failed to parse evaluation"
        ),
        "prompt_tokens": token_stats["prompt_tokens"],
        "completion_tokens": token_stats["completion_tokens"],
        "total_tokens": token_stats["total_tokens"],
        "eval_prompt_tokens": rel_token_stats["prompt_tokens"],
        "eval_completion_tokens": rel_token_stats["completion_tokens"],
        "eval_total_tokens": rel_token_stats["total_tokens"],
        "gemini_cost": gemini_cost,
    }

    return answer_data

@dataclass
class LLMCallRecord:
    model: str
    prompt: str
    instructions: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    cost: float
    timestamp: datetime = field(default_factory=datetime.now)

def to_log_record(answer_dict, prompt, instructions):
    return LLMCallRecord(
        model=answer_dict["model_used"],
        prompt=prompt,
        instructions=instructions,
        answer=answer_dict["answer"],
        prompt_tokens=answer_dict["prompt_tokens"],
        completion_tokens=answer_dict["completion_tokens"],
        total_tokens=answer_dict["total_tokens"],
        response_time=answer_dict["response_time"],
        cost=answer_dict["gemini_cost"],
    )
