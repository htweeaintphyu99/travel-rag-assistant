"""

Generate RAG evaluation ground truth dataset.
Pipeline:

1. Randomly sample chunks

2. Ask Gemini to generate questions

3. Save questions with relevant document IDs

"""

import json
import os
import random
import argparse
from google import genai
from google.genai import types
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class GroundTruthGenerator:
    def __init__(self, api_key: str, model: str = "gemini-3.5-flash"):

        self.client = genai.Client(api_key=api_key)

        self.model = model

    def generate_questions(self, text: str, num_questions: int = 3) -> list[str]:

        prompt = f"""
You are creating evaluation questions for a Retrieval Augmented Generation (RAG) based travel assistant.

Your goal is to simulate realistic questions asked by travellers when planning a trip.

Given the following travel document chunk, generate {num_questions} questions.

Requirements:
- Write questions from the perspective of a traveller.
- Questions should sound like natural user queries to a travel assistant.
- Questions must require information from this document to answer.
- Questions should not mention the document or the source.
- Avoid generic questions that can be answered without this specific information.

- Cover different aspects when possible:
  - attractions and sightseeing
  - culture and history
  - food and drinks
  - transportation
  - activities and experiences
  - travel tips

- Use different traveller intents (e.g., first-time visitor, family traveller, food lover, history enthusiast).
- Return ONLY a JSON list of strings.

Example style:
[
  "What are the must-see historical places in this city for a first-time visitor?",
  "Where can I experience traditional local culture during my trip?",
  "What local foods should I try when visiting this destination?"
]

Travel document:
{text}
"""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3, response_mime_type="application/json"
                ),
            )

            return json.loads(response.text)
        except Exception as e:
            print(f"Gemini JSON error ")
            print(e)

    def generate_dataset(self, chunks: list[dict], sample_size: int = 50) -> list[dict]:
        samples = random.sample(chunks, sample_size)
        dataset = []
        for idx, chunk in enumerate(samples):
            print(f"Generating {idx + 1}/{sample_size}: {chunk['id']}")

            questions = self.generate_questions(chunk["text"])

            dataset.append(
                {
                    "id": chunk["id"],
                    "city": chunk.get("city"),
                    "source": chunk.get("source"),
                    "page_title": chunk.get("page_title"),
                    "section": chunk.get("section"),
                    "text": chunk["text"],
                    "questions": questions,
                }
            )

        return dataset


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="chunks.json")
    parser.add_argument("--output", default="ground_truth.json")
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()

    chunks = load_json(args.input)
    generator = GroundTruthGenerator(api_key=os.getenv("GEMINI_API_KEY"))
    dataset = generator.generate_dataset(chunks, args.samples)

    save_json(dataset, args.output)
    print(f"Saved {len(dataset)} items to {args.output}")


if __name__ == "__main__":
    main()
