from __future__ import annotations
import json
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def actor_answer(example: QAExample, _attempt_id: int, _agent_type: str, reflection_memory: list[str]) -> tuple[str, int, int]:
    context_text = "\n\n".join(f"[{c.title}] {c.text}" for c in example.context)
    reflection_text = ""
    if reflection_memory:
        reflection_text = "\n\nReflection from previous attempts:\n" + "\n".join(f"- {r}" for r in reflection_memory)

    user_message = f"Question: {example.question}\n\nContext:\n{context_text}{reflection_text}"

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": ACTOR_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
    )
    latency_ms = int((time.perf_counter() - start) * 1000)
    answer = response.choices[0].message.content.strip()
    tokens = response.usage.total_tokens
    return answer, tokens, latency_ms


def evaluator(example: QAExample, answer: str) -> tuple[JudgeResult, int, int]:
    user_message = (
        f"Question: {example.question}\n"
        f"Gold answer: {example.gold_answer}\n"
        f"Predicted answer: {answer}"
    )

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": EVALUATOR_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
    )
    latency_ms = int((time.perf_counter() - start) * 1000)
    tokens = response.usage.total_tokens
    raw = response.choices[0].message.content.strip()

    try:
        data = json.loads(raw)
        judge = JudgeResult(score=int(data["score"]), reason=data.get("reason", ""))
    except Exception:
        judge = JudgeResult(score=0, reason=raw)

    return judge, tokens, latency_ms


def reflector(example: QAExample, attempt_id: int, judge: JudgeResult, wrong_answer: str = "") -> tuple[ReflectionEntry, int, int]:
    user_message = (
        f"Question: {example.question}\n"
        f"Gold answer: {example.gold_answer}\n"
        f"Wrong predicted answer: {wrong_answer}\n"
        f"Evaluator reason: {judge.reason}"
    )

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": REFLECTOR_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
    )
    latency_ms = int((time.perf_counter() - start) * 1000)
    tokens = response.usage.total_tokens
    raw = response.choices[0].message.content.strip()

    failure_reason, lesson, next_strategy = "", "", ""
    for line in raw.split("\n"):
        if line.startswith("Failure reason:"):
            failure_reason = line.replace("Failure reason:", "").strip()
        elif line.startswith("Lesson:"):
            lesson = line.replace("Lesson:", "").strip()
        elif line.startswith("Next strategy:"):
            next_strategy = line.replace("Next strategy:", "").strip()

    evidence_titles = [c.title for c in example.context]
    reflection = ReflectionEntry(
        attempt_id=attempt_id,
        failure_reason=failure_reason or judge.reason,
        lesson=lesson or raw,
        next_strategy=next_strategy or raw,
        evidence_titles=evidence_titles,
    )
    return reflection, tokens, latency_ms
