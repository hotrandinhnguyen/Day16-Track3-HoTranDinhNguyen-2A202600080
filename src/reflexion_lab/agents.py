from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .real_runtime import actor_answer, evaluator, reflector
from .mock_runtime import FAILURE_MODE_BY_QID
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        for attempt_id in range(1, self.max_attempts + 1):
            answer, actor_tokens, actor_latency = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            judge, eval_tokens, eval_latency = evaluator(example, answer)
            token_estimate = actor_tokens + eval_tokens
            latency_ms = actor_latency + eval_latency
            trace = AttemptTrace(attempt_id=attempt_id, answer=answer, score=judge.score, reason=judge.reason, token_estimate=token_estimate, latency_ms=latency_ms)
            final_answer = answer
            final_score = judge.score
            if judge.score == 1:
                traces.append(trace)
                break
            
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection, ref_tokens, ref_latency = reflector(example, attempt_id, judge, wrong_answer=answer)
                reflection_memory.append(reflection.next_strategy)
                reflections.append(reflection)
                token_estimate += ref_tokens
                latency_ms += ref_latency
            trace.token_estimate = token_estimate
            trace.latency_ms = latency_ms
            traces.append(trace)
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = "none" if final_score == 1 else FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)

class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)

@dataclass
class LATSAgent:
    max_attempts: int = 3
    branches: int = 2

    def run(self, example: QAExample) -> RunRecord:
        traces: list[AttemptTrace] = []
        reflections: list[ReflectionEntry] = []
        total_tokens = 0
        total_latency = 0
        final_answer = ""
        final_score = 0
        current_judge = None

        # Attempt 1: linear như ReAct
        answer, actor_tokens, actor_latency = actor_answer(example, 1, "lats", [])
        judge, eval_tokens, eval_latency = evaluator(example, answer)
        attempt_tokens = actor_tokens + eval_tokens
        attempt_latency = actor_latency + eval_latency
        total_tokens += attempt_tokens
        total_latency += attempt_latency
        final_answer = answer
        final_score = judge.score
        current_judge = judge
        trace = AttemptTrace(attempt_id=1, answer=answer, score=judge.score, reason=judge.reason,
                             token_estimate=attempt_tokens, latency_ms=attempt_latency)
        traces.append(trace)

        # Branching attempts
        for attempt_id in range(2, self.max_attempts + 1):
            if final_score == 1:
                break

            best_answer = final_answer
            best_judge = current_judge
            best_reflection = None
            branch_tokens = 0
            branch_latency = 0

            for _ in range(self.branches):
                refl, ref_tokens, ref_latency = reflector(example, attempt_id, current_judge, wrong_answer=final_answer)
                branch_tokens += ref_tokens
                branch_latency += ref_latency

                b_answer, ba_tokens, ba_latency = actor_answer(example, attempt_id, "lats", [refl.next_strategy])
                b_judge, be_tokens, be_latency = evaluator(example, b_answer)
                branch_tokens += ba_tokens + be_tokens
                branch_latency += ba_latency + be_latency

                if best_reflection is None:
                    best_answer, best_judge, best_reflection = b_answer, b_judge, refl

                if b_judge.score == 1:
                    best_answer, best_judge, best_reflection = b_answer, b_judge, refl
                    break

            total_tokens += branch_tokens
            total_latency += branch_latency
            reflections.append(best_reflection)
            final_answer = best_answer
            final_score = best_judge.score
            current_judge = best_judge
            trace = AttemptTrace(attempt_id=attempt_id, answer=final_answer, score=final_score,
                                 reason=best_judge.reason, token_estimate=branch_tokens, latency_ms=branch_latency)
            traces.append(trace)

        failure_mode = "none" if final_score == 1 else FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer,
                         agent_type="lats", predicted_answer=final_answer, is_correct=bool(final_score),
                         attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency,
                         failure_mode=failure_mode, reflections=reflections, traces=traces)
