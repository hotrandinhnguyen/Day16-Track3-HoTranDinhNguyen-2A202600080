from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Literal
from .real_runtime import actor_answer, evaluator, reflector
from .mock_runtime import FAILURE_MODE_BY_QID
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord
from .utils import partial_score

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

        # Branching attempts — parallel + partial score selection
        def run_branch(_b: int):
            refl, ref_tok, ref_lat = reflector(example, 0, current_judge, wrong_answer=final_answer)
            b_ans, ba_tok, ba_lat = actor_answer(example, 0, "lats", [refl.next_strategy])
            b_jdg, be_tok, be_lat = evaluator(example, b_ans)
            return b_ans, b_jdg, refl, ref_tok + ba_tok + be_tok, ref_lat + ba_lat + be_lat

        for attempt_id in range(2, self.max_attempts + 1):
            if final_score == 1:
                break

            with ThreadPoolExecutor(max_workers=self.branches) as ex:
                results = [f.result() for f in as_completed(ex.submit(run_branch, b) for b in range(self.branches))]

            branch_tokens = sum(r[3] for r in results)
            branch_latency = max(r[4] for r in results)
            total_tokens += branch_tokens
            total_latency += branch_latency

            correct = [(a, j, r) for a, j, r, _, _ in results if j.score == 1]
            if correct:
                best_answer, best_judge, best_reflection = correct[0]
            else:
                best_answer, best_judge, best_reflection = max(
                    [(a, j, r) for a, j, r, _, _ in results],
                    key=lambda x: partial_score(x[0], example.gold_answer)
                )

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
