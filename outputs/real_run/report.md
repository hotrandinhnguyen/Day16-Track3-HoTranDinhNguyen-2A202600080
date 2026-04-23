# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpotqa_100.json
- Mode: real
- Records: 300
- Agents: lats, react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.74 | 0.87 | 0.13 |
| Avg attempts | 1 | 1.41 | 0.41 |
| Avg token estimate | 1705.5 | 2563.25 | 857.75 |
| Avg latency (ms) | 2254.52 | 4081.08 | 1826.56 |

## Failure modes
```json
{
  "react": {
    "none": 74,
    "wrong_final_answer": 26
  },
  "reflexion": {
    "none": 87,
    "wrong_final_answer": 13
  },
  "lats": {
    "none": 86,
    "wrong_final_answer": 14
  },
  "combined": {
    "none": 247,
    "wrong_final_answer": 53
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding
- mini_lats_branching

## Discussion
Reflexion improved Exact Match by 13.0% over ReAct (74.00% -> 87.00%), demonstrating that iterative self-reflection is effective for multi-hop QA tasks. 13 questions that ReAct answered incorrectly were fixed by Reflexion through reflection memory. The primary remaining failure mode is 'wrong_final_answer', affecting 13 questions even after multiple attempts. This suggests the evaluator occasionally fails to provide actionable feedback, limiting the reflector's ability to course-correct. The cost of Reflexion is a 50.3% increase in token usage compared to ReAct, plus additional latency per retry. This tradeoff is worthwhile when accuracy is critical and the failure mode is recoverable through explicit reasoning strategy changes. Future improvements could include memory compression to avoid prompt bloat across attempts, and an adaptive retry limit that stops early when the reflection strategy has converged.
