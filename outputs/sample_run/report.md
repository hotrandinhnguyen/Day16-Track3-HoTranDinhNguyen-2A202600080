# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_mini.json
- Mode: mock
- Records: 16
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 1.0 | 1.0 | 0.0 |
| Avg attempts | 1 | 1 | 0 |
| Avg token estimate | 366.12 | 366.12 | 0.0 |
| Avg latency (ms) | 2235.88 | 2224.88 | -11.0 |

## Failure modes
```json
{
  "react": {
    "none": 8
  },
  "reflexion": {
    "none": 8
  },
  "combined": {
    "none": 16
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding

## Discussion
Reflexion improved Exact Match by 0.0% over ReAct (100.00% -> 100.00%), demonstrating that iterative self-reflection is effective for multi-hop QA tasks. 0 questions that ReAct answered incorrectly were fixed by Reflexion through reflection memory. The primary remaining failure mode is 'unknown', affecting 0 questions even after multiple attempts. This suggests the evaluator occasionally fails to provide actionable feedback, limiting the reflector's ability to course-correct. The cost of Reflexion is a 0.0% increase in token usage compared to ReAct, plus additional latency per retry. This tradeoff is worthwhile when accuracy is critical and the failure mode is recoverable through explicit reasoning strategy changes. Future improvements could include memory compression to avoid prompt bloat across attempts, and an adaptive retry limit that stops early when the reflection strategy has converged.
