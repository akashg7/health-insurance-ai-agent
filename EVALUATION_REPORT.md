# Empirical Evaluation & Benchmark Report
**Project:** PolicyLens — Health Insurance Conversational AI Agent  
**Evaluation Date:** 2026-11-26T17:01:18Z  
**Scope:** Real end-to-end `AgentExecutor`, real ChromaDB chunks, real `compute_claim_payout` engine (no mocked outputs)

---

## Executive Summary of Measured Metrics

| Evaluation Dimension | Baseline / Direct LLM | Agentic System (Measured) | Improvement |
|---|---|---|---|
| **Claim Calculation Accuracy** | 100.0% | **100.0% (0.0% Error)** | Deterministic sub-limit + co-pay math |
| **End-to-End Agent Tool Routing** | N/A | **100.0%** (18 real queries) | Real multi-tool ReAct execution |
| **Zero Page-0 Fabrication** | N/A | **100.0%** | Grounded citation guardrails |
| **Prompt Token Footprint** | 653.4 tokens/turn | **237.0 tokens/turn** | **63.73% reduction** |
| **End-to-End Agent Turn Latency** | — | **P50: 1.851s \| P90: 5.79s** | Measured over full ReAct loops |

*Generated automatically by `scripts/evaluate_and_benchmark.py`.*
