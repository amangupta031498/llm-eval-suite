# LLM Eval Suite
A lightweight LLM response quality evaluation framework built with DeepEval and Groq.

## What it does
Evaluates LLM outputs across 3 quality dimensions:
- Answer Relevancy — does the response address the question?
- Faithfulness — is the response grounded in retrieved context?
- Hallucination — does the response contradict known facts?

## Project structure
- groq_model.py — Custom Groq wrapper for DeepEval
- evaluator.py — Evaluation loop and report generation
- test_cases.json — Test case definitions
- eval_report.json — Auto-generated JSON report
- eval_report.html — Auto-generated HTML report

## Quickstart
1. pip install -r requirements.txt
2. Set GROQ_API_KEY in your environment
3. Open the notebook in Google Colab and run all cells

## Sample results
TC-001: Answer Relevancy 1.0, Faithfulness 1.0, Hallucination 0.0 — PASS
TC-002: Answer Relevancy 1.0, Faithfulness 1.0, Hallucination 0.0 — PASS
TC-003: Answer Relevancy 1.0, Faithfulness 0.5, Hallucination 1.0 — FAIL

## Tech stack
- DeepEval: LLM evaluation metrics
- Groq: fast inference using llama-3.3-70b-versatile
- Google Colab: execution environment
