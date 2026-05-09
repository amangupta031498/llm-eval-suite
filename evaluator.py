import json
from datetime import datetime
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase

def run_evaluation(test_cases_raw, model):
    results = []

    for raw in test_cases_raw:
        tc = LLMTestCase(
            input=raw["input"],
            actual_output=raw["actual_output"],
            expected_output=raw["expected_output"],
            retrieval_context=raw["context"],
            context=raw["context"]
        )

        metrics = [
            AnswerRelevancyMetric(threshold=0.7, model=model, include_reason=True),
            FaithfulnessMetric(threshold=0.7, model=model, include_reason=True),
            HallucinationMetric(threshold=0.3, model=model, include_reason=True),
        ]

        case_result = {"id": raw["id"], "scores": {}, "passed": {}}

        for metric in metrics:
            metric.measure(tc)
            name = metric.__class__.__name__
            case_result["scores"][name] = round(metric.score, 3)
            case_result["passed"][name] = metric.is_successful()

        case_result["overall_pass"] = all(case_result["passed"].values())
        results.append(case_result)
        print(f"{raw['id']} — {'PASS' if case_result['overall_pass'] else 'FAIL'}")

    return results

def save_report(results):
    passed = sum(1 for r in results if r["overall_pass"])
    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": round(passed / len(results) * 100, 1)
        },
        "results": results
    }
    with open("eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Report saved — eval_report.json")
    return report
