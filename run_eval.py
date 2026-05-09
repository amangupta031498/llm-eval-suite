import json
import sys
from datetime import datetime
from groq_model import GroqModel
from evaluator import run_evaluation, save_report
import os

def main():
    with open("test_cases.json") as f:
        test_cases = json.load(f)

    model = GroqModel()
    results = run_evaluation(test_cases, model)
    report = save_report(results)

    total = report["summary"]["total"]
    passed = report["summary"]["passed"]
    pass_rate = report["summary"]["pass_rate"]

    print(f"\n{\'─\'*40}")
    print(f"  Results : {passed}/{total} passed ({pass_rate}%)")
    print(f"{\'─\'*40}\n")

    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    main()
