
from pathlib import Path
import pandas as pd

RESULT_DIR = Path("research/results")


def main():
    comparison_file = RESULT_DIR / "ann_rbm_comparison.csv"
    threshold_file = RESULT_DIR / "ann_test_thresholds.csv"
    rbm_threshold_file = RESULT_DIR / "rbm_threshold_analysis.csv"
    ann_threshold_file = RESULT_DIR / "ann_threshold_analysis.csv"

    required_files = [
        comparison_file,
        threshold_file,
        rbm_threshold_file,
        ann_threshold_file,
    ]

    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing result files:\n" + "\n".join(missing)
        )

    comparison = pd.read_csv(comparison_file)
    ann_test = pd.read_csv(threshold_file)
    rbm_thresholds = pd.read_csv(rbm_threshold_file)
    ann_validation = pd.read_csv(ann_threshold_file)

    # Correct the displayed model name in the report copy only.
    comparison["Model"] = comparison["Model"].replace({
        "RBM_ANN_Hybrid": "RBM_Logistic_Hybrid"
    })

    # Keep the original experiment results unchanged.
    comparison.to_csv(
        RESULT_DIR / "final_model_comparison.csv",
        index=False
    )

    ann_test.to_csv(
        RESULT_DIR / "final_ann_test_thresholds.csv",
        index=False
    )

    # Create a readable Markdown report from existing results.
    report = [
        "# DetectBench AI - Research Results",
        "",
        "## 1. Model comparison",
        "",
        comparison.round(4).to_markdown(index=False),
        "",
        "Note: Runtime values are from the original experiment. "
        "Hybrid runtime does not include the full RBM training time.",
        "",
        "## 2. ANN independent test-set threshold evaluation",
        "",
        ann_test.round(4).to_markdown(index=False),
        "",
        "## 3. ANN validation threshold analysis",
        "",
        ann_validation.round(4).to_markdown(index=False),
        "",
        "## 4. RBM threshold analysis",
        "",
        rbm_thresholds.round(4).to_markdown(index=False),
        "",
        "## 5. Interpretation",
        "",
        "- The ANN has the strongest ROC-AUC in the original model comparison.",
        "- ANN threshold 0.3 has higher test recall and fewer missed attacks.",
        "- ANN threshold 0.4 has a higher test F1-score and fewer false positives.",
        "- The RBM-only detector has low recall with the current configuration.",
        "- Threshold choices should be based on validation data; the test set "
        "should be reserved for final evaluation.",
        "",
        "## 6. Limitations",
        "",
        "- The ANN training emitted a ConvergenceWarning at 100 iterations.",
        "- Model comparisons should account for the different evaluation runs "
        "and thresholds.",
        "- These results alone do not establish production security effectiveness.",
        "",
    ]

    report_path = RESULT_DIR / "final_research_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("=== Final research outputs created ===")
    print(RESULT_DIR / "final_model_comparison.csv")
    print(RESULT_DIR / "final_ann_test_thresholds.csv")
    print(report_path)
    print("\n=== Model comparison preview ===")
    print(comparison.round(4).to_string(index=False))
    print("\n=== ANN test thresholds preview ===")
    print(ann_test.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
