# DetectBench AI - Research Results

## 1. Model comparison

| Model               |   Accuracy |   Precision |   Recall |     F1 |   ROC_AUC |   PR_AUC |   False_Positive_Rate |   False_Positives |   Missed_Attacks |   True_Negatives |   True_Positives |   Runtime_Seconds |
|:--------------------|-----------:|------------:|---------:|-------:|----------:|---------:|----------------------:|------------------:|-----------------:|-----------------:|-----------------:|------------------:|
| MLP_ANN             |     0.8669 |      0.8225 |   0.967  | 0.8889 |    0.9758 |   0.9824 |                0.2556 |              9458 |             1498 |            27542 |            43834 |          106.201  |
| RBM_Anomaly         |     0.4585 |      0.5897 |   0.0544 | 0.0997 |    0.6346 |   0.624  |                0.0464 |              1717 |            42864 |            35283 |             2468 |           13.7176 |
| RBM_Logistic_Hybrid |     0.7353 |      0.8233 |   0.6611 | 0.7333 |    0.8189 |   0.8758 |                0.1739 |              6433 |            15363 |            30567 |            29969 |            0.7776 |

Note: Runtime values are from the original experiment. Hybrid runtime does not include the full RBM training time.

## 2. ANN independent test-set threshold evaluation

| Model   |   Threshold |   Accuracy |   Precision |   Recall |     F1 |   ROC_AUC |   PR_AUC |   False_Positive_Rate |   False_Positives |   Missed_Attacks |   True_Negatives |   True_Positives |   Training_Seconds |
|:--------|------------:|-----------:|------------:|---------:|-------:|----------:|---------:|----------------------:|------------------:|-----------------:|-----------------:|-----------------:|-------------------:|
| MLP_ANN |         0.3 |     0.8216 |      0.7607 |   0.9863 | 0.8589 |    0.9759 |   0.9825 |                0.3802 |             14066 |              621 |            22934 |            44711 |            131.433 |
| MLP_ANN |         0.4 |     0.8467 |      0.7922 |   0.9781 | 0.8754 |    0.9759 |   0.9825 |                0.3144 |             11631 |              994 |            25369 |            44338 |            131.433 |

## 3. ANN validation threshold analysis

|   Threshold |   Precision |   Recall |     F1 |   False_Positive_Rate |   False_Positives |   Missed_Attacks |   True_Negatives |   True_Positives |   ROC_AUC |
|------------:|------------:|---------:|-------:|----------------------:|------------------:|-----------------:|-----------------:|-----------------:|----------:|
|         0.3 |      0.9435 |   0.9835 | 0.9631 |                0.1254 |              1405 |              394 |             9795 |            23475 |    0.9912 |
|         0.4 |      0.9579 |   0.9677 | 0.9627 |                0.0907 |              1016 |              772 |            10184 |            23097 |    0.9912 |
|         0.5 |      0.9671 |   0.9537 | 0.9603 |                0.0691 |               774 |             1106 |            10426 |            22763 |    0.9912 |
|         0.6 |      0.975  |   0.9395 | 0.9569 |                0.0514 |               576 |             1443 |            10624 |            22426 |    0.9912 |
|         0.7 |      0.9835 |   0.9173 | 0.9493 |                0.0328 |               367 |             1973 |            10833 |            21896 |    0.9912 |
|         0.8 |      0.9899 |   0.8907 | 0.9377 |                0.0194 |               217 |             2609 |            10983 |            21260 |    0.9912 |

## 4. RBM threshold analysis

|   Threshold_Percentile |   Threshold_Value |   Precision |   Recall |     F1 |   False_Positive_Rate |   False_Positives |   Missed_Attacks |   True_Negatives |   True_Positives |   ROC_AUC |   RBM_Training_Seconds |
|-----------------------:|------------------:|------------:|---------:|-------:|----------------------:|------------------:|-----------------:|-----------------:|-----------------:|----------:|-----------------------:|
|                     90 |           18.109  |      0.5807 |   0.1079 | 0.182  |                0.0955 |              3532 |            40440 |            33468 |             4892 |    0.6346 |                13.3838 |
|                     95 |           85.6274 |      0.5897 |   0.0544 | 0.0997 |                0.0464 |              1717 |            42864 |            35283 |             2468 |    0.6346 |                13.3838 |
|                     97 |          122.112  |      0.5716 |   0.032  | 0.0605 |                0.0294 |              1086 |            43883 |            35914 |             1449 |    0.6346 |                13.3838 |
|                     99 |          218.966  |      0.6646 |   0.012  | 0.0236 |                0.0074 |               275 |            44787 |            36725 |              545 |    0.6346 |                13.3838 |

## 5. Interpretation

- The ANN has the strongest ROC-AUC in the original model comparison.
- ANN threshold 0.3 has higher test recall and fewer missed attacks.
- ANN threshold 0.4 has a higher test F1-score and fewer false positives.
- The RBM-only detector has low recall with the current configuration.
- Threshold choices should be based on validation data; the test set should be reserved for final evaluation.

## 6. Limitations

- The ANN training emitted a ConvergenceWarning at 100 iterations.
- Model comparisons should account for the different evaluation runs and thresholds.
- These results alone do not establish production security effectiveness.
