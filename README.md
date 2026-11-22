# Beyond Recommendation Accuracy — Preference Shift and LLM Recommendation Failure

## Research Question

Can temporal changes in users' product-category preferences improve the prediction of LLM recommendation failures beyond conventional user-history features?

This project investigates whether Total Variation Distance (TVD) provides useful information about recommendation reliability, particularly when users' observed category preferences change over time.

## Research Progress

### Attempt 1 — Initial Exploration

The initial pilot used the Amazon Short-term subset of AgentRecBench.

- 100 users and 458 historical interactions.
- 52 users eligible for TVD measurement.
- Mean TVD approximately 0.593.
- 14 of 52 users had TVD = 1.

**Key limitation:** Sparse interaction histories could inflate TVD, while small high- and low-TVD groups made accuracy comparisons unstable.

Files:
- `original_exploration_notebook.ipynb`
- `preference_shift_pilot.ipynb`

### Attempt 2 — Permutation-Adjusted Measurement and Recommendation Pilot

Following peer evaluation, Attempt 2 expanded the analysis to the Amazon Long-term subset and introduced a chance-adjusted measurement framework.

**Measurement:**
- 1,420 eligible users.
- 500 within-user category-order permutations.
- Mean Raw TVD: 0.525.
- Mean Shuffled TVD: 0.487.
- Mean Excess TVD: 0.038.

Excess TVD is defined as the difference between observed TVD and the mean TVD under randomized category ordering.

**Recommendation pilot:**
- Model: Gemini 3.1 Flash-Lite.
- 10 project-constructed next-item recommendation tasks.
- 20 candidate items per task.
- Hit@5: 60%.
- Failure@5: 40%.

These preliminary results demonstrate the feasibility of generating recommendation outcomes. They do not yet establish that TVD predicts recommendation failure.

All Attempt 2 materials are available in [`attempt2/`](./attempt2/), including the Colab notebook, measurement data, recommendation tasks, experiment configuration, pilot results, and figures.

## Planned Predictive Evaluation

The next stage will compare three logistic regression models:

| Model | Predictors |
|---|---|
| M0 | History length + category diversity |
| M1 | M0 + Raw TVD |
| M2 | M0 + Excess TVD |

The primary evaluation metric will be held-out log loss, with ROC-AUC and calibration as secondary checks.

The full-scale LLM recommendation experiment and predictive comparison remain future work.

## Reproducibility

For the revised experiment, see:

- [Attempt 2 README](./attempt2/README.md)
- [Attempt 2 Colab Notebook](./attempt2/attempt2_pilot.ipynb)
- [Measurement Results](./attempt2/data/amazon_tvd_final.csv)
- [Gemini Pilot Results](./attempt2/results/gemini_pilot_10.json)

The Gemini API key must be configured separately through Google Colab Secrets. API availability and model outputs may vary.

## Data Sources

Original dataset: [AgentRecBench](https://huggingface.co/datasets/SGJQovo/AgentRecBench)

Project dataset repository: [Hugging Face](https://huggingface.co/datasets/DKU-STATS201-Autumn2026/sherry_project_preference_shift_evaluation)

Source dataset revision and redistribution permissions should be verified before treating the released data as a fully reproducible archival snapshot.

## Limitations

TVD measures changes in observed category distributions, not necessarily underlying psychological preferences. Permutation adjustment provides a random-ordering benchmark but does not establish causality. Candidate sampling may also affect recommendation difficulty.

The current 10-task pilot is too small to support conclusions about the predictive value of TVD.
