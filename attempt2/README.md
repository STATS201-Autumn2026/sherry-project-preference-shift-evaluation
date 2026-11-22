# Attempt 2 — Preference Shift and Recommendation Failure Prediction

## Research Question

Can temporal changes in users' product-category preferences
help predict LLM recommendation failures beyond conventional
user-history features?

## Changes from Attempt 1

The initial study focused on comparing recommendation accuracy
between high- and low-TVD user groups.

Peer evaluation identified two concerns:
1. Short interaction histories could inflate TVD scores
   (14 of 52 users had TVD = 1).
2. Accuracy comparisons based on 26 users per group
   could be unstable.

Attempt 2 introduces:
- Amazon Long-term data (1,420 eligible users).
- 500 within-user permutations to estimate chance-level TVD.
- Excess TVD = Raw TVD - Mean Shuffled TVD.
- A prediction-focused framework using Failure@5.

## Dataset

Source: AgentRecBench (Shang et al., 2025)
https://huggingface.co/datasets/SGJQovo/AgentRecBench

Associated paper:
https://doi.org/10.52202/085713-5156

Derived measurement results and recommendation tasks
are stored in the `data/` folder.

Dataset revision: TO BE VERIFIED.

## Method

### 1. Preference-Shift Measurement

Each user's chronological history is divided into
earlier and recent halves.

Total Variation Distance (TVD) measures differences
between second-level product-category distributions.

A 500-permutation within-user shuffle test estimates
the TVD expected under random interaction ordering.

### 2. Recommendation Pilot

Model: gemini-3.1-flash-lite
Temperature: 0
Prompt version: v1.0

Each task contains:
- One held-out target product.
- 19 sampled negative candidates.
- User history excluding the target interaction.

Failure@5 = 1 if the target is absent from the top five.

These are project-constructed candidate sets,
not official AgentRecBench recommendation tasks.

### 3. Planned Prediction Evaluation

Three logistic regression models will be compared:

M0: History length + category diversity
M1: M0 + Raw TVD
M2: M0 + Excess TVD

Primary metric: Held-out log loss.
Additional checks: ROC-AUC and calibration.

The full prediction-model comparison has not yet been run.

## Reproduction

### Requirements

- Python 3
- Google Colab
- pandas
- numpy
- matplotlib
- google-genai
- A Google Gemini API key

### Run

1. Open `attempt2_pilot.ipynb` in Google Colab.
2. Run the notebook cells in order.
3. Configure `GEMINI_API_KEY` in Colab Secrets
   before executing Gemini API requests.
4. Review the generated measurement and pilot results.

Note: API availability and model outputs may vary.
Re-running the notebook may not reproduce identical
recommendation rankings.

## Recorded Results

Measurement sample: 1,420 users

Mean Raw TVD: 0.525
Mean Shuffled TVD: 0.487
Mean Excess TVD: 0.038

Gemini pilot:
- Valid rankings: 10/10 after retry.
- Hit@5: 0.60
- Failure@5: 0.40

## Files

- `attempt2_pilot.ipynb`: Colab experiment notebook.
- `data/amazon_tvd_final.csv`: TVD measurements.
- `data/amazon_recommendation_tasks.json`: Candidate sets.
- `results/gemini_pilot_10.json`: Recommendation pilot.
- `config/recommendation_experiment_config.json`: Model settings.
- `figures/`: Proposal Figures 2 and 4.

## Limitations

Raw TVD can be affected by short user histories.
Permutation adjustment provides a random-ordering
reference but does not establish true preference changes.

Candidate sampling affects recommendation difficulty.

The 10-task pilot establishes feasibility only;
it does not demonstrate the predictive value of TVD.
