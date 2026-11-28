# Consumer Heterogeneity and LLM Recommendation Failure — Attempt 3

STATS 201 | Duke Kunshan University | Collaborative final project

## Research question
Can historical consumer behavioral heterogeneity and preference shifts predict LLM recommendation failure, and can risk estimates inform fixed-budget targeting?

## Data and methods
Source: [AgentRecBench, Amazon long-term tasks](https://huggingface.co/datasets/SGJQovo/AgentRecBench) (Shang et al., 2025). The measurement sample includes 1,420 users. The final pilot contains 73 valid rankings (28 failures) from 100 sampled tasks; 80 were attempted. Failure@5 indicates that the held-out target ranks below fifth place. Candidates are target-conditioned, so this is a benchmark rather than a deployment evaluation.

Features include log history length, category entropy, permutation-adjusted TVD (500 shuffles), excess adjacent repetition, and exploration rate. L2 logistic regression and restricted random forests are compared using identical stratified five-fold splits repeated ten times. Metrics include log loss, ROC-AUC and Brier score. A separate fixed-budget simulation uses initial out-of-fold scores.

## Preliminary results
No model beats the naive mean log loss of 0.6660. Random Forest M3 improves on RF M0 (0.7209 vs 0.7515), but captures only seven of 28 failures at a budget of 15 users, within the simulated random range of 2–9. Results are exploratory, not causal or externally validated.

## Reproduction
`Project_Attempt3.ipynb` contains the analysis and figure code, with saved outputs removed for privacy. Run in Colab with pandas, numpy, scikit-learn, matplotlib, requests and google-genai; configure GEMINI_API_KEY through Colab Secrets. API availability and model responses may differ. Figures 2–4 and B1 are in `figures/`. Aggregate evaluation tables are in `results/`. The original notebook and private checkpoints are not included. User-level tables are withheld pending redistribution and disclosure review. Preserve Attempt 2 in the existing repository.

## Future research
Prediction → causal inference → optimization; measurement and explanation support the roadmap, with contextual bandits and reinforcement learning as potential extensions.

## Responsible AI and limitations
AI-assisted coding and drafting should be disclosed according to the course requirements; authors should verify all outputs. Interaction histories are behavioral proxies, not direct evidence of motives. The pilot is small and affected by API noncompletion. No actual intervention effects are established.
