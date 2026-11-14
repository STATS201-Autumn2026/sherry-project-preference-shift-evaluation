# Beyond Recommendation Accuracy — Preference Shift Pilot

**Research question:** To what extent do aggregate recommendation accuracy metrics capture models’ ability to adapt to evolving user preferences?

## Completed analysis
Amazon short-term AgentRecBench pilot: 100 users / 458 historical interactions, 52 users eligible for TVD measurement; mean TVD approximately 0.593. A second-level product-category distribution is computed from the earlier and recent halves of each eligible user’s pre-target history. Each half requires at least two interactions and complete category data.

## Reproduce
1. Open `preference_shift_pilot.ipynb` in Colab.
2. Set `REVISION` to the upstream AgentRecBench commit SHA after verification.
3. Run all cells in order. The notebook downloads the source data, streams large item metadata as JSONL, computes TVD, and creates outputs.
4. Compare generated CSV and Figures 2/4 to the submitted proposal.

## Data source
https://huggingface.co/datasets/SGJQovo/AgentRecBench

**Upstream revision:** TO BE PINNED. **Code revision:** record GitHub commit after upload. **License:** verify before redistributing derived data.

## Scope and limitations
The pilot measures observed category-distribution shifts, not psychological preference changes. Model HR@5 evaluation and recent-history ablation are **planned**, not completed. The derived CSV omits original user IDs; this is pseudonymization, not a guarantee of full anonymity.
