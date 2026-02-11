
The Reliability Curve and Coverage-vs-Accuracy plots are generated from the trained model.

Because the repository does not track large binary files or inference caches, these plots must be generated from your local checkpoint.

## How to Generate
1. Ensure you have a trained model in `checkpoints/best_model.pth`.
2. Run the generation script:
   ```bash
   python scripts/generate_plots.py
   ```
3. The following files will appear in this folder:
   - reliability_curve.png
   - coverage_vs_accuracy.png
