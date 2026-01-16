# Neural Networks from Scratch: Mystery Dataset Classification

**Author:** Edmund Tsou  
**Institution:** Johns Hopkins University  
**Objective:** Create the best Multi-Layer Perceptron (MLP) to classify a mystery tabular dataset.

---

## Dataset Exploration
The project utilizes a "mystery" dataset provided in CSV format.

* **Training Set:** 8,000 examples with 205 features.
* **Test Set:** 2,000 examples with 205 features (unlabeled).
* **Labels:** 5 possible classes (0, 1, 2, 3, 4).



### Key Observations
* **Class Imbalance:** There is a clear imbalance in the training set, with Class 0 being the most frequent and Class 4 being the least.
* **Feature Correlation:** Features are roughly uncorrelated according to Pearson's correlation, indicating no significant multicollinearity.
* **Complexity:** PCA, t-SNE, and UMAP analyses suggest that class differences are subtle and rely on non-obvious patterns.

---

## Model Development & Experiments

### Baseline Model: XGBoost
* **Method:** 5-fold cross-validation with 660 estimators and inverse class weights.
* **Results:** Accuracy of 0.84. It performed significantly worse on smaller classes (e.g., Class 4 had a recall of 0.330).

### MLP Model 1: Standard MLP
* **Architecture:** 2 hidden layers (512, 256) with Batch Normalization.
* **Regularization:** Dropout (0.35) and Weight Decay (1e-4).
* **Results:** 0.874 Accuracy. This outperformed XGBoost but showed clear signs of overfitting.

### MLP Model 2: Focal Loss
* **Method:** Same architecture as Model 1 but utilized Focal Loss to focus on "hard" examples (smaller classes).
* **Results:** 0.857 Accuracy. Performance worsened, likely due to Focal Loss being sensitive to outliers.

### MLP Model 4: Wide & Deep NN
* **Architecture:** Deep layers (256, 128, 64) for generalization and a wide path feeding features directly to final logits.
* **Results:** 0.859 Accuracy. This approach helped stabilize training but did not improve overall accuracy.

---

## Final Model: Enhanced Regularization (Model 6)
Model 6 achieved the best results by focusing on light regularization and training stability.

* **Strategy:** Standard MLP architecture plus label smoothing, early stopping, and increased weight decay.
* **Results:** **0.879 Accuracy**.
* **Feature Importance:** Top features were identified using CV-averaged permutation importance, with feature `f132` (index 133) ranking as the most important.



---

## Summary & Takeaways
* **Complexity vs. Performance:** More advanced techniques often overcomplicated the model, leading to worse results.
* **Regularization is Key:** It is difficult to beat a simple MLP model when it is properly regularized.
* **Limits:** The significant class imbalance likely held the model back from passing the high 80s in accuracy.

---