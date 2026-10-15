# abide-combat-harmonization
 Exploratory evaluation of ComBat site harmonization on fMRI connectivity features and its impact on autism subgroup stability and site confounds (ABIDE). Python, nilearn, scikit-learn, neuroCombat.



# Cross-Site fMRI Harmonization with ComBat (ABIDE)

## Question
Does ComBat site harmonization remove scanner hardware artifacts from functional connectivity features, and are connectivity-based autism subgroups stable across sites once site confounds are eliminated?

## Data
ABIDE I preprocessed fMRI (CPAC pipeline, CC200 atlas), via nilearn. 871 subjects across 20 scanner sites (403 autism, 468 neurotypical controls).

## Method
1. **Functional Connectivity:** Correlation between 200 ROI time series per subject ($200 \times 200$ matrix).
2. **Harmonization:** Applied empirical Bayes ComBat (`neuroCombat`) using site as batch, protecting `diagnosis` (categorical) and `age` (continuous) as biological covariates.
3. **Subtype Evaluation:** PCA (10 components) + K-means ($k=2$) on autism subjects pre- and post-harmonization.
4. **Validation:**
   - **Site Confound:** Chi-square test of independence between cluster labels and scanner site.
   - **Cross-Site Stability:** Adjusted Rand Index ($ARI$) calculated across 30 random half-site permutations.

## Results

| Feature Set | Subgroup Stability (ARI) | Cluster vs. Site Chi-Square ($p$-value) |
| :--- | :---: | :---: |
| **Raw Features** | 0.791 | $9.85 \times 10^{-4}$ (Significant Site Bias) |
| **ComBat Harmonized** | **0.728** | **0.972** (No Site Confound) |

## Key Findings
- **Elimination of Site Confound:** Applying ComBat successfully removed scanner-specific artifacts, shifting the cluster-vs-site chi-square $p$-value from highly significant ($p < 0.001$) to non-significant ($p = 0.972$).
- **Preserved Biological Stability:** Even after stripping out scanner batch effects, the connectivity-based autism subgroups maintained strong reproducibility across site splits ($ARI = 0.728$).

## Limitations & Next Steps
- **Limitations:** Retrospective dataset; simplified linear covariates in ComBat; single atlas (CC200).
- **Next Steps:** Evaluate nonlinear harmonization (e.g., CovBat / M-ComBat) and test whether harmonized features improve out-of-site classification accuracy.