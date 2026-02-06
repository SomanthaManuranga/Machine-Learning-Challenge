# Machine Learning Challenge - Winter 2025

A comprehensive machine learning project tackling two distinct predictive modeling tasks: 
high-dimensional regression with ensemble methods and interpretable rule-based modeling 
for edge device deployment.

## Project Overview

This project addresses real-world ML challenges commonly encountered in industry:
- **Task 1**: Predict continuous variable from 273 features with complex non-linear relationships
- **Task 2**: Reverse-engineer simple, interpretable rules suitable for edge devices without ML libraries

### Dataset Characteristics
- **Training samples**: 10,000
- **Features**: 273 high-dimensional features
- **Targets**: 2 distinct continuous variables
- **Challenge**: Weak individual feature signals requiring advanced ensemble techniques

---

## Task 1: High-Dimensional Regression

### Objective
Predict continuous variable "target01" with maximum accuracy from 273 features exhibiting 
weak linear correlations (max |r| < 0.17) and predominantly non-linear relationships.

### Methodology Pipeline

1. **Data Analysis**
   - Correlation analysis: Highest correlation only -0.168 (feat_134)
   - No missing values detected
   - Target range: 0.01 to 1.07 (right-skewed distribution)

2. **Preprocessing**
   - Standardization: z = (x - μ) / σ
   - Feature scaling for equal contribution
   - Cross-validation pipeline to prevent data leakage

3. **Feature Selection**
   - Mutual Information (MI) analysis for non-linear dependencies
   - Top features: feat_134 (MI=0.171), feat_163 (MI=0.159)
   - **Decision**: Retained all 273 features (weak individual signals require ensemble capture)

4. **Model Development & Results**

| Rank | Model | R² Score | RMSE | MAE | Improvement |
|------|-------|----------|------|-----|-------------|
| 1 | **Stacking Ensemble** | **0.848** | **0.0897** | **0.0734** | **Baseline → 1700%** |
| 2 | CatBoost (Tuned) | 0.804 | 0.1008 | 0.0850 | +1607% |
| 3 | Voting Ensemble | 0.752 | 0.1144 | 0.0928 | +1500% |
| 4 | XGBoost (Tuned) | 0.663 | 0.1322 | 0.1105 | +1310% |
| 5 | LightGBM (Tuned) | 0.569 | 0.1494 | 0.1279 | +1111% |
| 6 | Random Forest | 0.079 | 0.2185 | 0.1979 | +68% |
| 7 | Ridge/Linear | 0.047 | 0.2223 | 0.1989 | Baseline |

### Technical Implementation

**Baseline Models** (R² ≈ 0.047)
- Linear/Ridge Regression confirmed non-linear relationships
- Random Forest slightly better (R²=0.079) but insufficient

**Gradient Boosting with Optuna Hyperparameter Tuning**
```python
# XGBoost: 30 trials, 5-fold CV → R²=0.663
# LightGBM: Leaf-wise growth → R²=0.569  
# CatBoost: Ordered boosting → R²=0.782 (best single model)
```

**Stacking Ensemble Architecture**
```
Base Models:
├── XGBoost (level-wise boosting)
├── LightGBM (leaf-wise boosting)
├── CatBoost (ordered boosting)
└── Random Forest (bagging diversity)

Meta-Learner: Ridge Regression with L2 regularization
Final Performance: R²=0.848, RMSE=0.0897
```

### Key Findings - Task 1

✅ **84.8% variance explained** (vs 4.7% baseline - 18x improvement)  
✅ **CatBoost best single model** due to ordered boosting preventing overfitting  
✅ **Stacking outperforms voting** - intelligent weighting beats simple averaging  
✅ **Feature importance**: feat_217 (28.5%), feat_184 (25.9%), feat_124 (13.0%) dominate  
✅ **Model generalization validated** - prediction distribution matches training data  

---

## Task 2: Interpretable Rule Discovery

### Objective
Discover simple if-else rules for "target02" deployable on edge devices without ML libraries.
Rules must use only basic comparisons and numerical operations.

### Methodology

#### Phase 1: Feature Identification
**Correlation Analysis**
- feat_4: **0.43** (dominant controller variable)
- feat_185: 0.08
- feat_13: 0.034

**Random Forest Feature Importance**
- feat_4: **65.65%** (primary decision controller)
- feat_185: **18.25%** (calculation variable)
- feat_13: **13.34%** (calculation variable)
- Combined: **96.94%** of variance

**Validation**: Both methods independently confirmed same 3 features → High confidence

#### Phase 2: Threshold Discovery

**Decision Tree Analysis** (Depth=3, R²=0.847)
```
Root Split: feat_4 ≤ 0.2
├── Left branch: feat_4 ≤ 0.2 → negative values (-0.85 to -2.55)
├── Middle: 0.2 < feat_4 ≤ 0.7 → positive values (0.22 to 1.17)
└── Right: feat_4 > 0.7 → mixed values (-0.41 to 0.6)
```

**Threshold Validation Results**
| Threshold | Low Mean | High Mean | Separation |
|-----------|----------|-----------|------------|
| 0.3 | -0.918 | +0.465 | Strong (Δ=1.38) |
| 0.5 | -0.317 | +0.417 | Moderate |
| 0.7 | +0.009 | +0.144 | Weak |

**Decision**: 3-region model with thresholds at **0.2 and 0.7**

#### Phase 3: Formula Derivation

**Linear Regression per Region**
- Region-specific coefficient estimation
- Initial R² with 0.5 threshold: 0.597 (insufficient)
- Improved with Decision Tree thresholds: R²=0.9067

**Grid Search Validation** ({-2, -1, 0, 1, 2})
- 2-region model: R²=0.375
- 3-region model: R²=**0.915** ✓
- Confirmed integer coefficients robust against noise

### Final Discovered Rules
```python
if feat_4 <= 0.2:
    target02 = -2 × feat_185 - feat_13
elif 0.2 < feat_4 <= 0.7:
    target02 = +2 × feat_185 - feat_13
else:  # feat_4 > 0.7
    target02 = -feat_185 + feat_13
```

**Performance**: R² = **91.50%**

### 📊 Evaluation Dataset Distribution
| Region | Condition | Samples | Percentage |
|--------|-----------|---------|------------|
| 1 | feat_4 ≤ 0.2 | 2,005 | 20.1% |
| 2 | 0.2 < feat_4 ≤ 0.7 | 4,930 | 49.3% |
| 3 | feat_4 > 0.7 | 3,065 | 30.6% |

Distribution matches training data → Confirms generalizability

### ✅ Edge Device Implementation
- Uses only: `numpy`, `pandas`, `operator`, `argparse`
- No ML libraries required at runtime
- Simple if-else structure with basic arithmetic
- Deployed in `framework_81.py`

---

## 🛠️ Technical Stack

**Languages & Core Libraries**
- Python 3.x
- NumPy, Pandas (data manipulation)
- Scikit-learn (preprocessing, baseline models)

**Machine Learning Frameworks**
- XGBoost (extreme gradient boosting)
- LightGBM (leaf-wise gradient boosting)
- CatBoost (ordered boosting)
- Optuna (hyperparameter optimization)

**Visualization**
- Matplotlib, Seaborn

**Development Tools**
- Jupyter Notebook (experimentation)
- Cross-validation pipelines
- Grid search algorithms

---

## 📈 Key Results Summary

### Task 1: Regression Performance
| Metric | Baseline | Final Model | Improvement |
|--------|----------|-------------|-------------|
| R² Score | 0.047 | **0.848** | **+1700%** |
| RMSE | 0.2223 | **0.0897** | **-60%** |
| MAE | 0.1989 | **0.0734** | **-63%** |

### Task 2: Rule Discovery
| Metric | Value | Notes |
|--------|-------|-------|
| R² Score | **91.50%** | Simple 3-region piecewise model |
| Features Used | **3 of 273** | 98.9% reduction in complexity |
| Rules | **3 if-else** | Fully interpretable |
| Edge Compatible | **Yes** | No ML libraries needed |


---

## **Quick One-Liner:**
```
ML Challenge: Stacking ensemble (R²=0.848) for high-dim regression + interpretable rules (R²=0.915) for edge devices 🚀
