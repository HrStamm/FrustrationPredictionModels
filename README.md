# Frustration Prediction from Heart Rate Data  

This repository contains the code and analyses for my course project in **Statistical Evaluation of Artificial Intelligence and Data (02445)** at DTU, June 2025.  

**Author:** Valdemar Stamm (s244742)  
**Study line:** Artificial Intelligence and Data  

---

## 📌 Project Overview  
This project investigates how machine learning models can be applied to predict **self-reported frustration levels** from physiological **heart rate (HR) features**.  

Two approaches were implemented and compared:  
- **Decision Tree (DT)** → simple and interpretable, well-suited for small datasets.  
- **Artificial Neural Network (ANN)** → capable of modeling more complex patterns, but more data-demanding.  

The dataset is a subset of the **EmoPairCompete** dataset, consisting of repeated measures of heart rate signals from individuals across rounds and phases of a problem-solving task. The prediction target is the self-reported frustration level (0–10).  

---

## 🧠 Key Methods  
- **Input features:** HR Mean, HR Median, HR Std, HR Min, HR Max, HR AUC  
- **Task type:** Classification (ordinal frustration levels)  
- **Cross-validation:** GroupKFold (grouped by individual) to avoid data leakage from repeated measures  
- **Model optimization:** Hyperparameter tuning via `GridSearchCV`  

---

## 📊 Results in Brief  
- Both models showed **low overall performance**, reflecting dataset challenges (small size, imbalance, high variability).  
- **Decision Tree:** Higher stability in precision/recall/F1 across folds.  
- **ANN:** Slightly higher mean accuracy, but less consistent.  
- **Statistical testing:** Paired t-tests showed no significant difference in performance between the models.  
- **Conclusion:** The limitations lie primarily in the dataset rather than the models. Larger, more balanced datasets are needed for reliable frustration prediction.  

---

## 📂 Repository Structure  
- `project_code.ipynb` → Interactive Jupyter Notebook with full analysis, visualizations, and outputs.  
- `project_code.py` → Standalone Python script with identical code but without inline outputs.  

---

## 🔬 Reflections and Future Work  
- Current dataset size and imbalance strongly limit model performance.  
- Overfitting risk was managed with careful cross-validation, but generalizability remains low.  
- Future directions:  
  - Collect larger and more diverse datasets.  
  - Test models on external data for robustness.  
  - Explore additional ML methods and feature engineering.  

---

## 📖 References  
- EmoPairCompete dataset: Das et al. (2024). *Physiological signals dataset for emotion and frustration assessment under team and competitive behaviors*. ICLR 2024 Workshop.  
- scikit-learn documentation for GroupKFold and GridSearchCV.  
