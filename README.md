# ⚖️ Uncovering Bias and Ensuring Fairness: A COMPAS Algorithm Analysis

This project conducts a comprehensive fairness analysis and bias mitigation strategy on the **COMPAS dataset** — a system used to predict recidivism risk in criminal justice. We evaluate the ethical implications of algorithmic decision-making and implement fairness-aware machine learning using **Random Forest**, **Gradient Boosting**, and **FairLearn reduction techniques**.

---

## 👥 Author

- **Dev Divyendh Dhinakaran** (G01450299)

---

## 📦 Technologies Used

- **Python (scikit-learn, pandas, numpy)**
- **FairLearn** (`ExponentialGradient`, `DemographicParity`)
- **Matplotlib / Seaborn** for visualizations
- **Jupyter Notebook**

---

## 📊 Dataset

- 📥 [COMPAS Dataset on Kaggle](https://www.kaggle.com/datasets/danofer/compass)
- 18,316 entries | 52 attributes  
- Contains:
  - Defendant demographics
  - Criminal history
  - COMPAS risk scores
  - Sentencing outcomes

---

## 📋 Project Workflow

### 🔍 1. Data Preprocessing
- Null handling and label encoding
- Dropped 39+ irrelevant columns (e.g., IDs, names, timestamps)
- Feature engineering: `juv_fel_seriousness = juv_fel_count - juv_misd_count`

### ⚖️ 2. Bias Analysis
- COMPAS showed clear racial and gender disparities:
  - African Americans had **45% higher scores**
  - Women were **19.4% more likely** to receive higher scores than men
  - Youth (<25) were **2.5x more likely** to receive higher scores than middle-aged defendants

### 🔬 3. Fairness Mitigation Strategies

#### ✅ **Resampling with RandomOverSampler**
- Balanced representation across races using oversampling

#### ✅ **Fairness Reduction using FairLearn**
- **Exponential Gradient**: Prioritized underrepresented groups dynamically  
- **Demographic Parity**: Equalized prediction rates across racial/gender groups

---

## 🤖 Models & Experiments

| Experiment | Description |
|------------|-------------|
| **Exp 1** | Baseline Random Forest |
| **Exp 2** | Tuned Random Forest |
| **Exp 3** | Fairness-Aware Ensemble (Random Forest + Gradient Boosting) with FairLearn |

### 📈 Final Model Results (Experiment 3)

| Metric        | Value     |
|---------------|-----------|
| Accuracy      | 87.42%    |
| Precision     | 88%       |
| Recall        | 86%       |
| F1 Score      | 87%       |

#### 📊 Selection Rate by Race

| Race Group | Selection Rate |
|------------|----------------|
| Race 0     | 54.95%         |
| Race 1     | 57.24%         |
| Race 2     | 47.13%         |
| Race 3     | 57.06%         |
| Race 4     | 37.50%         |
| Race 5     | 50.00%         |


---

## 🧠 Key Learnings

- Fairness in ML requires both algorithmic adjustment and data-level interventions
- Ensemble methods offer strong baselines but can still inherit bias from data
- FairLearn provides scalable methods for enforcing fairness constraints

---
