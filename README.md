# Predicting Customer Churn and Designing Data-Driven Retention Strategies Using Machine Learning

## 1. Project Overview

Customer attrition poses a critical operational and financial threat in subscription-based ecosystems. This project establishes an end-to-end machine learning and behavioral analytics framework designed to predict customer churn with high predictive fidelity and translate these empirical findings into actionable, data-driven retention strategies. By synthesizing predictive modeling with consumer analytics, the repository provides a reproducible pipeline from raw customer telemetry to strategic business intervention.

---

## 2. Repository Architecture

```text
├── data/
│   ├── raw/
│   │   └── Dataset.csv
│   └── processed/
│       ├── cleaned_dataset.csv
│       └── processed_churn_data.csv
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb
│   ├── 02_feature_engineering_cltv.ipynb
│   └── 03_model_training_evaluation.ipynb
├── models/
│   └── best_churn_model.pkl
├── visualizations/
│   ├── churn_proportion_by_contract.png
│   ├── confusion_matrix_xgboost.png
│   ├── correlation_heatmap.png
│   ├── customer_age_distribution.png
│   ├── customer_tenure_distribution.png
│   ├── monthly_charges_boxplot.png
│   ├── retention_strategy_summary.png
│   └── top_features_importance.png
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 3. Methodology & Data Engineering

### Data Preprocessing & Leakage Control

The workflow begins with exhaustive exploratory data analysis and data cleansing (`01_data_cleaning_eda.ipynb`). To prevent artificial inflation of predictive performance, variables associated with direct post-hoc leakage (such as explicit termination reasons or post-churn scores) were systematically removed, ensuring the model operates under genuine predictive constraints.

### Feature Engineering & Class Balancing

Feature spaces were extended to capture customer lifecycle and behavioral dynamics (`02_feature_engineering_cltv.ipynb`). To address the inherent class imbalance typical of customer retention datasets, the **Synthetic Minority Over-sampling Technique (SMOTE)** was implemented exclusively across training folds, preserving an uncompromised holdout dataset for unbiased validation.

---

## 4. Model Evaluation & Performance

A comparative multi-model evaluation framework was executed (`03_model_training_evaluation.ipynb`):

- **Logistic Regression:** Established a linear baseline to benchmark feature coefficients.
- **Decision Trees & Random Forests:** Captured complex, non-linear feature thresholds.
- **XGBoost:** Outperformed alternative architectures, yielding near-optimal discrimination capabilities (F1-Score and ROC-AUC approaching upper thresholds).

The finalized **XGBoost** classifier has been serialized into `models/best_churn_model.pkl` for immediate deployment and inference integration.

---

## 5. Key Empirical Findings & Visual Insights

Through rigorous feature importance extraction and visual diagnostics (`visualizations/`), several core behavioral drivers of attrition were identified:

- **Satisfaction Dominance:** `SatisfactionScore` serves as the primary psychological and operational catalyst dictating customer defection.
- **Contractual & Temporal Vulnerabilities:** Analyses of customer tenure (`customer_tenure_distribution.png`) and contract types (`churn_proportion_by_contract.png`) indicate that month-to-month subscribers and newly onboarded cohorts carry exponentially higher attrition risks.

---

## 6. Strategic Retention Framework

Empirical insights from the modeling pipeline support three targeted retention interventions:

1. **Automated Early-Warning Triggers:** Real-time monitoring of customer satisfaction metrics to flag at-risk accounts before service cancellation occurs.
2. **Contract Structuring:** Incentivizing transition pathways from month-to-month agreements to long-term commitments via tailored pricing models.
3. **Onboarding Reinforcement:** Concentrating customer success resources during the critical early-tenure window to suppress early-stage attrition.

---

## 7. Conclusion

This project successfully bridges the gap between advanced predictive modeling and strategic business decision-making. By leveraging high-performing machine learning architectures like XGBoost and coupling them with rigorous feature interpretability, enterprises can transition from reactive customer service models to proactive, data-driven retention frameworks. The serialized model and reproducible pipeline provide a scalable foundation for mitigating financial losses associated with customer churn.

---

## 8. Installation & Setup

To replicate the computational environment, install the requisite packages via:

```bash
pip install -r requirements.txt
```

👤 Author

Mohammad Sourav

Interdisciplinary Researcher | Consumer Psychology, Artificial Intelligence & Data Science
