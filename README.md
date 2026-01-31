# Malta Traffic Accident Analysis - Machine Learning Project

**Course:** ICS5110 Applied Machine Learning  
**Institution:** University of Malta  
**Deadline:** January 31, 2026  
**Student:** Naomi Thornley and Giulia-Maria Montebonello 

**Repository:** https://github.com/MLTeam11/Malta-Traffic-Accident-ML

## Project Overview

This project applies machine learning techniques to predict traffic accident severity in Malta using textual data from police press releases and news articles. We extract structured features from unstructured text and compare three traditional ML approaches.

## Research Questions

1.	How accurately can machine learning predict whether an accident will result in minor or severe injuries?
2.	Which features (time, location, vehicle type, weather) matter most for predicting accident severity?
3.	Does motorcycle involvement increase the severity of accidents in Malta?

## Datasets

- **Police Press Releases:** [N] records from Malta Police Force
- **News Articles:** [N] records from Maltese media outlets
- **Period:** [2024/2025]
- **External Data:** Weather observations, geographic coordinates, public holidays

## Methods

We implement and compare three traditional machine learning approaches:

- **Logistic Regression** - Interpretable baseline model
- **Random Forest** - Ensemble method for handling non-linear relationships
- **Support Vector Machine (SVM)** - Effective classification with RBF kernel

## Repository Structure
```
Malta-Traffic-Accident-ML/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 1_data_preparation.ipynb          # Data extraction and cleaning
│   ├── 2_exploratory_analysis.ipynb      # EDA and visualizations
│   ├── 3a_logistic_regression_naomi.ipynb
│   ├── 3b_random_forest_giulia.ipynb
│   ├── 3c_svm_giulia.ipynb
│   └── 4_results_comparison.ipynb        # Model comparison
├── data/
│   ├── raw/                              # Original datasets (not committed)
│   ├── processed/                        # Cleaned datasets
│   └── external/                         # Weather, geographic data
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   └── visualization.py
├── outputs/
│   ├── figures/                          # Plots for report
│   └── models/                           # Saved model files
└── docs/
    └── report.pdf                        # Final report
```

## Setup Instructions
```bash
# Clone the repository
git clone https://github.com/MLTeam11/Malta-Traffic-Accident-ML.git
cd Malta-Traffic-Accident-ML

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

## Requirements

- Python 3.8+
- pandas, numpy, scipy
- scikit-learn
- matplotlib, seaborn
- jupyter
- nltk

## Key Results

[To be updated after analysis completion]

- **Best Model:** [TBD]
- **Accuracy:** [TBD]
- **Most Important Features:** [TBD]
- **Malta vs Gozo:** [TBD]
- **Motorcycle Risk:** [TBD]

## Ethical Considerations

This project addresses:
- Geographic fairness (Malta vs. Gozo equity)
- Temporal bias (night shift workers)
- Proxy variables (location encoding socioeconomics)
- Deployment risks (false negative vs. false positive costs)
