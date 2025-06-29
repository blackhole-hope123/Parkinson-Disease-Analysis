# Parkinson’s Disease Analysis: A Multifaceted Data Science Approach

This project explores the progression of Parkinson’s Disease (PD) using large-scale longitudinal clinical data. We aim to enhance disease understanding through predictive modeling, unsupervised learning, and survival analysis.

## Project Goals

### Goal I: Predict Disease Progression
- **Target**: UPDRS Part I, II, III scores
- **Model**: LightGBM with time-aware features
- **Result**: MAE = 3.53 (49.4% improvement over baseline)

### Goal II: Cluster Progression Patterns
- **Method**: K-Means clustering on trajectory features
- **Result**: Two distinct clusters with differing medication profiles and progression rates

### Goal III: Analyze Freezing of Gait
- **Method**: Cox Proportional Hazards Model
- **Result**: Medication status significantly increases risk of freezing (p = 1.91 × 10⁻¹¹)

## Repository Structure

```bash
.
├── Question 1 predictive modelling      # Predictive modeling for UPDRS scores
├── Question 2 Clustering/               # Progression patterns clustering exploration 
├── Question 3 Cox PH Analysis/          # Survival analysis for freezing of gait
├── executive summary/                   # Summary reports and slides
├── new-data/                            # Raw datasets, also in the folder for each question
├── presentation/                        # Final presentation materials
└── readme.md                            # You're here!
```

## Data

We used Version 4.0 of the AMP® PD dataset, which includes longitudinal clinical data across four major cohort studies: BioFIND, PPMI, PDBP, and HBS. Analysis was based on Unified Parkinson's Disease Rating Scale (UPDRS) scores, a standardized test for assessing Parkinson's disease progression.

Over 20,000 observations from 4,000 participants were used after strict filtering and data leakage prevention measures.

## Methodology Highlights

- **GroupKFold** to prevent data leakage
- **Lagged features** and **rolling statistics** for time-series modeling
- **MICE imputation** for missing values
- **Permutation testing** to assess biospecimen data utility
- **Feature engineering** and **hyperparameter tuning**

## Results

- Time-aware features were most predictive
- Clustering revealed biologically meaningful subgroups
- Survival analysis identified key risk factors for freezing of gait

## Installation

```bash
pip install -r requirements.txt



## Reports and Slides

- The [executive summary](./executive%20summary) provides a concise overview of findings.
- The [presentation folder](./presentation) contains our final slides for stakeholder communication.

## Team

Min Shi, Sayantan Roy Chowdhury, E.G.K.M.Gamlath
