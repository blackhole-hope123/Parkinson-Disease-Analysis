# The Problem

Parkinson’s disease (PD) is a disabling brain disorder that affects movements, cognition, sleep, and other normal functions.There are a large number of people sufferring from PD - an estimated 1.1 million people in the U.S. are living with PD and This number is expected to rise to 1.2 million by 2030, according to the Parkinson's Foundation. Parkinson's is the second-most common neurodegenerative disease after Alzheimer's disease.

However, it is not clear which protein or peptide abnormalities lead to PD and as a result, an effective targeted therapy cannot be delivered. In this project, we are looking to make some contribution in this direction.

# The DataSet

The dataset comes from a research study conducted by the Accelerating Medicines Partnership® Parkinson's Disease, with data for 1019 participants. There, 248 participants were tested on the level of a variety of peptides and proteins and their PD progression status are also accessed, while the other 771 participants are only accessed for their PD progression status. The target variable in these datasets are the Unified Parkinson's Disease Rating Scales, which concerns the severity of key symptoms of PD patients. The original dataset only has a handful features, `visit_month`(a contatenation of `visit_month` and `patient_id`), `visit_month`, `patient_id`, `UniProt` (indicating the type of the protein), `Peptide` (indicating the peptide structure), `PeptideAbundance` (indicating the frequency of the amino acid in the sample), `NPX` (indicating the normalized frequency of the protein's occurrence in the sample). So it is necessary to do an initial feature engineering. 



# The Objective

We plan to predict the PD's progression over time by the protein and peptide abundance data. The stakeholder will be millions of the PD patients, research institutes and Drug discovery companies. We plan to measure the outcome of this project by the robustness of our observation and the precision of our predictions.
