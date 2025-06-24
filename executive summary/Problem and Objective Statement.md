# Introduction

Parkinson’s Disease (PD) is a progressive neurodegenerative disorder that impairs movement, cognition, sleep, and other essential functions. Over 1.1 million individuals are currently living with PD in the United States alone, and this number is expected to grow to 1.2 million by 2030, according to the Parkinson’s Foundation. PD is the second-most common neurodegenerative disease after Alzheimer’s, presenting a significant and growing public health challenge.
Despite its prevalence, the biological mechanisms behind PD are not yet fully understood, particularly which protein or peptide abnormalities contribute to disease onset and progression. This lack of understanding hinders the development of targeted therapies and slows down drug discovery efforts. Our goal in this project is to investigate the link between protein and peptide expression levels and the progression of Parkinson’s Disease, with the broader aim of enabling earlier diagnosis and more targeted treatments.

# The DataSet And Objective

The data comes from a research study conducted by the Accelerating Medicines Partnership® Parkinson's Disease, containing longitudonal information for 1019 participants. The severity of the disease progression for PD is assessed through their Unified Parkinson’s Disease Rating Scale (UPDRS) scores. 
- For 771 participants, we have Parkinson’s severity scores (UPDRS) collected over a 3-year period at regular 6-month intervals.
- For 248 participants, data is available on the protein and peptide levels, along with Parkinson’s severity scores (UPDRS), collected over a 9-year period at varying intervals of 3, 6, and 12 months.

To elaborate, the original dataset for the 248 participants includes the following features: 
- `patient_id`
- `visit_month`
- `UniProt` (indicating the type of the protein)
- `Peptide` (indicating the peptide structure)
- `PeptideAbundance` (indicating the frequency of the amino acid in the sample)
- `NPX` (indicating the normalized frequency of the protein's occurrence in the sample). 


 We use the UPDRS scores as the target variable to model disease progression. (Remark: Explain in plain languages) These scores have been simplified into binary classes (0 and 1) to represent different stages of symptom severity. The goal is to predict these progression categories using the protein and peptide-level data available to us. The stakeholders for this project include Parkinson’s patients, medical researchers, and pharmaceutical companies focused on developing targeted therapies. 

# Challenges
