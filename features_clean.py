import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def clean_updrs1(features_path, output_path, demographics_path='demographics_new.csv'):
    sns.set_style("whitegrid")
    
    # Load data
    print("Loading data...")
    features = pd.read_csv(features_path)
    demographics_new = pd.read_csv(demographics_path)
    
    print("\nInitial Info:")
    print(features.info())
    print("\nFirst few rows:")
    print(features.head())

    # Filter based on participant_id in demographics
    print("\nFiltering participants using demographics_new.csv...")
    features = features[features['participant_id'].isin(demographics_new['participant_id'])]
    
    print(f"Number of unique values in each column:\n{features.nunique()}")
    print(f"\nNumber of NaNs in GUID: {features['GUID'].isna().sum()}")

    # Drop GUID column
    print("\nDropping GUID column...")
    features.drop('GUID', axis=1, inplace=True)
    
    # Check for duplicates
    duplicate_count = features.duplicated(subset=['visit_month', 'participant_id']).sum()
    print(f"\nNumber of duplicate rows based on (visit_month, participant_id): {duplicate_count}")

    # Show potential duplicates
    if duplicate_count > 0:
        print("\nInspecting potential duplicates:")
        features_duplicates = features.duplicated(subset=['visit_month', 'participant_id'], keep=False)
        print(features[features_duplicates])
    else:
        print("No duplicate visit_month-participant_id entries found.")

      # Save cleaned file
    print(f"\nSaving cleaned data to {output_path}...")
    features.to_csv(output_path, index=False)
    print("Done.")

    if __name__ == "__main__":
        if len(sys.argv) < 3:
            print("Usage: python clean_updrs1.py <features_csv_path> <output_csv_path>")
        else:
            features_path = sys.argv[1]
            output_path = sys.argv[2]
            clean_updrs1(features_path, output_path=output_path)