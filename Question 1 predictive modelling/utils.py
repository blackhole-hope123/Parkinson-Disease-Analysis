
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import sys

def get_dummies_and_sift(df, column_name, rare_category_map=None, percent_threshold=0.01):
    """
    Creates dummy variables for a column after grouping rare categories.
    Can operate in two modes: 'learn' (if rare_category_map is None) or 'apply'.
    """

    if rare_category_map is None:
        threshold = len(df) * percent_threshold
        counts = df[column_name].value_counts()
        categories_to_lump = counts[counts < threshold].index.tolist()

    else:
        categories_to_lump = rare_category_map



    df_processed = df.copy()
    
    df_processed.loc[df_processed[column_name].isin(categories_to_lump), column_name] = f'Other_{column_name}'
    
    temp = pd.get_dummies(df_processed[column_name], drop_first=True)
    temp = temp * 1

    new_column_names = temp.columns.tolist()
    
    return temp, categories_to_lump, new_column_names

def create_all_features(df, updrs_cols, time_dependent_cols, dummy_map=None):
    """
    Takes a raw dataframe and applies all feature engineering steps.
    Uses a pre-learned map for dummy variable creation to prevent data leakage.
    """

    df_featured = df.copy()

    initial_feature_cols = []
    
    # --- 1. Operations in data preprocessing (Dummies and Mappings) ---
    categorical_cols = ['education_level_years', 'race', 'sex']
    
    if dummy_map is None:
        dummy_map = {}

    for col in categorical_cols:
        learned_map = dummy_map.get(col)
        dummies, learned_rare_cats = get_dummies_and_sift(df_featured, col, rare_category_map=learned_map)[:2]
        initial_feature_cols += dummies.columns.tolist()
        df_featured = pd.concat([df_featured, dummies], axis=1)
        if col not in dummy_map:
            dummy_map[col] = learned_rare_cats

    map_cols = ["caff_drinks_current_use",
                "caff_drinks_ever_used_regularly", 
                "biological_mother_with_pd", 
                "biological_father_with_pd", 
                "other_relative_with_pd"]
    for col in map_cols:
      if col in df_featured.columns:
        df_featured[col] = df_featured[col].map({'Yes': 1, 'No': 0})
    
    initial_feature_cols += map_cols


    df_featured=df_featured.rename(columns={"mds_updrs_part_i_summary_score":"updrs_1", 
                              "mds_updrs_part_ii_summary_score":"updrs_2", 
                              "mds_updrs_part_iii_summary_score":"updrs_3"})

    # --- 2. Time-Dependent Feature Engineering ---
    df_featured = df_featured.sort_values(['participant_id', 'visit_month'])
    cols_for_ts = updrs_cols + time_dependent_cols
    for lag in [1, 2, 3]:
        for col in cols_for_ts:
            df_featured[f'{col}_lag_{lag}'] = df_featured.groupby('participant_id')[col].shift(lag)

    lagged_cols = [col for col in df_featured.columns if '_lag_' in col]

    # Rate of Change / Slopes
    print("Adding change rate features...")
    for score in updrs_cols:

        df_featured[f'{score}_slope_1'] = df_featured[f'{score}_lag_1'] - df_featured[f'{score}_lag_2']

        df_featured[f'{score}_slope_2'] = df_featured[f'{score}_lag_2'] - df_featured[f'{score}_lag_3']

        df_featured[f'{score}_slope_mean'] = (df_featured[f'{score}_lag_1'] - df_featured[f'{score}_lag_3']) / 2

        
    # (Weighted) Rolling Averages
    print("Adding rolling average features...")
    for score in updrs_cols:
        df_featured[f'{score}_mean_lags'] = df_featured[lagged_cols].mean(axis=1)

        df_featured[f'{score}_weighted'] = (
            0.6 * df_featured[f'{score}_lag_1'] +
            0.3 * df_featured[f'{score}_lag_2'] +
            0.1 * df_featured[f'{score}_lag_3']
        )
    
    
    # Volatility Features
    print("Adding volatility features...")
    for score in updrs_cols:

        df_featured[f'{score}_lag_std'] = df_featured[lagged_cols].std(axis=1)

        df_featured[f'{score}_lag_range'] = df_featured[lagged_cols].max(axis=1) - df_featured[lagged_cols].min(axis=1)

    # Time-Adjusted Features
    print("Adding time-adjusted features...")
    for score in updrs_cols:

        df_featured[f'{score}_time_adj'] = df_featured[f'{score}_lag_1'] / (df_featured['visit_month'] + 2)

    # Patient age progression
    df_featured['age_progression'] = df_featured['age_at_baseline'] + df_featured['visit_month'] / 12


    # interaction feature
    df_featured['age_x_visit_month'] = df_featured['age_at_baseline'] * (df_featured['visit_month'] )

    # UPDRS Ratios
    print("Adding UPDRS ratio features...")
    df_featured['motor_ratio'] = df_featured['updrs_3_lag_1'] / (
        df_featured['updrs_1_lag_1'] +
        df_featured['updrs_2_lag_1'] +
        1e-5  # Prevents division by zero
    )
        
    return df_featured, dummy_map, initial_feature_cols





def averaging_scores(df, cols):
    """
    Return a dataframe where each patient has at most one score at a certain month, for people who originally have two or more scores, their average will be the value.
    """
    try:
        cols_to_average = cols

        cols_to_take_first = df.select_dtypes(exclude=np.number).columns.tolist()
        agg_dict = {col: 'mean' for col in cols}
        for col in cols_to_take_first:  
            agg_dict[col] = 'first'   
        agg_dict["visit_month"] = "max"
        screening_mask = df['visit_month'] < 0
        df_screening = df[screening_mask]
        df_regular = df[~screening_mask]

        cleaned_parts = []
        if not df_screening.empty:
            print("Processing screening records by participant...")
            screening_agg_dict = agg_dict.copy()

            def process_screening_group(group):
                averaged_data = group.agg(agg_dict)
                return averaged_data
        
            groups=df_screening.groupby(by=['participant_id'])
            df_screening_cleaned = process_screening_group(groups)
           
            df_screening_cleaned = df_screening_cleaned.drop("participant_id", axis=1)
            df_screening_cleaned = df_screening_cleaned.reset_index()
            
            if 'participant_id' in df_screening_cleaned.columns:
                pass
            df_screening_cleaned = df_screening_cleaned.drop_duplicates(subset=['participant_id'])


            cleaned_parts.append(df_screening_cleaned)
            print(f"Consolidated {len(df_screening)} screening records into {len(df_screening_cleaned)} baseline records.")


        if not df_regular.empty:
            print("Processing regular visit records by participant and month...")
            # For regular visits, group by participant AND visit_month
            df_regular_cleaned = df_regular.groupby(['participant_id', 'visit_month']).agg(agg_dict)
            cleaned_parts.append(df_regular_cleaned)
            print(f"Consolidated {len(df_regular)} regular records into {len(df_regular_cleaned)} records.")

        df_cleaned = pd.concat(cleaned_parts, ignore_index=True)

        df_cleaned['visit_name'] = df_cleaned['visit_name'].str.split('#', expand=True)[0]
        df_cleaned.loc[df_cleaned.visit_name=="SC","visit_month"]=-1

        return df_cleaned

        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def delete_patients_with_the_same_GUID(df):
    if "GUID" not in df.columns:
        print("There is no GUID column in the given dataframe")
    else:
        GUIDs=df["GUID"].unique()
        for GUID in GUIDs:
            if len(df[df["GUID"]==GUID].participant_id.unique())>=2:
                df=df[df["GUID"]!=GUID]
        return df

def delete_patients_without_GUID(df):
    if "GUID" not in df.columns:
        print("There is no GUID column in the given dataframe")
    else:
        return df.dropna(subset=['GUID'])

