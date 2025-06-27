
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import sys


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