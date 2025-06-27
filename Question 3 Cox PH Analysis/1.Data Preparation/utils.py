
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import sys


class GroupedTemporalHoldoutSplitter:
    """
    Custom scikit-learn compatible splitter for train/test and cross-validation
    based on a temporal holdout strategy for a subset of groups (patients).

    - For a single split (n_splits=1):
        A `test_patient_ratio` of patients are designated as "temporal holdout" patients.
        For these patients, data at or before `cutoff_month` is training data,
        and data after `cutoff_month` is test data.
        All data from other patients go entirely into the training set.

    - For cross-validation (n_splits > 1):
        In each fold, a distinct subset of patients (approximately 1/n_splits of total patients)
        serves as the "temporal holdout" group.
        For these holdout patients, their data at or before `cutoff_month` is added to the
        training set for that fold, and their data after `cutoff_month` forms the
        validation set for that fold.
        All data from patients not in the current holdout group go entirely into the
        training set for that fold.
    """
    def __init__(self, cutoff_month, n_splits=5, test_patient_ratio=0.2,
                 random_state=None, shuffle_patients=True):
        """
        Args:
            cutoff_month (int): The visit_month used to split the data for
                                temporal holdout patients.
            n_splits (int): Number of splitting iterations in the cross-validator.
                            If 1, performs a single train/test split.
            test_patient_ratio (float): Proportion of patients to designate as
                                        temporal holdout in a single split (n_splits=1).
                                        Ignored if n_splits > 1.
            random_state (int, optional): Seed for shuffling patient IDs.
                                          Ensures reproducibility.
            shuffle_patients (bool): Whether to shuffle patient IDs before splitting.
                                     Default is True.
        """
        if not isinstance(n_splits, int) or n_splits <= 0:
            raise ValueError("n_splits must be a positive integer.")
        if not isinstance(cutoff_month, int) or cutoff_month < 0:
            raise ValueError("cutoff_month must be a non-negative integer.")
        if n_splits == 1 and not (0 < test_patient_ratio < 1):
            raise ValueError("test_patient_ratio must be between 0 and 1 (exclusive) for a single split.")

        self.cutoff_month = cutoff_month
        self.n_splits = n_splits
        self.test_patient_ratio = test_patient_ratio
        self.random_state = random_state
        self.shuffle_patients = shuffle_patients

    def get_n_splits(self, X=None, y=None):
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Args:
            X (pd.DataFrame): DataFrame containing the data. Must include a
                              'visit_month' column.
            y (array-like, optional): The target variable. Not used in this
                                      splitting logic directly.
            groups (pd.Series): Series containing the group labels (e.g., patient IDs),
                                aligned with the index of X.

        Yields:
            train_idx (np.array): The training set indices (positional) for that split.
            test_idx (np.array): The testing set indices (positional) for that split.
        """
        if groups is None:
            raise ValueError("The 'groups' parameter cannot be None and should contain patient IDs.")
        if not isinstance(X, pd.DataFrame) or 'visit_month' not in X.columns:
            raise ValueError("X must be a pandas DataFrame with a 'visit_month' column.")
        if not isinstance(groups, pd.Series) or not X.index.equals(groups.index):
            raise ValueError("X.index and groups.index must be aligned.")

        unique_patient_ids = groups.unique()

        original_indices_map = {original_idx: pos_idx for pos_idx, original_idx in enumerate(X.index)}

        if self.n_splits == 1:
            # --- Single Train/Test Split ---
            if len(unique_patient_ids) < 2:
                 raise ValueError(f"Not enough unique patients ({len(unique_patient_ids)}) to perform a train/test split "
                                  f"with test_patient_ratio={self.test_patient_ratio}.")

            # Split unique patient IDs into 'full_train_patients' and 'temporal_holdout_patients'
            # The 'test_size' in train_test_split refers to the proportion for the second returned array.
            full_train_patient_ids, temporal_holdout_patient_ids = train_test_split(
                unique_patient_ids,
                test_size=self.test_patient_ratio, # This proportion becomes temporal_holdout_patients
                random_state=self.random_state,
                shuffle=self.shuffle_patients
            )

            # Handle cases where one group might be empty due to small numbers and rounding
            if len(temporal_holdout_patient_ids) == 0 and len(full_train_patient_ids) > 0:
                temporal_holdout_patient_ids = np.array([full_train_patient_ids[-1]])
                full_train_patient_ids = full_train_patient_ids[:-1]
            if len(full_train_patient_ids) == 0 and len(temporal_holdout_patient_ids) > 0:
                 full_train_patient_ids = np.array([temporal_holdout_patient_ids[-1]])
                 temporal_holdout_patient_ids = temporal_holdout_patient_ids[:-1]
            if len(temporal_holdout_patient_ids) == 0 or len(full_train_patient_ids) == 0:
                 raise ValueError("Could not create a valid split. One of the patient groups is empty.")

            train_mask = pd.Series(False, index=X.index)
            test_mask = pd.Series(False, index=X.index)
            train_mask[groups.isin(full_train_patient_ids)] = True
            is_temporal_holdout_mask = groups.isin(temporal_holdout_patient_ids)
            train_mask[is_temporal_holdout_mask & (X['visit_month'] <= self.cutoff_month)] = True
            test_mask[is_temporal_holdout_mask & (X['visit_month'] > self.cutoff_month)] = True

            train_indices_pos = np.array([original_indices_map[idx] for idx in X.index[train_mask]])
            test_indices_pos = np.array([original_indices_map[idx] for idx in X.index[test_mask]])

            if len(test_indices_pos) == 0:
                raise ValueError("Test set is empty for the single split. "
                                 "Ensure temporal holdout patients have data after the cutoff_month.")
            if len(train_indices_pos) == 0:
                 raise ValueError("Train set is empty for the single split. This is unexpected.")

            yield train_indices_pos, test_indices_pos

        else:
            # --- Cross-Validation Split ---
            if len(unique_patient_ids) < self.n_splits:
                raise ValueError(f"Number of unique patients ({len(unique_patient_ids)}) "
                                 f"is less than n_splits ({self.n_splits}). "
                                 "Reduce n_splits or provide more diverse patient data.")

            patient_folder = KFold(n_splits=self.n_splits, shuffle=self.shuffle_patients,
                                   random_state=self.random_state)

            folds_generated = 0
            for _, val_patient_indices_in_unique_list in patient_folder.split(unique_patient_ids):
                # Patients designated as temporal holdout for this CV fold
                cv_temporal_holdout_ids = unique_patient_ids[val_patient_indices_in_unique_list]

                cv_train_mask = pd.Series(False, index=X.index)
                cv_val_mask = pd.Series(False, index=X.index)

                # Patients NOT in cv_temporal_holdout_ids have all their data in training for this fold
                is_full_train_for_cv_fold_mask = ~groups.isin(cv_temporal_holdout_ids)
                cv_train_mask[is_full_train_for_cv_fold_mask] = True
                is_cv_temporal_holdout_mask = groups.isin(cv_temporal_holdout_ids)
                cv_train_mask[is_cv_temporal_holdout_mask & (X['visit_month'] <= self.cutoff_month)] = True
                cv_val_mask[is_cv_temporal_holdout_mask & (X['visit_month'] > self.cutoff_month)] = True

                train_indices_pos = np.array([original_indices_map[idx] for idx in X.index[cv_train_mask]])
                val_indices_pos = np.array([original_indices_map[idx] for idx in X.index[cv_val_mask]])

                if len(val_indices_pos) == 0:
                    print(f"Warning: CV Fold - Validation set is empty for temporal_holdout_ids: {list(cv_temporal_holdout_ids)}. Skipping this fold.")
                    continue
                if len(train_indices_pos) == 0:
                    print(f"Warning: CV Fold - Training set is empty. Skipping this fold.")
                    continue

                yield train_indices_pos, val_indices_pos
                folds_generated += 1

            if folds_generated == 0 and self.n_splits > 0 :
                 raise ValueError("No valid CV splits were generated. Check data and cutoff_month. "
                                  "It's possible no temporal holdout patients had data after the cutoff month in any fold configuration.")




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