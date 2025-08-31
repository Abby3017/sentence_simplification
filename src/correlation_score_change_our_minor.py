import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler, minmax_scale

if __name__ == "__main__":
    original_df = pd.read_csv("data/original_data_evaluated.csv")
    aug_classification = 'minor_effect'
    input_folder = '/Users/abhinavkumar/prj/uzh/sentence_simplification/abby37kumar@outlookcom/repo/sentence_simplification/output/minor_effect'
    duplicate_ws_2_1_df = pd.read_csv(
        f"output/minor_effect/duplicate_ws_2_1_augmented_evaluated.csv")
    duplicate_ws_2_2_df = pd.read_csv(
        f"output/minor_effect/duplicate_ws_2_2_augmented_evaluated.csv")
    insert_random_punctuation_2_df = pd.read_csv(
        f"output/minor_effect/insert_random_punctuation_2_augmented_evaluated.csv")
    upper_case_df = pd.read_csv(
        f"output/minor_effect/upper_case_modification_augmented_evaluated.csv")

    prefix_ws_df = pd.DataFrame()
    prefix_zws_df = pd.DataFrame()
    suffix_ws_df = pd.DataFrame()
    suffix_zws_df = pd.DataFrame()

    for root, dirs, _ in os.walk(input_folder):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            df_model_prefix_ws = pd.read_csv(
                f"{dir_path}/prefix_whitespace_augmented_evaluated.csv")
            df_model_prefix_ws['model'] = dir
            prefix_ws_df = pd.concat(
                [prefix_ws_df, df_model_prefix_ws], ignore_index=True)
            df_model_prefix_zws = pd.read_csv(
                f"{dir_path}/prefix_zero_width_space_augmented_evaluated.csv")
            df_model_prefix_zws['model'] = dir
            prefix_zws_df = pd.concat(
                [prefix_zws_df, df_model_prefix_zws], ignore_index=True)
            df_model_suffix_ws = pd.read_csv(
                f"{dir_path}/suffix_whitespace_augmented_evaluated.csv")
            df_model_suffix_ws['model'] = dir
            suffix_ws_df = pd.concat(
                [suffix_ws_df, df_model_suffix_ws], ignore_index=True)
            df_model_suffix_zws = pd.read_csv(
                f"{dir_path}/suffix_zero_width_space_augmented_evaluated.csv")
            df_model_suffix_zws['model'] = dir
            suffix_zws_df = pd.concat(
                [suffix_zws_df, df_model_suffix_zws], ignore_index=True)

    scores = ['sari', 'bertscore_f1', 'lens', 'lens_salsa']
    models = original_df['model'].unique()
    text_types = original_df['text_type'].unique()

    # Create a list to store all score changes
    all_changes = []

    for text_type in text_types:
        for model in models:
            original_subset = original_df[(original_df['text_type'] == text_type) & (
                original_df['model'] == model)]
            duplicate_ws_2_1_subset = duplicate_ws_2_1_df[(duplicate_ws_2_1_df['text_type'] == text_type) & (
                duplicate_ws_2_1_df['model'] == model)]
            duplicate_ws_2_2_subset = duplicate_ws_2_2_df[(duplicate_ws_2_2_df['text_type'] == text_type) & (
                duplicate_ws_2_2_df['model'] == model)]
            insert_random_punctuation_2_subset = insert_random_punctuation_2_df[(insert_random_punctuation_2_df['text_type'] == text_type) & (
                insert_random_punctuation_2_df['model'] == model)]
            upper_case_subset = upper_case_df[(upper_case_df['text_type'] == text_type) & (
                upper_case_df['model'] == model)]
            prefix_ws_subset = prefix_ws_df[(prefix_ws_df['text_type'] == text_type) & (
                prefix_ws_df['model'] == model)]
            prefix_zws_subset = prefix_zws_df[(prefix_zws_df['text_type'] == text_type) & (
                prefix_zws_df['model'] == model)]
            suffix_ws_subset = suffix_ws_df[(suffix_ws_df['text_type'] == text_type) & (
                suffix_ws_df['model'] == model)]
            suffix_zws_subset = suffix_zws_df[(suffix_zws_df['text_type'] == text_type) & (
                suffix_zws_df['model'] == model)]

            if not original_subset.empty:
                original_means = original_subset[scores].mean()
                scaler_original = StandardScaler()
                original_scaled = scaler_original.fit_transform(
                    original_subset[scores])
                original_scaled_means = pd.Series(
                    original_scaled.mean(axis=0), index=scores)

                augmentations = {
                    'Duplicate WS 2-1': duplicate_ws_2_1_subset,
                    'Duplicate WS 2-2': duplicate_ws_2_2_subset,
                    'Insert Random Punctuation 2': insert_random_punctuation_2_subset,
                    'Upper Case Modification': upper_case_subset,
                    'Prefix WS': prefix_ws_subset,
                    'Prefix ZWS': prefix_zws_subset,
                    'Suffix WS': suffix_ws_subset,
                    'Suffix ZWS': suffix_zws_subset
                }

                for aug_name, aug_subset in augmentations.items():
                    if not aug_subset.empty:
                        aug_means = aug_subset[scores].mean()
                        scaler_aug = StandardScaler()
                        aug_scaled = scaler_aug.fit_transform(
                            aug_subset[scores])
                        aug_scaled_means = pd.Series(
                            aug_scaled.mean(axis=0), index=scores)
                        # Calculate percentage change: (new - old) / old * 100
                        score_changes = (
                            (aug_means - original_means) / original_means) * 100
                        score_changes = score_changes.round(4)
                        # import pdb
                        # pdb.set_trace()

                        original_subset = original_subset.sort_values(
                            'complex').reset_index(drop=True)
                        aug_subset = aug_subset.sort_values(
                            'complex').reset_index(drop=True)

                        # Add row for each score
                        for score in scores:
                            # use two different one for each aug and original for each score
                            original_scaled = minmax_scale(
                                original_subset[score])
                            original_scaled_mean = original_scaled.mean()
                            aug_scaled = minmax_scale(aug_subset[score])
                            aug_scaled_mean = aug_scaled.mean()
                            correlation, p_value = spearmanr(
                                original_subset[score], aug_subset[score])
                            correlation = round(correlation, 4)
                            all_changes.append({
                                'text_type': text_type,
                                'model': model,
                                'augmentation': aug_name,
                                'score_type': score,
                                'original_score': original_means[score],
                                'augmented_score': aug_means[score],
                                'score_change_percentage': score_changes[score],
                                'correlation': correlation,
                                'original_scaled_mean': original_scaled_mean,
                                'augmented_scaled_mean': aug_scaled_mean
                            })

    # Create DataFrame
    changes_df = pd.DataFrame(all_changes)
    changes_df.to_csv(
        f"data/{aug_classification}/score_changes_minor_scaled.csv", index=False)
