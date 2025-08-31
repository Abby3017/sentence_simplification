import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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

    augmentations = {
        'Duplicate WS 2-1': duplicate_ws_2_1_df,
        'Duplicate WS 2-2': duplicate_ws_2_2_df,
        'Insert Random Punctuation 2': insert_random_punctuation_2_df,
        'Upper Case Modification': upper_case_df,
        'Prefix WS': prefix_ws_df,
        'Prefix ZWS': prefix_zws_df,
        'Suffix WS': suffix_ws_df,
        'Suffix ZWS': suffix_zws_df
    }

    rank_correlations = []
    for text_type in text_types:
        for score in scores:
            original_subset = original_df[(
                original_df['text_type'] == text_type)]
            for aug_name, aug_df in augmentations.items():

                aug_subset = aug_df[(aug_df['text_type'] == text_type)]

                original_subset = original_subset.sort_values(
                    ['model', 'complex']).reset_index(drop=True)
                aug_subset = aug_subset.sort_values(
                    ['model', 'complex']).reset_index(drop=True)

                # Calculate average scores for original and augmented data
                original_avg = original_subset.groupby('model')[score].mean()
                aug_avg = aug_subset.groupby('model')[score].mean()

                avg_correlation, avg_p_value = spearmanr(original_avg, aug_avg)
                top_10_models = original_avg.nlargest(3)
                top_3_aug_models = aug_avg.nlargest(3)
                original_winner = original_avg.idxmax()
                augmented_winner = aug_avg.idxmax()
                winner_preserved = (original_winner == augmented_winner)
                original_top_3_ranks = np.array([1, 2, 3])
                aug_ranks_all = (-aug_avg).argsort().argsort() + 1
                augmented_top_3_ranks = aug_ranks_all[top_10_models.index].values
                top_3_rmse = np.sqrt(
                    np.mean((original_top_3_ranks - augmented_top_3_ranks) ** 2))
                top_3_rmse = round(top_3_rmse, 4)
                bottom_10_models = original_avg.nsmallest(10)
                top_10_aug = aug_avg[top_10_models.index]
                bottom_10_aug = aug_avg[bottom_10_models.index]
                top_10_aug_correlation, _ = spearmanr(
                    top_10_models, top_10_aug)
                bottom_10_aug_correlation, _ = spearmanr(
                    bottom_10_models, bottom_10_aug)
                rank_correlations.append({
                    'text_type': text_type,
                    'augmentation': aug_name,
                    'score_type': score,
                    'top_3_models': top_10_models.index.tolist(),
                    'top_3_aug_models': top_3_aug_models.index.tolist(),
                    'top_3_correlation': round(top_10_aug_correlation, 4),
                    'winner_preserved': winner_preserved,
                    'top_3_rmse': top_3_rmse,
                })
    rank_correlation_df = pd.DataFrame(rank_correlations)
    rank_correlation_df.to_csv(
        f"data/{aug_classification}/top3_correlation_minor.csv", index=False)
    # print(
    #     f"Rank correlation results saved to data/{aug_classification}/rank_correlation.csv")
