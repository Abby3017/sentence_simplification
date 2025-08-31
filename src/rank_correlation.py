import numpy as np
import pandas as pd
from scipy.stats import spearmanr

if __name__ == "__main__":
    original_df = pd.read_csv("data/original_data_evaluated.csv")
    aug_classification = 'major_effect'
    # aug_classification = 'tansprasert_aug'
    # rand_insert_period_df = pd.read_csv(
    #     f"data/{aug_classification}/random_insert_period_augmented_evaluated.csv")
    # rand_insert_the_df = pd.read_csv(
    #     f"data/{aug_classification}/random_insert_the_augmented_evaluated.csv")
    # replace_longest_df = pd.read_csv(
    #     f"data/{aug_classification}/replace_longest_augmented_evaluated.csv")
    # replace_longest_rand_period_df = pd.read_csv(
    #     f"data/{aug_classification}/replace_longest_rand_period_augmented_evaluated.csv")
    # replace_random_period_df = pd.read_csv(
    #     f"data/{aug_classification}/replace_random_period_augmented_evaluated.csv")
    # replace_random_the_df = pd.read_csv(
    #     f"data/{aug_classification}/replace_random_the_augmented_evaluated.csv")

    duplicate_longest_word_df = pd.read_csv(
        f"data/{aug_classification}/duplicate_longest_word_augmented_evaluated.csv")
    duplicate_period_df = pd.read_csv(
        f"data/{aug_classification}/duplicate_period_augmented_evaluated.csv")
    prefix_sentence_with_df = pd.read_csv(
        f"data/{aug_classification}/prefix_sentence_with_augmented_evaluated.csv")
    duplicate_whole_sentence_df = pd.read_csv(
        f"data/{aug_classification}/duplicate_whole_sentence_augmented_evaluated.csv")
    duplicate_the_df = pd.read_csv(
        f"data/{aug_classification}/duplicate_the_augmented_evaluated.csv")
    randomly_replace_number_df = pd.read_csv(
        f"data/{aug_classification}/randomly_replace_number_augmented_evaluated.csv")

    # scores = ['sari', 'bertscore_f1', 'lens', 'lens_salsa']
    scores = ['sari', 'bertscore_f1', 'lens']
    models = original_df['model'].unique()
    text_types = original_df['text_type'].unique()

    # augmentations = {
    #     'Random Insert Period': rand_insert_period_df,
    #     'Random Insert The': rand_insert_the_df,
    #     'Replace Longest Word': replace_longest_df,
    #     'Replace Longest Word with Random Period': replace_longest_rand_period_df,
    #     'Replace Random Period': replace_random_period_df,
    #     'Replace Random The': replace_random_the_df
    # }

    augmentations = {
        # 'Duplicate The': duplicate_the_df,
        # 'Duplicate Longest Word': duplicate_longest_word_df,
        # 'Duplicate Period': duplicate_period_df,
        # 'Prefix Sentence With': prefix_sentence_with_df,
        'Duplicate Whole Sentence': duplicate_whole_sentence_df
        # 'Randomly Replace Number': randomly_replace_number_df
    }

    rank_correlations = []
    for text_type in text_types:
        for score in scores:
            original_subset = original_df[(
                original_df['text_type'] == text_type)]
            for aug_name, aug_df in augmentations.items():

                aug_subset = aug_df[(aug_df['text_type'] == text_type)]

                aug_model_complex_pairs = set(
                    zip(aug_subset['dir'], aug_subset['complex']))

                # Filter original_subset to only include model-complex pairs present in aug_subset
                filtered_original_subset = original_subset[
                    original_subset.apply(lambda row: (
                        row['model'], row['complex']) in aug_model_complex_pairs, axis=1)
                ]

                filtered_original_subset = filtered_original_subset.sort_values(
                    ['model', 'complex']).reset_index(drop=True)
                aug_subset = aug_subset.sort_values(
                    ['dir', 'complex']).reset_index(drop=True)

                # Calculate average scores for original and augmented data
                original_avg = filtered_original_subset.groupby('model')[
                    score].mean()
                aug_avg = aug_subset.groupby('dir')[score].mean()

                # Calculate Spearman correlation between original and augmented averages
                avg_correlation, avg_p_value = spearmanr(original_avg, aug_avg)
                # bottom_10_models = original_avg.nsmallest(3)
                top_10_models = original_avg.nlargest(3)
                top_10_aug = aug_avg[top_10_models.index]
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
                # bottom_10_aug = aug_avg[bottom_10_models.index]
                # augmented_bottom_3_ranks = aug_ranks_all[bottom_10_models.index].values
                # bottom_3_ranks = np.array([1, 2, 3])
                # bottom_3_rmse = np.sqrt(np.mean((original_top_3_ranks - augmented_bottom_3_ranks) ** 2))
                # bottom_3_rmse = round(bottom_3_rmse, 4)
                top_10_aug_correlation, _ = spearmanr(
                    top_10_models, top_10_aug)
                # bottom_10_aug_correlation, _ = spearmanr(
                #     bottom_10_models, bottom_10_aug)
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
        f"data/{aug_classification}/top3_correlation_major_dws.csv", index=False)
    # print(
    #     f"Rank correlation results saved to data/{aug_classification}/rank_correlation.csv")
