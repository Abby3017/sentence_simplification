import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler, minmax_scale

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

    # Create a list to store all score changes
    all_changes = []

    for text_type in text_types:
        for model in models:
            original_subset = original_df[(original_df['text_type'] == text_type) & (
                original_df['model'] == model)]
            # rand_insert_period_subset = rand_insert_period_df[(
            #     rand_insert_period_df['text_type'] == text_type) & (rand_insert_period_df['model'] == model)]
            # rand_insert_the_subset = rand_insert_the_df[(
            #     rand_insert_the_df['text_type'] == text_type) & (rand_insert_the_df['model'] == model)]
            # replace_longest_subset = replace_longest_df[(
            #     replace_longest_df['text_type'] == text_type) & (replace_longest_df['model'] == model)]
            # replace_longest_rand_period_subset = replace_longest_rand_period_df[(
            #     replace_longest_rand_period_df['text_type'] == text_type) & (replace_longest_rand_period_df['model'] == model)]
            # replace_random_period_subset = replace_random_period_df[(
            #     replace_random_period_df['text_type'] == text_type) & (replace_random_period_df['model'] == model)]
            # replace_random_the_subset = replace_random_the_df[(
            #     replace_random_the_df['text_type'] == text_type) & (replace_random_the_df['model'] == model)]

            duplicate_longest_word_subset = duplicate_longest_word_df[(
                duplicate_longest_word_df['text_type'] == text_type) & (duplicate_longest_word_df['dir'] == model)]
            duplicate_period_subset = duplicate_period_df[(
                duplicate_period_df['text_type'] == text_type) & (duplicate_period_df['dir'] == model)]
            prefix_sentence_with_subset = prefix_sentence_with_df[(
                prefix_sentence_with_df['text_type'] == text_type) & (prefix_sentence_with_df['dir'] == model)]
            duplicate_whole_sentence_subset = duplicate_whole_sentence_df[(
                duplicate_whole_sentence_df['text_type'] == text_type) & (duplicate_whole_sentence_df['dir'] == model)]
            duplicate_the_subset = duplicate_the_df[(
                duplicate_the_df['text_type'] == text_type) & (duplicate_the_df['dir'] == model)]
            randomly_replace_number_subset = randomly_replace_number_df[(
                randomly_replace_number_df['text_type'] == text_type) & (randomly_replace_number_df['dir'] == model)]

            if not original_subset.empty:
                original_means = original_subset[scores].mean()

                # augmentations = {
                #     'Random Insert Period': rand_insert_period_subset,
                #     'Random Insert The': rand_insert_the_subset,
                #     'Replace Longest Word': replace_longest_subset,
                #     'Replace Longest Word with Random Period': replace_longest_rand_period_subset,
                #     'Replace Random Period': replace_random_period_subset,
                #     'Replace Random The': replace_random_the_subset
                # }

                augmentations = {
                    # 'Duplicate The': duplicate_the_subset,
                    # 'Duplicate Longest Word': duplicate_longest_word_subset,
                    # 'Duplicate Period': duplicate_period_subset,
                    # 'Prefix Sentence With': prefix_sentence_with_subset,
                    'Duplicate Whole Sentence': duplicate_whole_sentence_subset
                    # 'Randomly Replace Number': randomly_replace_number_subset
                }

                for aug_name, aug_subset in augmentations.items():
                    if not aug_subset.empty:
                        aug_means = aug_subset[scores].mean()

                        complex_values = aug_subset['complex'].unique()
                        filtered_original_df = original_subset[original_subset['complex'].isin(
                            complex_values)]
                        original_means = filtered_original_df[scores].mean()
                        score_changes = (
                            (aug_means - original_means) / original_means) * 100
                        score_changes = score_changes.round(4)

                        filtered_original_df = filtered_original_df.sort_values(
                            'complex').reset_index(drop=True)
                        aug_subset = aug_subset.sort_values(
                            'complex').reset_index(drop=True)

                        # Add row for each score
                        for score in scores:
                            original_scaled = minmax_scale(
                                original_subset[score])
                            original_scaled_mean = original_scaled.mean()
                            aug_scaled = minmax_scale(aug_subset[score])
                            aug_scaled_mean = aug_scaled.mean()
                            correlation, p_value = spearmanr(
                                filtered_original_df[score], aug_subset[score])
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
        f"data/{aug_classification}/score_changes_major_duplicate_scaled.csv", index=False)
