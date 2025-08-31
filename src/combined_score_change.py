import numpy as np
import pandas as pd
from scipy.stats import spearmanr

minor_df = pd.read_csv(
    "/Users/abhinavkumar/prj/uzh/sentence_simplification/abby37kumar@outlookcom/repo/sentence_simplification/data/minor_effect/score_changes_minor_scaled.csv")
major_df = pd.read_csv(
    "/Users/abhinavkumar/prj/uzh/sentence_simplification/abby37kumar@outlookcom/repo/sentence_simplification/data/major_effect/score_changes_major_scaled.csv")
tansprasert_df = pd.read_csv(
    "/Users/abhinavkumar/prj/uzh/sentence_simplification/abby37kumar@outlookcom/repo/sentence_simplification/data/tansprasert_aug/score_changes_tansprasert_scaled.csv")


all_df = pd.concat([minor_df, major_df, tansprasert_df], ignore_index=True)
all_df = all_df[all_df['score_type'] != 'lens_salsa']

our_df = all_df.groupby(['text_type', 'model', 'augmentation']).agg({
    'augmented_score': 'mean',
    'augmented_scaled_mean': 'mean',
    'original_score': 'mean',
    'original_scaled_mean': 'mean',
}).reset_index()

# Rename the columns
our_df.rename(columns={
    'augmented_score': 'our',
    'augmented_scaled_mean': 'our_scaled',
    'original_score': 'original',
    'original_scaled_mean': 'original_scaled'
}, inplace=True)

score_types = ['sari', 'bertscore_f1', 'lens', 'lens_salsa']
augmentations = all_df['augmentation'].unique()
text_types = all_df['text_type'].unique()

result = []

for augmentation in augmentations:
    for text_type in text_types:
        all_our_df = our_df[(our_df['augmentation'] == augmentation) & (
            our_df['text_type'] == text_type)]
        original_scores_df = all_our_df[['model', 'original']]
        scaled_scores_df = all_our_df[['model', 'original_scaled']]
        augmented_scores_df = all_our_df[['model', 'our']]
        augmented_scaled_scores_df = all_our_df[[
            'model', 'our_scaled']]
        correlation, _ = spearmanr(
            original_scores_df['original'], augmented_scores_df['our'])
        scaled_correlation, _ = spearmanr(
            scaled_scores_df['original_scaled'], augmented_scaled_scores_df['our_scaled'])
        correlation = round(correlation, 4)
        scaled_correlation = round(scaled_correlation, 4)

        original_scores_df['rank'] = original_scores_df['original'].rank(
            ascending=False, method='min')
        augmented_scores_df['rank'] = augmented_scores_df['our'].rank(
            ascending=False, method='min')
        top_3_original_ranked = original_scores_df.nsmallest(3, 'rank')
        top_3_augmented_ranked = augmented_scores_df[augmented_scores_df['model'].isin(
            top_3_original_ranked['model'])].sort_values('rank')
        original_ranks = top_3_original_ranked['rank']
        augmented_ranks_top3 = top_3_augmented_ranked['rank']
        rmse = np.sqrt(np.mean((original_ranks - augmented_ranks_top3) ** 2))
        rmse = round(rmse, 4)

        scaled_scores_df['rank'] = scaled_scores_df['original_scaled'].rank(
            ascending=False, method='min')
        augmented_scaled_scores_df['rank'] = augmented_scaled_scores_df['our_scaled'].rank(
            ascending=False, method='min')
        top_3_original_scaled_ranked = scaled_scores_df.nsmallest(3, 'rank')
        top_3_augmented_scaled_ranked = augmented_scaled_scores_df[augmented_scaled_scores_df['model'].isin(
            top_3_original_scaled_ranked['model'])].sort_values('rank')
        original_scaled_ranks = top_3_original_scaled_ranked['rank']
        augmented_scaled_ranks_top3 = top_3_augmented_scaled_ranked['rank']
        scaled_rmse = np.sqrt(
            np.mean((original_scaled_ranks - augmented_scaled_ranks_top3) ** 2))
        scaled_rmse = round(scaled_rmse, 4)

        winner_preserved = (top_3_original_ranked.iloc[0]['model'] ==
                            top_3_augmented_ranked.iloc[0]['model'])
        scaled_winner_preserved = (top_3_original_scaled_ranked.iloc[0]['model'] ==
                                   top_3_augmented_scaled_ranked.iloc[0]['model'])

        result.append({
            'text_type': text_type,
            'augmentation': augmentation,
            'correlation': correlation,
            'scaled_correlation': scaled_correlation,
            'rmse': rmse,
            'scaled_rmse': scaled_rmse,
            'winner_preserved': winner_preserved,
            'scaled_winner_preserved': scaled_winner_preserved
        })

result_df = pd.DataFrame(result)
result_df.to_csv(
    "/Users/abhinavkumar/prj/uzh/sentence_simplification/abby37kumar@outlookcom/repo/sentence_simplification/data/combined_our_score_analysis_wo_lens_salsa.csv", index=False)
