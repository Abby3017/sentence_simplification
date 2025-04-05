import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

np.random.seed(42)

if __name__ == "__main__":
    original_df = pd.read_csv("data/original_data_evaluated.csv")

    aug_classification = 'tansprasert_aug'
    # take care of df having different shape
    rand_insert_period_df = pd.read_csv(
        f"data/{aug_classification}/random_insert_period_augmented_evaluated.csv")
    rand_insert_the_df = pd.read_csv(
        f"data/{aug_classification}/random_insert_the_augmented_evaluated.csv")
    replace_longest_df = pd.read_csv(
        f"data/{aug_classification}/replace_longest_augmented_evaluated.csv")
    replace_longest_rand_period_df = pd.read_csv(
        f"data/{aug_classification}/replace_longest_rand_period_augmented_evaluated.csv")
    replace_random_period_df = pd.read_csv(
        f"data/{aug_classification}/replace_random_period_augmented_evaluated.csv")
    replace_random_the_df = pd.read_csv(
        f"data/{aug_classification}/replace_random_the_augmented_evaluated.csv")

    models = original_df['model'].unique()
    score_types = ['sari', 'bertscore_precision',
                   'bertscore_recall', 'bertscore_f1', 'lens', 'lens_salsa']
    text_types = original_df['text_type'].unique()
    """
    We applied each of the modification techniques to
    a varied percentage of output sentences, from 10%
    to 100% in increments of 10%, for the five text simplification systems. The sentences to be modified
    were randomly selected from the system output.
    """
    result_dict = {}
    for text_type in text_types:
        result_dict[text_type] = {}
        for model in tqdm(models, desc="Processing models"):
            original_model_df = original_df[(original_df['model'] == model) & (
                original_df['text_type'] == text_type)]
            # Add '_original' suffix to all columns in original_model_df except 'complex'
            exclude_columns = ['complex']
            original_model_df = original_model_df.rename(
                columns={
                    col: f"{col}_original" for col in original_model_df.columns if col not in exclude_columns}
            )

            # Filter dataframes for current model
            rand_insert_period_model_df = rand_insert_period_df[(
                rand_insert_period_df['model'] == model) & (original_df['text_type'] == text_type)]
            rand_insert_the_model_df = rand_insert_the_df[(
                rand_insert_the_df['model'] == model) & (original_df['text_type'] == text_type)]
            replace_longest_model_df = replace_longest_df[(
                replace_longest_df['model'] == model) & (original_df['text_type'] == text_type)]
            replace_longest_rand_period_model_df = replace_longest_rand_period_df[(
                replace_longest_rand_period_df['model'] == model) & (original_df['text_type'] == text_type)]
            replace_random_period_model_df = replace_random_period_df[(
                replace_random_period_df['model'] == model) & (original_df['text_type'] == text_type)]
            replace_random_the_model_df = replace_random_the_df[(
                replace_random_the_df['model'] == model) & (original_df['text_type'] == text_type)]

            # Select relevant columns from each dataframe
            columns_to_select = ['complex', 'sari', 'bertscore_precision', 'bertscore_recall',
                                 'bertscore_f1', 'lens', 'lens_salsa']

            rand_insert_period_model_df = rand_insert_period_model_df[columns_to_select]
            rand_insert_the_model_df = rand_insert_the_model_df[columns_to_select]
            replace_longest_model_df = replace_longest_model_df[columns_to_select]
            replace_longest_rand_period_model_df = replace_longest_rand_period_model_df[
                columns_to_select]
            replace_random_period_model_df = replace_random_period_model_df[columns_to_select]
            replace_random_the_model_df = replace_random_the_model_df[columns_to_select]

            rand_insert_period_model_df = rand_insert_period_model_df.rename(
                columns={
                    col: f"{col}_rip_aug" for col in rand_insert_period_model_df.columns if col not in exclude_columns}
            )
            rand_insert_the_model_df = rand_insert_the_model_df.rename(
                columns={
                    col: f"{col}_rit_aug" for col in rand_insert_the_model_df.columns if col not in exclude_columns}
            )
            replace_longest_model_df = replace_longest_model_df.rename(
                columns={
                    col: f"{col}_rl_aug" for col in replace_longest_model_df.columns if col not in exclude_columns}
            )
            replace_longest_rand_period_model_df = replace_longest_rand_period_model_df.rename(
                columns={
                    col: f"{col}_rlr_aug" for col in replace_longest_rand_period_model_df.columns if col not in exclude_columns}
            )
            replace_random_period_model_df = replace_random_period_model_df.rename(
                columns={
                    col: f"{col}_rrp_aug" for col in replace_random_period_model_df.columns if col not in exclude_columns}
            )
            replace_random_the_model_df = replace_random_the_model_df.rename(
                columns={
                    col: f"{col}_rrt_aug" for col in replace_random_the_model_df.columns if col not in exclude_columns}
            )

            merged_df = pd.merge(
                original_model_df,
                rand_insert_period_model_df,
                on='complex'
            )
            merged_df = pd.merge(
                merged_df,
                rand_insert_the_model_df,
                on='complex'
            )
            merged_df = pd.merge(
                merged_df,
                replace_longest_model_df,
                on='complex'
            )
            merged_df = pd.merge(
                merged_df,
                replace_longest_rand_period_model_df,
                on='complex'
            )
            merged_df = pd.merge(
                merged_df,
                replace_random_period_model_df,
                on='complex'
            )
            merged_df = pd.merge(
                merged_df,
                replace_random_the_model_df,
                on='complex'
            )
            # pdb.set_trace()
            result_dict[text_type][model] = {}
            for score_type in score_types:
                result_dict[text_type][model][f"{score_type}_rip_aug"] = []
                result_dict[text_type][model][f"{score_type}_rit_aug"] = []
                result_dict[text_type][model][f"{score_type}_rl_aug"] = []
                result_dict[text_type][model][f"{score_type}_rlr_aug"] = []
                result_dict[text_type][model][f"{score_type}_rrp_aug"] = []
                result_dict[text_type][model][f"{score_type}_rrt_aug"] = []
            for val in range(10, 110, 10):
                aug_percentage = val
                original_percentage = 100 - val
                total_rows = len(merged_df)
                aug_rows = int(total_rows * aug_percentage / 100)
                original_rows = total_rows - aug_rows
                # Get indices for sampling
                all_indices = np.arange(total_rows)
                aug_indices = np.random.choice(
                    all_indices, size=aug_rows, replace=False)
                original_indices = np.array(
                    list(set(all_indices) - set(aug_indices)))
                # Create samples
                aug_sample = merged_df.iloc[aug_indices]
                original_sample = merged_df.iloc[original_indices]
                for score_type in tqdm(score_types, desc=f"Processing scores for {model} at {aug_percentage}%"):
                    score_type_original = f"{score_type}_original"
                    score_type_rip_aug = f"{score_type}_rip_aug"
                    score_type_rit_aug = f"{score_type}_rit_aug"
                    score_type_rl_aug = f"{score_type}_rl_aug"
                    score_type_rlr_aug = f"{score_type}_rlr_aug"
                    score_type_rrp_aug = f"{score_type}_rrp_aug"
                    score_type_rrt_aug = f"{score_type}_rrt_aug"

                    original_mean = original_sample[score_type_original].mean()
                    rip_augmented_mean = aug_sample[score_type_rip_aug].mean()
                    rit_augmented_mean = aug_sample[score_type_rit_aug].mean()
                    rl_augmented_mean = aug_sample[score_type_rl_aug].mean()
                    rlr_augmented_mean = aug_sample[score_type_rlr_aug].mean()
                    rrp_augmented_mean = aug_sample[score_type_rrp_aug].mean()
                    rrt_augmented_mean = aug_sample[score_type_rrt_aug].mean()
                    if original_mean is np.nan:
                        original_mean = 0
                    rip_combined_mean = (original_rows * original_mean +
                                         aug_rows * rip_augmented_mean) / total_rows
                    rit_combined_mean = (original_rows * original_mean +
                                         aug_rows * rit_augmented_mean) / total_rows
                    rl_combined_mean = (original_rows * original_mean +
                                        aug_rows * rl_augmented_mean) / total_rows
                    rlr_combined_mean = (original_rows * original_mean +
                                         aug_rows * rlr_augmented_mean) / total_rows
                    rrp_combined_mean = (original_rows * original_mean +
                                         aug_rows * rrp_augmented_mean) / total_rows
                    rrt_combined_mean = (original_rows * original_mean +
                                         aug_rows * rrt_augmented_mean) / total_rows
                    result_dict[text_type][model][score_type_rip_aug].append(
                        rip_combined_mean)
                    result_dict[text_type][model][score_type_rit_aug].append(
                        rit_combined_mean)
                    result_dict[text_type][model][score_type_rl_aug].append(
                        rl_combined_mean)
                    result_dict[text_type][model][score_type_rlr_aug].append(
                        rlr_combined_mean)
                    result_dict[text_type][model][score_type_rrp_aug].append(
                        rrp_combined_mean)
                    result_dict[text_type][model][score_type_rrt_aug].append(
                        rrt_combined_mean)

    # Create a directory for plots if it doesn't exist
    import os
    os.makedirs("plots", exist_ok=True)
    os.makedirs("plots/tansprasert", exist_ok=True)

    # Group plots by text_type and score_type
    for text_type in result_dict.keys():
        for score_type in score_types:
            # Count number of models for this text_type
            models_for_type = [m for m in result_dict[text_type].keys()]
            num_models = len(models_for_type)

            if num_models == 0:
                continue

            # Calculate grid dimensions (aim for roughly square layout)
            import math
            grid_size = math.ceil(math.sqrt(num_models))
            rows = math.ceil(num_models / grid_size)
            cols = min(grid_size, num_models)

            # Create figure with subplots
            fig, axes = plt.subplots(
                rows, cols, figsize=(5*cols, 4*rows), sharey=True)
            fig.suptitle(
                f"{text_type.upper()} - {score_type.upper()}", fontsize=16)

            # Make axes iterable even if there's only one subplot
            if num_models == 1:
                axes = np.array([axes])

            # Flatten axes array for easy iteration
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

            # Plot each model in its own subplot
            for i, model in enumerate(models_for_type):
                ax = axes[i]
                rip_aug = f"{score_type}_rip_aug"
                rit_aug = f"{score_type}_rit_aug"
                rl_aug = f"{score_type}_rl_aug"
                rlr_aug = f"{score_type}_rlr_aug"
                rrp_aug = f"{score_type}_rrp_aug"
                rrt_aug = f"{score_type}_rrt_aug"

                # if rip_aug in result_dict[text_type][model] and rit_aug in result_dict[text_type][model]:
                x_values = list(range(10, 110, 10))

                # Plot RIP augmentation
                ax.plot(x_values, result_dict[text_type][model][rip_aug],
                        linestyle='-', label='Random Insert Period')
                # Plot RIT augmentation
                ax.plot(x_values, result_dict[text_type][model][rit_aug],
                        linestyle='--', label='Random Insert The')
                # Plot RL augmentation
                ax.plot(x_values, result_dict[text_type][model][rl_aug],
                        linestyle=':', label='Replace Longest')
                # Plot RLR augmentation
                ax.plot(x_values, result_dict[text_type][model][rlr_aug],
                        linestyle='-.', label='Replace Longest Random Period')
                # Plot RRP augmentation
                ax.plot(x_values, result_dict[text_type][model][rrp_aug],
                        linestyle='-', label='Replace Random Period')
                # Plot RRT augmentation
                ax.plot(x_values, result_dict[text_type][model][rrt_aug],
                        linestyle='--', label='Replace Random The')

                ax.set_title(f"Model: {model}")
                ax.set_xlabel('Percentage of Augmented Sentences')
                ax.set_ylabel(f'{score_type.upper()} Score')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()

            # Hide any unused subplots
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)

            # Adjust layout
            plt.tight_layout()
            fig.subplots_adjust(top=0.9)  # Add space for the suptitle

            # Save the figure with all subplots
            plt.savefig(
                f"plots/{text_type}_{score_type}_all_models.pdf", dpi=300)
            plt.close(fig)

            print(f"Created plot for {text_type} - {score_type}")
