import gc
import os
import pdb

import pandas as pd
import torch
from evaluate import load
from lens import LENS, LENS_SALSA, download_model
from tqdm import tqdm


def reset_cuda():
    """Reset CUDA and clear memory caches"""
    torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()


def convert_string_to_list(string_list):
    """Convert string representation of list to actual list."""
    if isinstance(string_list, str):
        try:
            return eval(string_list)
        except:
            return [string_list]
    return string_list


def calculate_sari(sari, complex_sentences, modified_sentences, reference_sentences_list):
    complex_sentences = complex_sentences.tolist() if hasattr(
        complex_sentences, 'tolist') else complex_sentences
    modified_sentences = modified_sentences.tolist() if hasattr(
        modified_sentences, 'tolist') else modified_sentences

    individual_sari_scores = []

    for i in tqdm(range(len(complex_sentences)), desc="Calculating individual SARI scores"):
        complex_sent = [complex_sentences[i]]
        modified_sent = [modified_sentences[i]]
        references = [reference_sentences_list[i]]

        item_score = sari.compute(
            sources=complex_sent,
            predictions=modified_sent,
            references=references
        )

        individual_sari_scores.append(item_score['sari'])

    return individual_sari_scores


# bert, roberta
def calculate_bertscore(bertscore, modified_sentences, reference_sentences_list, model_type="roberta-large"):
    model_mapping = {
        "bert": "bert-base-uncased",
        "roberta": "roberta-large",
    }

    model_name = model_mapping.get(model_type, "roberta-large")

    # Process entire batch at once
    scores = bertscore.compute(
        predictions=modified_sentences,
        references=reference_sentences_list,
        model_type=model_name,
        lang="en",
        batch_size=64
    )

    # Calculate average scores
    avg_precision = sum(scores['precision']) / len(scores['precision'])
    avg_recall = sum(scores['recall']) / len(scores['recall'])
    avg_f1 = sum(scores['f1']) / len(scores['f1'])

    return {
        'bertscore_precision': avg_precision,
        'bertscore_recall': avg_recall,
        'bertscore_f1': avg_f1,
        'bertscore_model': model_name,
        # Include per-example scores for detailed analysis
        'precision_per_example': scores['precision'],
        'recall_per_example': scores['recall'],
        'f1_per_example': scores['f1']
    }


def calculate_lens(lens, complex_sentences, modified_sentences, reference_sentences_list):
    scores = lens.score(complex_sentences, modified_sentences,
                        reference_sentences_list, batch_size=64, devices=[0])
    return scores


def calculate_lens_salsa(lens_salsa, complex_sentences, modified_sentences):
    # lens_salsa_path = download_model("davidheineman/lens-salsa")
    # lens_salsa = LENS_SALSA(lens_salsa_path)
    scores, _ = lens_salsa.score(
        complex_sentences, modified_sentences, batch_size=64, devices=[0])
    return scores


if __name__ == '__main__':
    data_folder = '/content/sentence_simplification/data/minor_effect'
    sari = load("sari")
    bertscore = load("bertscore")
    lens_path = download_model("davidheineman/lens")
    lens = LENS(lens_path, rescale=True)
    lens_salsa_path = download_model("davidheineman/lens-salsa")
    lens_salsa = LENS_SALSA(lens_salsa_path)
    processing_file_names = [
        'duplicate_ws_2_1_augmented.csv', 'duplicate_ws_2_2_augmented.csv']

    all_dirs = []
    all_df_list = []
    for i in processing_file_names:
        all_df_list.append(pd.DataFrame())

    for root, dirs, files in os.walk(data_folder):
        all_dirs.extend([os.path.join(root, d) for d in dirs])

    for dir_path in tqdm(all_dirs, desc="Processing directories"):
        dir_name = os.path.basename(dir_path)
        print(f"\nProcessing directory: {dir_name}")

        # Get all files in the directory
        files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]

        # Progress bar for files
        for file_name in tqdm(files, desc=f"Files in {dir_name}"):
            if file_name not in processing_file_names:
                continue
            file_index = processing_file_names.index(file_name)
            current_df = all_df_list[file_index]
            file_path = os.path.join(dir_path, file_name)
            print(f"\n  Processing file: {file_name}")
            data_df = pd.read_csv(file_path)
            data_df['model'] = dir_name
            data_df['references'] = data_df['references'].apply(
                convert_string_to_list)
            all_df_list[file_index] = pd.concat(
                [current_df, data_df]).reset_index(drop=True)

    i = 0
    for all_df in all_df_list:
        complex_sentences = all_df['complex']
        if 'original' in all_df.columns:
            modified_sentences = all_df['original']
        else:
            modified_sentences = all_df['modified']
        reference_sentences_list = all_df['references']
        print(f"Calculating scores for {processing_file_names[i]}")
        sari_scores = calculate_sari(
            sari, complex_sentences, modified_sentences, reference_sentences_list)
        print("SARI scores calculated")
        bertscore_scores = calculate_bertscore(
            bertscore, modified_sentences, reference_sentences_list)
        print("BERTScore scores calculated")
        reset_cuda()
        lens_scores = calculate_lens(
            lens, complex_sentences, modified_sentences, reference_sentences_list)
        print("LENS scores calculated")
        reset_cuda()
        lens_salsa_scores = calculate_lens_salsa(
            lens_salsa, complex_sentences, modified_sentences)
        print("LENS-SALSA scores calculated")
        reset_cuda()
        all_df['sari'] = sari_scores
        all_df['bertscore_precision'] = bertscore_scores['precision_per_example']
        all_df['bertscore_recall'] = bertscore_scores['recall_per_example']
        all_df['bertscore_f1'] = bertscore_scores['f1_per_example']
        all_df['lens'] = lens_scores
        all_df['lens_salsa'] = lens_salsa_scores
        current_file_name = processing_file_names[i]
        save_file_name = f'{current_file_name.split(".")[0]}_evaluated.csv'
        save_file_path = f'{data_folder}/{save_file_name}'
        print(f"Saving evaluated data to {save_file_path}")
        all_df.to_csv(save_file_path, index=False)
        i += 1
