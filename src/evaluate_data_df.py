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
    data_folder = '/content/sentence_simplification/data/major_effect'
    sari = load("sari")
    bertscore = load("bertscore")
    lens_path = download_model("davidheineman/lens")
    lens = LENS(lens_path, rescale=True)
    lens_salsa_path = download_model("davidheineman/lens-salsa")
    lens_salsa = LENS_SALSA(lens_salsa_path)
    processing_files = ['duplicate_longest_word_augmented.csv',
                        'duplicate_period_augmented.csv']

    # Get all files in the data folder
    all_files = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.csv'):
                all_files.append(os.path.join(root, file))

    print(f"Found {len(all_files)} CSV files in {data_folder}")

    for file_path in tqdm(all_files, desc="Reading CSV files"):
        file_name = os.path.basename(file_path)
        if file_name not in processing_files:
            continue
        print(f"Reading file: {file_name}")
        data_df = pd.read_csv(file_path)
        complex_sentences = data_df['complex']
        if 'original' in data_df.columns:
            modified_sentences = data_df['original']
        else:
            modified_sentences = data_df['modified']
        reference_sentences_list = data_df['references'].apply(
            convert_string_to_list)
        print(f"Calculating scores for {file_name}")
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
        data_df['sari'] = sari_scores
        data_df['bertscore_precision'] = bertscore_scores['precision_per_example']
        data_df['bertscore_recall'] = bertscore_scores['recall_per_example']
        data_df['bertscore_f1'] = bertscore_scores['f1_per_example']
        data_df['lens'] = lens_scores
        data_df['lens_salsa'] = lens_salsa_scores
        save_file_name = f'{file_name.split(".")[0]}_evaluated.csv'
        save_file_path = f'{data_folder}/{save_file_name}'
        print(f"Saving evaluated data to {save_file_path}")
        data_df.to_csv(save_file_path, index=False)

# https://stackoverflow.com/questions/78634235/numpy-dtype-size-changed-may-indicate-binary-incompatibility-expected-96-from
