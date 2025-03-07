import os
import pdb

import pandas as pd
from evaluate import load
from lens import LENS, LENS_SALSA, download_model
from tqdm import tqdm


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

    for i in range(len(complex_sentences)):
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
        lang="en"
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
                        reference_sentences_list, batch_size=16, devices=[0])
    return scores


def calculate_lens_salsa(lens_salsa, complex_sentences, modified_sentences):
    # lens_salsa_path = download_model("davidheineman/lens-salsa")
    # lens_salsa = LENS_SALSA(lens_salsa_path)
    scores, _ = lens_salsa.score(
        complex_sentences, modified_sentences, batch_size=16, devices=[0])
    return scores


if __name__ == '__main__':
    data_folder = '/content/sentence_simplification/data/minor_effect'
    sari = load("sari")
    bertscore = load("bertscore")
    lens_path = download_model("davidheineman/lens")
    lens = LENS(lens_path, rescale=True)
    lens_salsa_path = download_model("davidheineman/lens-salsa")
    lens_salsa = LENS_SALSA(lens_salsa_path)

    all_dirs = []
    for root, dirs, files in os.walk(data_folder):
        all_dirs.extend([os.path.join(root, d) for d in dirs])

    for dir_path in tqdm(all_dirs, desc="Processing directories"):
        dir_name = os.path.basename(dir_path)
        print(f"\nProcessing directory: {dir_name}")

        # Get all files in the directory
        files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]

        # Progress bar for files
        for file_name in tqdm(files, desc=f"Files in {dir_name}"):
            file_path = os.path.join(dir_path, file_name)
            print(f"\n  Processing file: {file_name}")
            data_df = pd.read_csv(file_path)
            complex_sentences = data_df['complex']
            if 'original' in data_df.columns:
                modified_sentences = data_df['original']
            else:
                modified_sentences = data_df['modified']
            reference_sentences_list = data_df['references'].apply(
                convert_string_to_list)
            sari_scores = calculate_sari(
                sari, complex_sentences, modified_sentences, reference_sentences_list)
            bertscore_scores = calculate_bertscore(
                bertscore, modified_sentences, reference_sentences_list)
            lens_scores = calculate_lens(
                lens, complex_sentences, modified_sentences, reference_sentences_list)
            lens_salsa_scores = calculate_lens_salsa(
                lens_salsa, complex_sentences, modified_sentences)
            data_df['sari'] = sari_scores
            # data_df.to_csv(file_path, index=False)
