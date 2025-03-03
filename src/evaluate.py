import os
import pdb

import pandas as pd
from lens import LENS, LENS_SALSA, download_model

from evaluate import load


def calculate_sari(sari, complex_sentences, modified_sentences, reference_sentences_list):
    # sari = load("sari")
    sari_scores = sari.compute(
        sources=complex_sentences,
        predictions=modified_sentences,
        references=reference_sentences_list
    )
    return sari_scores


# bert, roberta, cannine
def calculate_bertscore(bertscore, modified_sentences, reference_sentences_list, model_type="roberta-large"):
    # bertscore = load("bertscore")
    model_mapping = {
        "bert": "bert-base-uncased",
        "roberta": "roberta-large",
        "canine": "google/canine-s"
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
    # lens_path = download_model("davidheineman/lens")
    # lens = LENS(lens_path, rescale=True)
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
    data_folder = '/Users/abhinavkumar/prj/uzh/sentence_simplification/abby37kumar@outlookcom/repo/sentence_simplification/data/minor_effect'
    sari = load("sari")
    bertscore = load("bertscore")
    lens_path = download_model("davidheineman/lens")
    lens = LENS(lens_path, rescale=True)
    lens_salsa_path = download_model("davidheineman/lens-salsa")
    lens_salsa = LENS_SALSA(lens_salsa_path)
    for root, dirs, files in os.walk(data_folder):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                data_df = pd.read_csv(file_path)
                complex_sentences = data_df['complex']
                if 'original' in data_df.columns:
                    modified_sentences = data_df['original']
                else:
                    modified_sentences = data_df['modified']
                reference_sentences_list = data_df['references']
                sari_scores = calculate_sari(
                    sari, complex_sentences, modified_sentences, reference_sentences_list)
                bertscore_scores = calculate_bertscore(
                    bertscore, modified_sentences, reference_sentences_list)
                lens_scores = calculate_lens(
                    lens, complex_sentences, modified_sentences, reference_sentences_list)
                lens_salsa_scores = calculate_lens_salsa(
                    lens_salsa, complex_sentences, modified_sentences)
                data_df['sari'] = sari_scores['sari']
                # data_df.to_csv(file_path, index=False)
