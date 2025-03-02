import os
import pdb
import random
import string

import pandas as pd


def duplicate_whitespace(text, num_spaces=2, num_words_to_affect=None):
    words = text.split()
    if num_words_to_affect is None or num_words_to_affect > len(words) - 1:
        num_words_to_affect = len(words) - 1
    result = words[0]
    for i in range(1, len(words)):
        if i <= num_words_to_affect:
            result += ' ' * num_spaces + words[i]
        else:
            result += ' ' + words[i]

    return result


def insert_random_punctuation(text, num_insertions=2):
    # https://aclanthology.org/2021.findings-emnlp.234.pdf (AEDA) augmentation
    punctuation_marks = [".", ";", "?", ":", "!", ","]
    punctuations = list(string.punctuation)
    words = text.split()
    valid_positions = []
    for i in range(1, len(words)):
        prev_word = words[i-1]
        if not any(prev_word.endswith(p) for p in punctuations):
            valid_positions.append(i)
    positions = valid_positions[:num_insertions]
    result = list(words)
    for pos in sorted(positions, reverse=True):
        punct = random.choice(punctuation_marks)
        result[pos-1] = result[pos-1] + punct
    return ' '.join(result)


def upper_case_modification(text):
    # List of common non-meaning-bearing words
    non_meaning_words = [
        'the', 'a', 'an', 'and', 'or', 'but', 'nor', 'for', 'so', 'yet',
        'to', 'of', 'in', 'on', 'at', 'by', 'with', 'from', 'as', 'into',
        'about', 'like', 'through', 'after', 'over', 'between', 'out', 'against',
        'during', 'without', 'before', 'under', 'around', 'among'
    ]
    words = text.split()
    result = [words[0]]
    for word in words[1:]:
        word_clean = word.lower()
        trailing_punct = ''
        while word_clean and word_clean[-1] in string.punctuation:
            trailing_punct = word_clean[-1] + trailing_punct
            word_clean = word_clean[:-1]

        if word_clean.lower() in non_meaning_words:
            modified_word = word_clean.title()
            result.append(modified_word + trailing_punct)
        else:
            result.append(word)

    return ' '.join(result)


def augment_bless_data(folder_path, target_files):
    augment_type = 'upper_case_modification'
    output_folder = '/Users/abhinavkumar/prj/uzh/sentence_simplification/abby37kumar@outlookcom/repo/sentence_simplification/data/minor_effect/'
    complex, modified, references, type = [], [], [], []
    zero_width_space = '\u200B'
    for file in target_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_json(file_path, lines=True)
        data_type = file.split('-')[0]
        for _, row in df.iterrows():
            simple_sentence = row['model_output']
            modified_sentence = upper_case_modification(simple_sentence)
            modified.append(modified_sentence)
            complex.append(row['source'])
            references.append(row['references'])
            type.append(data_type)

    result_df = pd.DataFrame({
        'complex': complex,
        'modified': modified,
        'references': references,
        'text_type': type
    })
    result_df.to_csv(os.path.join(
        output_folder, f'{augment_type}_augmented.csv'), index=False)


def augment_bless_folder(folder_path, target_files):
    augment_type = 'duplicate_ws_2_2'
    output_folder = '/Users/abhinavkumar/prj/uzh/sentence_simplification/abby37kumar@outlookcom/repo/sentence_simplification/data/minor_effect'
    zero_width_space = '\u200B'
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            if dir in ['cohere-command-light', 'ground_truth', 'muss']:
                continue
            complex, modified, references, type = [], [], [], []
            for file in target_files:
                file_path = os.path.join(root, dir, file)
                # print(f'Processing file {file_path}')
                df = pd.read_json(file_path, lines=True)
                data_type = file.split('-')[0]
                for _, row in df.iterrows():
                    simple_sentence = row['model_output']
                    modified_sentence = duplicate_whitespace(
                        simple_sentence, 2, 2)
                    modified.append(modified_sentence)
                    complex.append(row['source'])
                    references.append(row['references'])
                    type.append(data_type)
            result_df = pd.DataFrame({
                'complex': complex,
                'modified': modified,
                'references': references,
                'text_type': type
            })
            print(
                f'Saving file to {output_folder}/{dir}/{augment_type}_augmented.csv')
            save_file_path = f'{output_folder}/{dir}/{augment_type}_augmented.csv'
            os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
            result_df.to_csv(save_file_path, index=False)


if __name__ == "__main__":
    folder_path = '/Users/abhinavkumar/prj/uzh/sentence_simplification/abby37kumar@outlookcom/repo/BLESS_RESULTS_INCLUDING_PREDICTIONS/model_outputs_and_evals/'
    target_files = ['asset-test_asset-valid_p0_random_fs3_nr1_s723.jsonl',
                    'med-easi-test_med-easi-validation_p0_random_fs3_nr1_s723.jsonl',
                    'news-manual-all-test_news-manual-all-val_p0_random_fs3_nr1_s723.jsonl']
    augment_bless_folder(folder_path, target_files)
