import os
import pdb
import random
import string

import pandas as pd

# https://aclanthology.org/2021.gem-1.1.pdf inspired from this paper


def random_insert_period(sentence):
    words = sentence.split()
    if len(words) <= 2:  # Not enough words to insert a period
        return sentence
    insert_pos = random.randint(1, len(words) - 2)
    words[insert_pos] = words[insert_pos] + "."
    modified_sentence = " ".join(words)
    return modified_sentence


def random_insert_the(sentence):
    words = sentence.split()
    if len(words) <= 2:
        return sentence
    insert_pos = random.randint(1, len(words) - 2)
    words[insert_pos] = "the " + words[insert_pos]
    modified_sentence = " ".join(words)
    return modified_sentence


def replace_longest_word_with_the(sentence):
    words = sentence.split()
    longest_word = max(words, key=len)
    longest_word_index = words.index(longest_word)
    words[longest_word_index] = "the"
    modified_sentence = " ".join(words)
    return modified_sentence


def replace_random_word_with_period(sentence):
    words = sentence.split()
    if len(words) <= 2:
        return sentence
    replace_pos = random.randint(1, len(words) - 2)
    del words[replace_pos]
    words[replace_pos - 1] = words[replace_pos - 1] + "."
    modified_sentence = " ".join(words)
    return modified_sentence


def replace_random_word_with_the(sentence):
    words = sentence.split()
    if len(words) <= 2:
        return sentence
    valid_positions = [i for i in range(
        1, len(words) - 1) if words[i].lower() != 'the']
    if not valid_positions:
        return sentence
    replace_pos = random.choice(valid_positions)
    words[replace_pos] = 'the'
    modified_sentence = " ".join(words)
    return modified_sentence


def combine_random_insert_period_and_replace_longest_word_with_the(sentence):
    sentence = random_insert_period(sentence)
    sentence = replace_longest_word_with_the(sentence)
    return sentence


def augment_bless_data(folder_path, target_files):
    augment_type = 'replace_longest_rand_period'
    output_folder = '/Users/abhinavkumar/prj/uzh/sentence_simplification/abby37kumar@outlookcom/repo/sentence_simplification/data/tansprasert_aug/'
    complex, modified, references, type = [], [], [], []
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
                    modified_sentence = combine_random_insert_period_and_replace_longest_word_with_the(
                        simple_sentence)
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
    augment_bless_data(folder_path, target_files)
