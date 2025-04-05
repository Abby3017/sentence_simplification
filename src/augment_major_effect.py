import os
import pdb
import random
import re
import string

import pandas as pd


def duplicate_period(sentence):
    if sentence and sentence[-1] == '.':
        return sentence + '.'
    else:
        return sentence


def duplicate_the(sentence):
    words = sentence.split()
    the_positions = [i for i, word in enumerate(
        words) if word.lower() == 'the']
    if not the_positions:
        return sentence
    # If there's only one 'the', duplicate it
    if len(the_positions) == 1:
        pos = the_positions[0]
        words.insert(pos + 1, words[pos])
    else:
        # Choose a random 'the' to duplicate, but not the first one
        pos = the_positions[random.randint(1, len(the_positions) - 1)]
        words.insert(pos + 1, words[pos])

    return " ".join(words)


def duplicate_longest_word(sentence):
    # The spacecraft consists of two main elements: the NASA Cassini orbiter, named after the Italian-French Italian-French astronomer Giovanni Domenico Cassini, and the ESA Huygens probe, named after the Dutch astronomer Christiaan Huygens.
    # He was also named 1982 "Sportsman of the Year" by Sports Illustrated. Illustrated.
    # In addition, Tagore emulated numerous styles, including craftwork from northern New Ireland, Haida carvings from the west coast of Canada (British Columbia), Columbia), and woodcuts by Max Pechstein.
    # handle above cases
    words = sentence.split()
    longest_word = max(words, key=len)
    longest_word_index = words.index(longest_word)
    words.insert(longest_word_index + 1, longest_word)
    modified_sentence = " ".join(words)
    return modified_sentence


def randomly_replace_number(sentence):
    match = re.search(r'\d+', sentence)
    if not match:
        return sentence  # No number found, return the original sentence

    # Get the number and its position
    num = match.group()
    start, end = match.span()
    # Generate a new number with the same length
    new_num = num
    while new_num == num:
        new_num = ''.join(random.choice(string.digits)
                          for _ in range(len(num)))
    # Replace the number in the sentence
    modified_sentence = sentence[:start] + new_num + sentence[end:]

    return modified_sentence


def duplicate_whole_sentence(sentence):
    return sentence + ' ' + sentence


def prefix_sentence_with(sentence, prefix="Sure, Here's a simplified version:"):
    return prefix + ' ' + sentence


def augment_bless_data(folder_path, target_files):
    augment_type = 'randomly_replace_number'
    output_folder = '/Users/abhinavkumar/prj/uzh/sentence_simplification/abby37kumar@outlookcom/repo/sentence_simplification/data/major_effect'
    all_df = pd.DataFrame()
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            if dir in ['cohere-command-light', 'ground_truth', 'muss']:
                continue
            complex, modified, references, type = [], [], [], []
            for file in target_files:
                file_path = os.path.join(root, dir, file)
                df = pd.read_json(file_path, lines=True)
                # df = df[df['model_output'].str.contains(r'\d', regex=True)]
                if df.empty:
                    continue
                data_type = file.split('-')[0]
                for _, row in df.iterrows():
                    simple_sentence = row['model_output']
                    modified_sentence = randomly_replace_number(
                        simple_sentence)
                    modified.append(modified_sentence)
                    complex.append(row['source'])
                    references.append(row['references'])
                    type.append(data_type)
            result_df = pd.DataFrame({
                'complex': complex,
                'modified': modified,
                'references': references,
                'text_type': type,
                'dir': dir
            })
            all_df = pd.concat([all_df, result_df])
    print(
        f'Saving file to {output_folder}/{augment_type}_augmented.csv')
    save_file_path = f'{output_folder}/{augment_type}_augmented.csv'
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
    all_df.to_csv(save_file_path, index=False)


if __name__ == "__main__":

    folder_path = '/Users/abhinavkumar/prj/uzh/sentence_simplification/abby37kumar@outlookcom/repo/BLESS_RESULTS_INCLUDING_PREDICTIONS/model_outputs_and_evals/'
    target_files = ['asset-test_asset-valid_p0_random_fs3_nr1_s723.jsonl',
                    'med-easi-test_med-easi-validation_p0_random_fs3_nr1_s723.jsonl',
                    'news-manual-all-test_news-manual-all-val_p0_random_fs3_nr1_s723.jsonl']
    augment_bless_data(folder_path, target_files)
