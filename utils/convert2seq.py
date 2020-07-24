import pandas as pd
import pathlib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.model_selection import train_test_split
import json
import os
import argparse

nltk.download('punkt')

snippet_list = []
label_list = []

def label2onehot(label):
    """
    Turns string of label into onehot value
    param: string label 'positive', 'negative', and 'neutral'
    return: 'positive' -> 1, 'negative' -> -1, and 'neutral' -> 0
    """
    if label == 'negative':
        return -1
    elif label == 'positive':
        return 1
    else:
        return 0

def make_records(row, for_evaluation):
    global snippet_list, label_list
    """
    Converts data, Finds 'antibody', and Converts label to onehot
    params: row for df.values, for_evaluation or not (to avoid duplicate data)
    return: [
        {
            'snippet': str,
            'target': str,
            'label': int
        },
        ...
    ]
    """
    records = []
    # convert label to onehot
    labels = row[1].lower()
    label = label2onehot(labels)
    # clean snippet and find 'antibod'
    snippet = row[0].replace('\t', ' ').replace('\n', ' ')
    snippet = re.sub(r'[0-9]*&#[0-9]+', 'NUMBER', snippet)
    snippet = re.sub(r'http[s]?://\S+', 'URL', snippet)
    snippet_split = word_tokenize(snippet)
    pattern = r'[A|a]ntibod'

    for index in range(len(snippet_split)):
        if re.match(pattern, snippet_split[index]):
            word = snippet_split[index]
            snippet_split[index] = '$T$'
            new_snippet = " ".join(snippet_split)
            records.append({'snippet': new_snippet, 'target': word, 'label': label})
            snippet_split[index] = word

            if for_evaluation:
                snippet_list.append(row[0])
                label_list.append(row[1])
                break
    return records

def preprocess_for_5folds(main_file, for_evaluation=False, require_holdout=True, input_path='./', output_path='./'):
    """
    Convert data for 5 folds cross-validation
    params: main_file - file name
            for_evaluation - does the data use for bert model or absa model?
            require_holdout - require holdout data for testing or not?
            input_path - input path
            output_path - output path
    """
    df = pd.read_csv(input_path + main_file)

    all_records = []

    for row in df.values:
        records = make_records(row, for_evaluation)
        all_records.extend(records)

    if for_evaluation:
        df_save = pd.DataFrame({'SNIPPET': snippet_list, 'label': label_list})
        df_save.to_csv(output_path + 'dataset.csv', index=False)

        with open(output_path + 'dataset.txt', 'w', encoding="utf-8") as file:
            for item in all_records:
                file.write(item['snippet'] + '\n')
                file.write(item['target'] + '\n')
                file.write(str(item['label']) + '\n')

        return None

    if require_holdout:
        trainset, testset = train_test_split(all_records, test_size=0.1, random_state=2020, shuffle=True)
        with open(output_path + 'dataset-train_seq.txt', 'w', encoding="utf-8") as file:
            for item in trainset:
                file.write(item['snippet'] + '\n')
                file.write(item['target'] + '\n')
                file.write(str(item['label']) + '\n')

        with open(output_path + 'dataset-test_seq.txt', 'w', encoding="utf-8") as file:
            for item in testset:
                file.write(item['snippet'] + '\n')
                file.write(item['target'] + '\n')
                file.write(str(item['label']) + '\n')
    else:
        with open(output_path + 'dataset-train_seq.txt', 'w', encoding="utf-8") as file:
            for item in all_records:
                file.write(item['snippet'] + '\n')
                file.write(item['target'] + '\n')
                file.write(str(item['label']) + '\n')

def preprocess_train_test(fold=0, for_evaluation=False, require_holdout=True, input_path='./', output_path='./'):
    """
    Convert data for 5 folds cross-validation
    params: fold - fold number of the dataset
            for_evaluation - does the data use for bert model or absa model?
            require_holdout - require holdout data for testing or not?
            input_path - input path
            output_path - output path
    """
    df_train = pd.read_csv(input_path + 'dataset-train-{}.csv'.format(fold))
    df_test = pd.read_csv(input_path + 'dataset-test-{}.csv'.format(fold))

    train_records = []
    test_records = []
    for row in df_train.values:
        records = make_records(row, for_evaluation)
        train_records.extend(records)
    for row in df_test.values:
        records = make_records(row, for_evaluation)
        test_records.extend(records)

    with open(output_path + 'dataset-train_seq-{}.txt'.format(fold), 'w') as file:
        for item in all_records:
            file.write(item['snippet'] + '\n')
            file.write(item['target'] + '\n')
            file.write(str(item['label']) + '\n')
    with open(output_path + 'dataset-test_seq-{}.txt'.format(fold), 'w') as file:
        for item in all_records:
            file.write(item['snippet'] + '\n')
            file.write(item['target'] + '\n')
            file.write(str(item['label']) + '\n')

if __name__ == '__main__':
    if not os.path.exists('../datasets/evaluation'):
        os.mkdir('../datasets/evaluation')
    preprocess_for_5folds(main_file='dataset.csv', for_evaluation=False, require_holdout=True, input_path='../datasets/antibody_specificity/', output_path='../datasets/antibody_specificity/')
    preprocess_for_5folds(main_file='dataset.csv', for_evaluation=True, require_holdout=False, input_path='../datasets/antibody_specificity/', output_path='../datasets/evaluation/')
