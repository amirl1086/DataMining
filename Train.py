
# DATA MINING, EX1
# Written by :
# Amir Lavi, id 3002020595 & Eliran Yehezkel, id 021682042
# ========================================================
# Find 3 scripts in the bottom of the page, remove # (comment) accordingly to your will
#

import pandas as pd
import pickle
import time
import numpy as np
import re
import hashlib
from nltk import ngrams


dictionary = dict()
jaccard_compatibility = []


def create_signature(doc):
    # remove unnecessary characters
    doc = re.compile('[^a-zA-Z ]').sub('', doc)
    doc = doc.lower()

    # create bin hash
    three_grams = ngrams(doc.split(), 3)
    doc_shingles = []

    for gram in three_grams:
        hash_shingle = (''.join(gram)).encode('utf-8')
        doc_shingles.append(bin(int(hashlib.md5(hash_shingle).hexdigest(), 16))[2:34])

    return create_sig_from_shingels(doc_shingles)


def create_sig_from_shingels(doc_shingles):
    sig_hash = [0] * 32  # hash size

    list_of_bin = []
    for i in range(len(doc_shingles)):
        list_of_bin.append(np.fromstring(doc_shingles[i], 'u1') - ord('0'))

    bins_matrix = np.matrix(list_of_bin)

    sum_of_the_matrix = bins_matrix.sum(axis=0, dtype='float')

    for i in range(len(sig_hash)):
        if sum_of_the_matrix[0, i] < len(doc_shingles) / 2:
            sig_hash[i] = 1
        else:
            sig_hash[i] = 0

    return sig_hash


def create_string_from_signature(signature):
    sig_string = ''
    for i in range(len(signature)):
        sig_string += str(signature[i])
    return sig_string


def find_signature_bucket(key, value):
    global dictionary
    if key in dictionary:
        dictionary[key].append(value)
    else:
        dictionary[key] = [value]
    return


def print_dict(doc_list):
    for key, value in dictionary.items():
        print(key, '-->')
        for i in range(len(value)):
            print('  -->', doc_list[value[i]])

    return


def jaccard_score(vec_1, vec_2):
    intersection = len(set.intersection(*[set(vec_1), set(vec_2)]))
    union = len(set.union(*[set(vec_1), set(vec_2)]))
    return intersection / float(union)


def to_three_grams(doc):
    three_grams = ngrams(doc.split(), 3)
    grams_list = []

    for grams in three_grams:
        grams_list.append(''.join(grams))

    return grams_list


def calc_jaccard_similarity(train_csv):
    global dictionary
    global jaccard_compatibility

    all_similarities = []
    for key, val in dictionary.items():
        curr_simi = []
        for i in range(len(val)):
            for j in range(i, len(val)):
                if i != j:
                    jaccard_curr = jaccard_score(to_three_grams(train_csv[val[i]]), to_three_grams(train_csv[val[j]]))
                    if jaccard_curr > float(0.8):
                        jaccard_compatibility.append(jaccard_curr)
                    curr_simi.append(jaccard_curr)

        if len(curr_simi) != 0:
            all_similarities.append(sum(curr_simi) / float(len(curr_simi)))

    if len(all_similarities) != 0:
        return sum(all_similarities) / float(len(all_similarities))
    else:
        return -1


def write_dictionary_pkl(file_name):
    global dictionary
    pkl_file = open(file_name, 'wb')
    pickle.dump(dictionary, pkl_file)
    pkl_file.close()


def run_train():

    train_csv = pd.read_csv("./Train.csv")['FullDescription']

    global dictionary
    global jaccard_compatibility
    jaccard_compatibility = []

    delta = time.time()
    i = 0
    for desc_row in train_csv:

        # insert to dictionary
        signature = create_signature(desc_row)

        # find signature bucket
        sig_string = create_string_from_signature(signature[:32])
        find_signature_bucket(sig_string, i)
        i += 1

    write_dictionary_pkl('train.pkl')

    final_time = (time.time() - delta) / len(train_csv)
    print('\n\n\nRun Train.csv')
    print('Final Time: ' + str(final_time))
    print('Jaccard Average Similarity: ' + str(calc_jaccard_similarity(train_csv)))
    print('Jaccard number of items with more then 0.8 score: ' + str(len(jaccard_compatibility)))


def run_test():

    test_csv = pd.read_csv("./Test.csv")['FullDescription']

    global dictionary
    global jaccard_compatibility
    jaccard_compatibility = []

    delta = time.time()
    i = 0
    for desc_row in test_csv:
        # insert to dictionary
        signature = create_signature(desc_row)

        # find signature bucket
        sig_string = create_string_from_signature(signature[:32])
        find_signature_bucket(sig_string, i)
        i += 1

    write_dictionary_pkl('test.pkl')

    final_time = (time.time() - delta) / len(test_csv)
    print('\n\n\nRun Test.csv')
    print('Final Time: ' + str(final_time))
    print('Jaccard Average Similarity: ' + str(calc_jaccard_similarity(test_csv)))
    print('Jaccard number of items with more then 0.8 score: ' + str(len(jaccard_compatibility)))


def read_dictionary_pkl(file_name):
    global dictionary
    pkl_file = open(file_name, 'rb')
    dictionary = pickle.load(pkl_file)
    pkl_file.close()


def read_test():
    global dictionary
    global jaccard_compatibility
    jaccard_compatibility = []

    read_dictionary_pkl('test.pkl')
    test_csv = pd.read_csv("./Test.csv")['FullDescription']

    print('\n\n\nRead test.pkl')
    print('Jaccard Average Similarity: ' + str(calc_jaccard_similarity(test_csv)))
    print('Jaccard number of items with more then 0.8 score: ' + str(len(jaccard_compatibility)))


def read_train():
    global dictionary
    global jaccard_compatibility
    jaccard_compatibility = []

    read_dictionary_pkl('train.pkl')
    train_csv = pd.read_csv("./Train.csv")['FullDescription']

    print('\n\n\nRead train.pkl')
    print('Jaccard Average Similarity: ' + str(calc_jaccard_similarity(train_csv)))
    print('Jaccard number of items with more then 0.8 score: ' + str(len(jaccard_compatibility)))

# run_train()
# read_train()

# run_test()
# read_test()

