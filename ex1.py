
import pandas as pd
import pickle
import time
import numpy as np
import re
import hashlib


# from simhash import Simhash
from collections import deque
from nltk import ngrams


dictionary = {}


def create_signature(doc):
    # remove unnecessary characters
    doc = re.compile('[^a-zA-Z ]').sub('', doc)
    doc = doc.lower()

    # create bin hash
    six_grams = ngrams(doc.split(), 3)
    doc_shingles = []

    for grams in six_grams:
        hash_shingle = (''.join(grams)).encode('utf-8')
        # hash_shingle = int(hashlib.md5(shingle.encode('utf-8')).hexdigest(), 16)
        doc_shingles.append(bin(int(hashlib.md5(hash_shingle.encode('utf-8')).hexdigest(), 16))[2:34])

    return create_sig_from_shingels(doc_shingles)


def create_sig_from_shingels(doc_shingles):
    sig_hash = [0] * 32  # hash size

    list_of_bin = []
    for i in range(len(doc_shingles)):
        list_of_bin.append(np.fromstring(doc_shingles[i], 'u1') - ord('0'))

    a = np.matrix(list_of_bin)

    sum_of_the_matrix = a.sum(axis=0)

    for i in range(len(sig_hash)):
        if sum_of_the_matrix[0, i] < len(doc_shingles) / 2:
            sig_hash[i] = 1
        else:
            sig_hash[i] = 0

    return sig_hash


def init():
    delta = time.time()

    train_csv = pd.read_csv("./Test.csv")
    test_pkl = open('test.pkl', 'wb')

    global dictionary

    i = 0
    for description_data in train_csv["FullDescription"]:

        # insert to dictionary
        dictionary["row_" + str(i)] = create_signature(description_data)

        # insert to pkl file
        pickle.dump(dictionary["row_" + str(i)], test_pkl)

        for data in dictionary["row_" + str(i)]:
            print(str(i) + data)

        i += 1


init()
