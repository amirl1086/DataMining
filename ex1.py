
import pandas as pd
import pickle


dictionary = {}


def create_signature(word_list):
    word_dic = {}
    for word in word_list:
        if word not in word_dic:
            word_dic[word] = 0
        else:
            word_dic[word] += 1
    return word_dic


def init():
    train_csv = pd.read_csv("./Test.csv")
    test_pkl = open('test.pkl', 'wb')

    global dictionary

    i = 0
    for description_data in train_csv["FullDescription"]:

        # create the list of words
        description_row = list(set(description_data.split(" ")))

        # insert to dictionary
        dictionary["row_" + str(i)] = create_signature(description_row)

        # insert to pkl file
        pickle.dump(dictionary["row_" + str(i)], test_pkl)

        for data in dictionary["row_" + str(i)]:
            print(str(i) + data)

        i += 1


init()
