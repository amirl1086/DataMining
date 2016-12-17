
import pandas as pd


dictionary = []


def init():
    train_csv = pd.read_csv("./Train.csv")
    global dictionary

    for description_data in train_csv['FullDescription']:
        description_row = list(set(description_data.split(" ")))
        temp = list(set(description_row) - set(dictionary))
        dictionary.append(temp)


init()
print(dictionary)
