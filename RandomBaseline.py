import os
import pandas as pd
from random import randrange


#label_dict = {label.split('_')[0]: idx for idx, label in enumerate(os.listdir("videos"))}
label_dict = {'one': 0, 'one_left': 1,
              'two': 2, 'two_left': 3,
              'three': 4, 'three_left': 5,
              'four': 6, 'four_left': 7,
              'five': 8, 'five_left': 9,
              'rock': 10, 'rock_left':  11}


print(label_dict.keys())
print(len(label_dict.keys()))

labels = pd.read_csv("annotations_file.csv")
#targets = [label_dict[label] for label in labels.iloc[:, 1].values if not "love" in label]
targets = []
for label in labels.iloc[:, 1].values:
    if "love" in label:
        continue
    else:
        targets.append(label_dict[label])

hit = 0
for target in targets:
    if target == randrange(12):
        hit += 1

print((hit/len(targets))*100, "%")