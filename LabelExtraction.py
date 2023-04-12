import os
import csv

labels = [[label, label.split('_')[0]] for label in os.listdir("frames")]
print(labels[:5])

import pandas as pd

df = pd.DataFrame(labels, columns=["path", "label"])
df.to_csv('annotations_file.csv', index=False)