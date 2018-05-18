import os
import pandas as pd

df = pd.read_csv('outputs/ner_sequence/labels.csv')
for text, label in zip(df['text'].values, df['label'].values):
    print(zip(text.split(' '), label.split(' ')))
