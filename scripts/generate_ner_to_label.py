import os
import pandas as pd

df = pd.read_csv('outputs/query_generation/labels.csv')
labels = df['label']
for index, label in enumerate(labels):
    if not os.path.exists('dataset/ner_sequence/'): os.makedirs('dataset/ner_sequence/')
    f = 'dataset/ner_sequence/query-{}.txt'.format(index)
    print('Wrote "{}" to {}'.format(label, f))
    open(f, 'w').write(label)
