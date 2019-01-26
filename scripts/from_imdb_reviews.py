import plac
import os

import pandas as pd

def read_texts(directory):
    paths = os.listdir(directory)
    for path in paths:
        with open(os.path.join(directory, path)) as f:
            text = f.read()
            yield text


@plac.annotations(
    acl_dir=("aclImdb directory", "option", "d", str),
    num_samples=("number of samples to read from pos / neg directories", "option", "n", str),
    epochs=("Num epochs.", "option", "e", int))
def main(acl_dir, num_samples=100, epochs=3):
    positive_texts = []
    for i, text in enumerate(read_texts(os.path.join(acl_dir, 'train/pos'))):
        if i >= num_samples: break
        positive_texts.append(text)

    negative_texts = []
    for i, text in enumerate(read_texts(os.path.join(acl_dir, 'train/neg'))):
        if i >= num_samples: break
        negative_texts.append(text)

    data = list(zip(negative_texts, [0] * len(negative_texts)))
    data += list(zip(positive_texts, [1] * len(positive_texts)))
    # produce a csv with 2 columns: text and label
    df = pd.DataFrame(data, columns=['text', 'label'])
    df.to_csv('dataset/imdb_pretrain.csv', index=False)
    print("Wrote {} lines to dataset/imdb_pretrain.csv".format(len(df)))

if __name__ == "__main__":
    plac.call(main)
