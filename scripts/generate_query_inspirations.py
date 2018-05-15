import os
import random
import plac
import pdb
import yaml
import numpy as np

def sample(entities, num=3):
    # First sample an entity, then sample an option from the entity uniformly
    entity_names = entities.keys()
    entity_freqs = np.array([entities[name]['frequency_ratio'] for name in entity_names])
    entity_freqs = entity_freqs / entity_freqs.sum()
    chosen_entities = np.random.choice(list(entity_names), num, p=entity_freqs, replace=False)

    suggestions = [np.random.choice(entities[entity]['options']) for entity in chosen_entities]
    prompt = dict(zip(chosen_entities, suggestions))
    return prompt

@plac.annotations(
    datafile=("Data file", 'option', 'd', str),
    n=("Number of queries to generate", 'option', 'n', int))
def main(datafile, n=100):
    with open(datafile) as f:
        entities = yaml.load(f)

    if not os.path.exists('dataset/query_generation/'): os.makedirs('dataset/query_generation/')
    for i in range(n):
        num_entities = random.randint(1, 3)
        prompt = sample(entities, num_entities)
        filename = 'dataset/query_generation/query-prompt-{}.txt'.format(i)
        print('Wrote {} to {}'.format(prompt, filename))
        open(filename, 'w').write(str(prompt))

if __name__ == "__main__":
    plac.call(main)
