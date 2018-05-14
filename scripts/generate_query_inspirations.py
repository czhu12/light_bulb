import os
import random
import plac

ENTITIES = set([
    'VERTICAL',
    'LOCATION',
    'TIME',
    'PRICE',
    'REFINEMENT',
    'ATTRIBUTE',
    'OCCUPANCY',
])

def sample():
    entities = ENTITIES.copy()
    num_filters = random.randint(1, 3)
    selected = set()
    for i in range(num_filters):
        entity = random.choice(list(entities))
        selected.add(entity)
        entities.remove(entity)
    return selected

@plac.annotations(
    n=("Number of queries to generate", 'option', 'n', int))
def main(n=100):
    for i in range(n):
        prompt = sample()
        if not os.path.exists('dataset/query_generation/'): os.makedirs('dataset/query_generation/')

        filename = 'dataset/query_generation/query-prompt-{}.txt'.format(i)
        print('Wrote {} to {}'.format(prompt, filename))
        open(filename, 'w').write(str(prompt))

if __name__ == "__main__":
    plac.call(main)
