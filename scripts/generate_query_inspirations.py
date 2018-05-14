import random

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

for i in range(100):
    prompt = sample()
    filename = 'dataset/query_generation/query-prompt-{}.txt'.format(i)
    print('Wrote {} to {}'.format(prompt, filename))
    open(filename, 'w').write(str(prompt))
