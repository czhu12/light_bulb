import requests
import os

result = requests.get("https://hacker-news.firebaseio.com/v0/topstories.json?print=pretty")
try:
    os.makedirs('../dataset/hacker_news')
except:
    pass

for idx, _id in enumerate(result.json()):
    story = requests.get("https://hacker-news.firebaseio.com/v0/item/{}.json?print=pretty".format(_id))
    title = story.json()['title']
    filename = '../dataset/hacker_news/title-{}.txt'.format(idx)
    print("Writing {} to {}".format(title, filename))
    open(filename, 'w+').write(title)
