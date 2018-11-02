VIRTUALENV ?= .virt
PYTHON ?= $(VIRTUALENV)/bin/python
PWD = $(shell pwd)

.PHONY: clean run docker_build

all: light-bulb/ui/build/index.html docker_build dataset vendor/glove.6B

docker_build:
	docker build -t light-bulb .

light-bulb/ui/build/index.html:
	cd light-bulb/ui; yarn install
	cd light-bulb/ui; yarn build

.virt:
	virtualenv -p python3 $@
	$@/bin/pip install --upgrade pip
	$@/bin/pip install -r requirements.txt

vendor/glove.6B: vendor
	curl -o ./vendor/glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip
	unzip ./vendor/glove.6B.zip -d vendor/glove.6B/
	rm ./vendor/glove.6B.zip

vendor/keras_language_model: vendor
	curl -o vendor/keras_language_model.tar.gz https://gitlab.com/chriszhu12/light-bulb-custom-models/raw/master/keras_language_model.tar.gz
	tar -xvf vendor/keras_language_model.tar.gz -C vendor
	rm vendor/keras_language_model.tar.gz

dataset/cat_not_cat: dataset
	curl -o dataset/cat_not_cat.tar.gz https://gitlab.com/chriszhu12/light-bulb-custom-models/raw/master/cat_not_cat.tar.gz
	tar -xvf dataset/cat_not_cat.tar.gz -C dataset
	rm dataset/cat_not_cat.tar.gz

dataset/small_imdb_reviews: dataset
	curl -o dataset/small_imdb_reviews.tar.gz https://gitlab.com/chriszhu12/light-bulb-custom-models/raw/master/small_imdb_reviews.tar.gz
	tar -xvf dataset/small_imdb_reviews.tar.gz -C dataset
	rm dataset/small_imdb_reviews.tar.gz

dataset/json_classification: dataset
	curl -o dataset/json_classification.tar.gz https://gitlab.com/chriszhu12/light-bulb-custom-models/raw/master/json_classification.tar.gz
	tar -xvf dataset/json_classification.tar.gz -C dataset
	rm dataset/json_classification.tar.gz

run:
	docker run -it \
		-v ${PWD}/config:/app/config \
	 	-v ${PWD}/dataset:/app/dataset \
	 	-v ${PWD}/outputs:/app/outputs \
	 	-v ${PWD}/vendor:/app/vendor \
	 	-p 5000:5000 light-bulb /bin/bash

dataset:
	mkdir dataset

vendor:
	mkdir vendor

clean:
	rm -r vendor/*
