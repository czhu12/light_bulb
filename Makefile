VIRTUALENV ?= .virt
PYTHON ?= $(VIRTUALENV)/bin/python

.PHONY: clean

all: .virt code/ui/build/index.html vendor/keras_language_model dataset vendor

code/ui/build/index.html:
	cd code/ui; yarn install
	cd code/ui; yarn build

.virt:
	python3 -m venv $@
	$@/bin/pip install --upgrade pip
	$@/bin/pip install -r requirements.txt

run:
	$(PYTHON) code/server.py --config ${CONFIG}

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

dataset:
	mkdir dataset

vendor:
	mkdir vendor

clean:
	rm -rf vendor/keras_langauge_model
	rm -rf dataset/cat_not_cat
