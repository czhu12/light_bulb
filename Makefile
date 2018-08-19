VIRTUALENV ?= .virt
PYTHON ?= $(VIRTUALENV)/bin/python

.PHONY: start requirements.txt

all: .virt code/ui/build/index.html vendor/keras_langauge_model

vendor/keras_langauge_model:
	curl -o vendor/keras_language_model.tar.gz https://gitlab.com/chriszhu12/light-bulb-custom-models/raw/master/keras_language_model.tar.gz
	tar -xvf vendor/keras_language_model.tar.gz -C vendor

code/ui/build/index.html:
	cd code/ui; yarn install
	cd code/ui; yarn build

.virt:
	python3 -m venv $@
	$@/bin/pip install --upgrade pip
	$@/bin/pip install -r requirements.txt

run:
	$(PYTHON) code/server.py --config ${CONFIG}
