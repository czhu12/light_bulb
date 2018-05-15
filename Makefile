VIRTUALENV ?= .virt
PYTHON ?= $(VIRTUALENV)/bin/python

.PHONY: start requirements.txt

all: .virt

.virt:
	python3.6 -m venv $@
	$@/bin/pip install --upgrade pip
	$@/bin/pip install -r requirements.txt

generate_query_inspirations: .virt
	$(PYTHON) scripts/generate_query_inspirations.py -d scripts/data.yml -n 100

query_generation: .virt
	$(PYTHON) code/server.py --config config/query_generation.yml

generate_ner: .virt
	$(PYTHON) scripts/generate_ner_to_label.py

ner_tagging: .virt
	$(PYTHON) code/server.py --config config/query_generation.yml
