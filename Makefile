VIRTUALENV ?= .virt
PYTHON ?= $(VIRTUALENV)/bin/python

.PHONY: start requirements.txt

all: .virt code/ui/build/index.html

code/ui/build/index.html:
	cd code/ui; yarn install
	cd code/ui; yarn build

.virt:
	python3 -m venv $@
	$@/bin/pip install --upgrade pip
	$@/bin/pip install -r requirements.txt

generate_query_inspirations: .virt
	$(PYTHON) scripts/generate_query_inspirations.py -d scripts/data.yml -n 100

query_generation: .virt
	$(PYTHON) code/server.py --config config/query_generation.yml

generate_ner: .virt
	$(PYTHON) scripts/generate_ner_to_label.py

ner_tagging: .virt
	$(PYTHON) code/server.py --config config/ner_sequence.yml

dataset:
	mkdir dataset

vendor:
	mkdir vendor

setup_seo_labelling: dataset vendor
	curl -o vendor/sequence_tagger-0.1.0-py3-none-any.whl http://sssp.d.musta.ch/ai-lab-knowledge-graph/data/sequence_tagger-0.1.0-py3-none-any.whl
	curl -o vendor/all_seo_logs.tar.gz http://sssp.d.musta.ch/ai-lab-knowledge-graph/data/all_seo_logs.tar.gz
	curl -o vendor/pretrained_ner_model.tar.gz http://sssp.d.musta.ch/ai-lab-knowledge-graph/data/pretrained_ner_model.tar.gz
	tar -xf vendor/all_seo_logs.tar.gz -C dataset/
	tar -xf vendor/pretrained_ner_model.tar.gz -C vendor
