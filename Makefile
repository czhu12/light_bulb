VIRTUALENV ?= .virt
PYTHON ?= $(VIRTUALENV)/bin/python

.PHONY: start requirements.txt

all: .virt

.virt:
	virtualenv $@
	$@/bin/pip install --upgrade pip
	$@/bin/pip install -r requirements.txt

start: .virt
	$(PYTHON) code/server.py --config config/query_generation.yml
