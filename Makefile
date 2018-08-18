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

run:
	$(PYTHON) code/server.py --config ${CONFIG}
