PYTHON ?= python

.PHONY: check validate test compile

check: compile validate test

compile:
	$(PYTHON) -m compileall -q vw_rag scripts tests

validate:
	$(PYTHON) scripts/validate_bundle.py

test:
	$(PYTHON) -m pytest -q
