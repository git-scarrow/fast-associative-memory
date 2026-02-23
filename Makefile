VENV   := .venv
PYTHON := $(VENV)/bin/python

.PHONY: install run clean

install:
	@bash bootstrap.sh

run: $(VENV)/bin/activate
	$(PYTHON) -m shutter_deck.service config.toml

clean:
	rm -rf $(VENV)
	find . -name '*.pt' -delete
	find . -name '*.db' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
