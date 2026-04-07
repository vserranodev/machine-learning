.PHONY: install tests format 
install:
		pip install --upgrade pip &&\
				pip install -r requirements.txt

tests:
	python -m pytest -vv tests/

format:
	black *.py

lint:
	pylint --disable=R,C file.py

#all: install lint test