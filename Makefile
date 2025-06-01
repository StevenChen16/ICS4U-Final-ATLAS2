.PHONY: install train dashboard test clean

install:
	pip install -e .

data:
	python -m src.data

train:
	python -m src.atlas2

dashboard:
	python -m src.dashboard

test:
	python -m tests.atlas_performance_test

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

demo: install data
	python main.py --demo