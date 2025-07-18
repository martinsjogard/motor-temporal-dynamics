all: data models visualize

data:
	python scripts/generate_data.py

models:
	python scripts/run_ml_models.py
	python scripts/run_dl_models.py
	python scripts/run_mixed_models.py

visualize:
	jupyter nbconvert --to html notebooks/exploratory_analysis.ipynb --output-dir=figures

clean:
	rm -rf models/* logs/* figures/*
