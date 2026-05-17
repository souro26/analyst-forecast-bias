all: reports/writeup.html

data/processed/panel.parquet: data/raw/ src/clean.py
	python src/clean.py

data/processed/features.parquet: data/processed/panel.parquet src/features.py
	python src/features.py

data/processed/regimes.parquet: data/external/ src/regime.py
	python src/regime.py

results/hierarchical_model.nc: data/processed/ src/models.py
	python src/models.py

reports/figures/: results/ src/visualize.py
	python src/visualize.py

reports/writeup.html: reports/writeup.md reports/figures/
	pandoc reports/writeup.md -o reports/writeup.html
