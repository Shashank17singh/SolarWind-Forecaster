.PHONY: data features train dashboard format lint test

data:
	python src/data_ingestion.py --input-dir data/omni --output-csv data/processed/omni.csv

features:
	python src/feature_engineering.py --omni-csv data/processed/omni.csv --flare-reports-dir data/flare_reports --output-csv data/processed/dataset.csv --horizon-min 15

train:
	python src/data_sharding.py --data-csv data/processed/dataset.csv --out-dir data/processed/parquet --train-end 2017-12-31 --val-end 2021-12-31 --chunksize 100000
	python src/model_training.py --parquet-dir data/processed/parquet --model-dir models --rounds-per-shard 10 --max-eval-rows 300000

dashboard:
	streamlit run app.py

format:
	black src/ app.py
	isort src/ app.py

lint:
	flake8 src/ app.py

test:
	pytest tests/
