.PHONY: test run install docker-up docker-down synthetic-data

install:
	python3 -m pip install -e '.[api,validation,dev]'

test:
	python3 -m unittest discover -s tests -v
	python3 -m compileall -q neuroscad training

run:
	uvicorn neuroscad.api:app --host 0.0.0.0 --port $${PORT:-8000} --reload

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

synthetic-data:
	python3 -m training.prepare --output data/processed --samples 10000 --seed 42
