.PHONY: train

docker-start:
	docker-compose up --build -d --remove-orphans

docker-stop:
	docker-compose down