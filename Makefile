.PHONY: lint dev-api

lint:
	shellcheck scripts/*.sh

dev-api:
	bash scripts/run_api_local.sh

index:
	bash scripts/run_indexer.sh
