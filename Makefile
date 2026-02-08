.PHONY: lint

lint:
	shellcheck scripts/*.sh

index:
	bash scripts/run_indexer.sh
