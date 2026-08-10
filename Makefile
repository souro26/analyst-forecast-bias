PYTHON := python

.PHONY: all pipeline clean help
.DEFAULT_GOAL := help

STAMPS := .stamps

$(STAMPS):
	mkdir -p $(STAMPS)

$(STAMPS)/ingest: src/ingest.py configs/model_params.yml | $(STAMPS)
	$(PYTHON) src/ingest.py
	@touch $@

$(STAMPS)/clean: $(STAMPS)/ingest src/clean.py configs/model_params.yml
	$(PYTHON) src/clean.py
	@touch $@

$(STAMPS)/macro: $(STAMPS)/clean src/macro.py
	$(PYTHON) src/macro.py
	@touch $@

$(STAMPS)/regime: src/regime.py configs/model_params.yml | $(STAMPS)
	$(PYTHON) src/regime.py
	@touch $@

$(STAMPS)/features: $(STAMPS)/clean src/features.py configs/model_params.yml
	$(PYTHON) src/features.py
	@touch $@

$(STAMPS)/models: $(STAMPS)/macro $(STAMPS)/features src/models.py configs/model_params.yml
	$(PYTHON) src/models.py
	@touch $@

$(STAMPS)/signals: $(STAMPS)/macro $(STAMPS)/features src/signals.py configs/model_params.yml
	$(PYTHON) src/signals.py
	@touch $@

pipeline: $(STAMPS)/models $(STAMPS)/signals

all: pipeline

clean:
	rm -rf $(STAMPS)
	rm -f data/processed/*.parquet
	rm -f models/*.nc models/*.csv models/*.json
	rm -rf logs/

help:
	@echo ""
	@echo "  make all        Run the full pipeline (ingest → clean → macro → regime → features → models → signals)"
	@echo "  make pipeline   Same as all"
	@echo "  make clean      Remove all generated files (processed data, models, logs, stamps)"
	@echo ""
	@echo "  Individual stages:"
	@echo "    make .stamps/ingest"
	@echo "    make .stamps/clean"
	@echo "    make .stamps/macro"
	@echo "    make .stamps/regime"
	@echo "    make .stamps/features"
	@echo "    make .stamps/models"
	@echo "    make .stamps/signals"
	@echo ""
	@echo "  Requires GNU make. On Windows: use Git Bash, WSL, or 'choco install make'."
	@echo ""
