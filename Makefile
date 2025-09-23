# Config
PYTHON   ?= python
PKG      ?= life_expectancy
INPUT    ?= eu_life_expectancy_raw.tsv   # filename inside $(PKG)/data/
COUNTRY  ?= PT

# Phony
.PHONY: help run runPT runDE runERROR test lint clean

help:
	@echo "Targets:"
	@echo "  make run COUNTRY=PT INPUT=eu_life_expectancy_raw.tsv  # generic run"
	@echo "  make runPT                                           # PT (default input)"
	@echo "  make runDE                                           # DE (default input)"
	@echo "  make runERROR                                        # invalid country to test error"
	@echo "  make test                                            # pytest with coverage"
	@echo "  make lint                                            # pylint on package"
	@echo "  make clean                                           # remove __pycache__ and cleaned CSVs"

# Generic run (parametric)
run:
	@echo "Running life expectancy"
	@echo "----------------------------------------"
	$(PYTHON) -m $(PKG).cleaning --country $(COUNTRY) --input $(INPUT)

# Specific countries (keep your originals)
runPT:
	@echo "Running life expectancy for PT"
	@echo "----------------------------------------"
	$(PYTHON) -m $(PKG).cleaning --country PT

run-de:
	@echo "Running life expectancy for DE"
	@echo "----------------------------------------"
	$(PYTHON) -m $(PKG).cleaning --country DE

run-error:
	@echo "Running life expectancy for invalid country XX"
	@echo "----------------------------------------"
	$(PYTHON) -m $(PKG).cleaning --country XX

# Tests & Lint
test:
	@echo "Running tests for life expectancy cleaning script"
	@echo "----------------------------------------"
	@echo "Running pytest with coverage report"
	pytest $(PKG)/ --cov --cov-report=term-missing

lint:
	@echo "Running pylint for life expectancy cleaning script"
	@echo "----------------------------------------"
	@echo "Checking code quality with pylint"
	pylint $(PKG)/

# Housekeeping
clean:
	@echo "Cleaning up life expectancy project"
	@echo "----------------------------------------"
	find $(PKG)/ -name "__pycache__" -type d -exec rm -rf {} +
	rm -f $(PKG)/data/cleaned/life_expectancy_*.csv