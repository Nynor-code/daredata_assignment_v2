# run life expectancy
run-pt:
	@echo "Running life expectancy"
	@echo "----------------------------------------"
	python -m life_expectancy.cleaning --country PT

run-de:
	@echo "Running life expectancy"
	@echo "----------------------------------------"
	python -m life_expectancy.cleaning --country DE

run-error:
	@echo "Running life expectancy"
	@echo "----------------------------------------"
	python -m life_expectancy.cleaning --country XX

# test life expectancy
test:
	@echo "Running tests for life expectancy cleaning script"
	@echo "----------------------------------------"
	@echo "Running pytest with coverage report"
	pytest life_expectancy/ --cov --cov-report=term-missing

lint:
	@echo "Running pylint for life expectancy cleaning script"
	@echo "----------------------------------------"
	@echo "Checking code quality with pylint"
	pylint life_expectancy/

# Housekeeping
clean:
	@echo "Cleaning up life expectancy project"
	@echo "----------------------------------------"
	find life_expectancy/ -name "__pycache__" -type d -exec rm -rf {} +
	rm -f life_expectancy/data/*.csv
