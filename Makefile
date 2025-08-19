# Makefile for Treasury Curve GAN Project

.PHONY: help install test data train evaluate generate clean

# Default target
help: ## Show this help message
	@echo "🎯 Treasury GAN Project - Available Commands:"
	@echo "=================================================="
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make install       - Install dependencies"
	@echo "  make auto-report   - Complete pipeline: train + report"
	@echo ""
	@echo "📊 Individual Steps:"
	@echo "  make data          - Collect treasury data"
	@echo "  make train         - Train GAN model"
	@echo "  make evaluate      - Evaluate performance"
	@echo "  make generate      - Generate synthetic data"
	@echo "  make quick-report  - Generate report from existing model"
	@echo ""
	@echo "🔧 Utility Commands:"
	@echo "  make test          - Run project tests"
	@echo "  make clean         - Clean generated files"
	@echo "  make help-data     - Data collection help"
	@echo "  make help-training - Training help"
	@echo "  make help-evaluation - Evaluation help"
	@echo "  make help-generation - Generation help"
	@echo "  make help-reporting - Reporting help"
	@echo ""
	@echo "📁 Project Structure:"
	@echo "  data/              - Treasury data collection"
	@echo "  models/            - GAN model definitions"
	@echo "  training/          - Training logic"
	@echo "  evaluation/        - Performance metrics"
	@echo "  reports/           - Generated reports (auto-created)"
	@echo "  config/            - Configuration files"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed successfully!"

# Run tests
test:
	@echo "Running project tests..."
	python test_project.py

# Collect and prepare data
data:
	@echo "Collecting treasury data..."
	python data/collector.py --start-date 2022-01-01 --end-date 2024-01-01 --enhance-features
	@echo "Data collection completed!"

# Train the model
train:
	@echo "Starting GAN training..."
	python train_gan.py --start-date 2022-01-01 --end-date 2024-01-01
	@echo "Training completed!"

# Evaluate trained model
evaluate:
	@echo "Evaluating trained model..."
	python train_gan.py --skip-training --checkpoint checkpoints/best_model.pth
	@echo "Evaluation completed!"

# Generate synthetic data
generate:
	@echo "Generating synthetic data..."
	python models/generate.py --model-path checkpoints/best_model.pth --nowcasting --hedging
	@echo "Synthetic data generation completed!"

# Launch Jupyter notebook
notebook:
	@echo "Launching Jupyter notebook..."
	jupyter notebook notebooks/

# Clean generated files
clean:
	@echo "Cleaning project files..."
	rm -rf checkpoints/*
	rm -rf data/*.npy
	rm -rf data/*.parquet
	rm -rf data/*.csv
	rm -rf data/*.pkl
	rm -rf synthetic_data/
	rm -rf results/
	rm -f *.png
	@echo "Cleanup completed!"

# Quick start - run full pipeline
quickstart: install test data train evaluate generate
	@echo "🎉 Full pipeline completed successfully!"
	@echo "Check the following directories for results:"
	@echo "  - checkpoints/     : Trained models"
	@echo "  - results/         : Evaluation results"
	@echo "  - synthetic_data/  : Generated synthetic data"
	@echo "  - notebooks/       : Jupyter notebooks for exploration"

# Development setup
dev-setup: install
	@echo "Setting up development environment..."
	pip install -e .
	@echo "Development environment ready!"

# Run with custom dates
train-custom:
	@echo "Training with custom date range..."
	@read -p "Enter start date (YYYY-MM-DD): " start_date; \
	read -p "Enter end date (YYYY-MM-DD): " end_date; \
	python train_gan.py --start-date $$start_date --end-date $$end_date

# Monitor training progress
monitor:
	@echo "Monitoring training progress..."
	tail -f checkpoints/training.log 2>/dev/null || echo "No training log found. Start training first."

# Generate specific scenario types
generate-normal:
	python models/generate.py --model-path checkpoints/best_model.pth --scenario-type normal --num-scenarios 100

generate-stress:
	python models/generate.py --model-path checkpoints/best_model.pth --scenario-type stress --num-scenarios 50

generate-extreme:
	python models/generate.py --model-path checkpoints/best_model.pth --scenario-type extreme --num-scenarios 25

# Auto-report generation
auto-report: train
	@echo "🚀 Generating comprehensive report automatically..."
	@python auto_report.py

# Quick report (if model already exists)
quick-report:
	@echo "📊 Generating report from existing model..."
	@python auto_report.py

# Help with specific commands
help-data:
	@echo "📊 Data Collection Commands:"
	@echo "  make data          - Collect and simulate treasury data"
	@echo "  make clean-data    - Clean collected data"

help-training:
	@echo "🎯 Training Commands:"
	@echo "  make train         - Train the GAN model"
	@echo "  make train-gan     - Train GAN with custom parameters"

help-evaluation:
	@echo "📈 Evaluation Commands:"
	@echo "  make evaluate      - Evaluate model performance"
	@echo "  make test          - Run project tests"

help-generation:
	@echo "🚀 Generation Commands:"
	@echo "  make generate      - Generate synthetic data"
	@echo "  make generate-extreme - Generate extreme scenarios"

help-reporting:
	@echo "📊 Reporting Commands:"
	@echo "  make auto-report   - Train model and generate report"
	@echo "  make quick-report  - Generate report from existing model" 