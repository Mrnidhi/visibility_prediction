# Enhanced Aviation Visibility Prediction System Makefile

.PHONY: help install train eval demo serve clean test

# Default target
help:
	@echo "Enhanced Aviation Visibility Prediction System"
	@echo "=============================================="
	@echo ""
	@echo "Available targets:"
	@echo "  install    - Install dependencies"
	@echo "  train      - Train enhanced models"
	@echo "  eval       - Evaluate trained models"
	@echo "  demo       - Run prediction demo"
	@echo "  serve      - Start web server"
	@echo "  test       - Run unit tests"
	@echo "  clean      - Clean artifacts and temporary files"
	@echo "  all        - Run train, eval, and demo"
	@echo ""

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed successfully!"

# Train enhanced models
train:
	@echo "Training enhanced models..."
	python -c "
import sys
sys.path.append('src')
from src.pipeline.enhanced_training_pipeline import EnhancedTrainingPipeline
pipeline = EnhancedTrainingPipeline()
results = pipeline.run_pipeline()
print('Training completed successfully!')
print(f'Best model type: {results[\"training_results\"][\"best_model_type\"]}')
"
	@echo "Training completed!"

# Evaluate models
eval:
	@echo "Evaluating models..."
	python evaluate_enhanced_models.py
	@echo "Evaluation completed!"

# Run demo
demo:
	@echo "Running prediction demo..."
	python demo_enhanced_prediction.py
	@echo "Demo completed!"

# Start web server
serve:
	@echo "Starting web server..."
	@echo "Access the application at: http://localhost:8080"
	@echo "Training endpoint: http://localhost:8080/train"
	@echo "Prediction endpoint: http://localhost:8080/predict"
	python app.py

# Run unit tests
test:
	@echo "Running unit tests..."
	python -c "
import sys
sys.path.append('src')
from src.features.physics_features import test_dewpoint_calculation, test_fog_signal
from src.components.guardrail_system import test_guardrail_system
from src.components.enhanced_data_transformation import test_enhanced_transformation
from src.components.enhanced_model_trainer import test_enhanced_model_trainer

print('Running physics feature tests...')
test_dewpoint_calculation()
test_fog_signal()

print('Running guardrail system tests...')
test_guardrail_system()

print('Running data transformation tests...')
test_enhanced_transformation()

print('Running model trainer tests...')
test_enhanced_model_trainer()

print('All unit tests passed!')
"
	@echo "Unit tests completed!"

# Clean artifacts
clean:
	@echo "Cleaning artifacts and temporary files..."
	rm -rf artifacts/
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf src/*/__pycache__/
	rm -rf *.pyc
	rm -rf src/*.pyc
	rm -rf src/*/*.pyc
	rm -f evaluation_report.txt
	rm -rf evaluation_plots/
	@echo "Cleanup completed!"

# Run all: train, eval, demo
all: train eval demo
	@echo "Complete pipeline executed successfully!"

# Docker targets
docker-build:
	@echo "Building Docker image..."
	docker build -t aerovision-ml-enhanced .

docker-run:
	@echo "Running Docker container..."
	docker run -d --name aerovision-app-enhanced -p 8080:5000 aerovision-ml-enhanced

docker-stop:
	@echo "Stopping Docker container..."
	docker stop aerovision-app-enhanced || true
	docker rm aerovision-app-enhanced || true

docker-logs:
	@echo "Showing Docker logs..."
	docker logs -f aerovision-app-enhanced

# Development targets
dev-install:
	@echo "Installing development dependencies..."
	pip install -r requirements.txt
	pip install pytest black flake8 mypy
	@echo "Development dependencies installed!"

format:
	@echo "Formatting code..."
	black src/ *.py
	@echo "Code formatted!"

lint:
	@echo "Linting code..."
	flake8 src/ *.py
	@echo "Linting completed!"

type-check:
	@echo "Type checking..."
	mypy src/ --ignore-missing-imports
	@echo "Type checking completed!"

# Quick test without full training
quick-test:
	@echo "Running quick tests..."
	python -c "
import sys
sys.path.append('src')
from src.features.physics_features import test_dewpoint_calculation, test_fog_signal
from src.components.guardrail_system import test_guardrail_system

test_dewpoint_calculation()
test_fog_signal()
test_guardrail_system()
print('Quick tests passed!')
"
	@echo "Quick tests completed!"

# Show system status
status:
	@echo "System Status:"
	@echo "=============="
	@echo "Python version:"
	@python --version
	@echo ""
	@echo "Installed packages:"
	@pip list | grep -E "(scikit-learn|pandas|numpy|flask|xgboost)"
	@echo ""
	@echo "Artifacts directory:"
	@ls -la artifacts/ 2>/dev/null || echo "No artifacts found"
	@echo ""
	@echo "Configuration files:"
	@ls -la config/
	@echo ""
	@echo "Source files:"
	@find src/ -name "*.py" | wc -l | xargs echo "Python files:"
