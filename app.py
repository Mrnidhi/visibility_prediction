from flask import Flask, render_template, jsonify, request, send_file
from src.exception import VisibilityException
from src.logger import logging as lg
import os,sys

from src.pipeline.training_pipeline import TraininingPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.enhanced_training_pipeline import EnhancedTrainingPipeline
from src.pipeline.enhanced_prediction_pipeline import EnhancedPredictionPipeline

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/train")
def train_route():
    try:
        # Use enhanced training pipeline
        train_pipeline = EnhancedTrainingPipeline()
        results = train_pipeline.run_pipeline()
        
        return jsonify({
            "status": "Training Successful",
            "best_model_type": results["training_results"]["best_model_type"],
            "message": "Enhanced models trained with physics features and advanced techniques"
        })

    except Exception as e:
        raise VisibilityException(e,sys)
    

@app.route("/predict", methods = ['POST', 'GET'])
def predict():
    try:
        if request.method == "POST":
            # Use enhanced prediction pipeline
            prediction_pipeline = EnhancedPredictionPipeline(request=request)
            prediction_result = prediction_pipeline.run_pipeline()
            
            # Enhanced result template with uncertainty and fog information
            return render_template("enhanced_result.html", 
                                 prediction=prediction_result["visibility_miles"],
                                 low_vis_prob=prediction_result["low_vis_prob"],
                                 fog_signal=prediction_result["fog_signal"],
                                 pi95=prediction_result["pi95"],
                                 model_type=prediction_result["model_type"],
                                 guardrail_applied=prediction_result["guardrail_applied"])
    except Exception as e:
        raise VisibilityException(e,sys)

    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug= True)