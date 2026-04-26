"""
Flask REST API for the Ayurvedic Formulation Recommender.
Exposes a single endpoint: POST /recommend
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from ayurveda_recommender import load_and_preprocess, AyurvedaRecommender

app = Flask(__name__)
CORS(app)   # Allow cross-origin requests from the React UI

CSV_PATH = os.environ.get("CSV_PATH", "herbal_formulation.csv")

# Warm up the model at startup
print("Warming up the recommender …")
_df  = load_and_preprocess(CSV_PATH)
_rec = AyurvedaRecommender(_df)
print("Recommender ready ✓")


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    POST /recommend
    Body: { "query": "bloating nausea stomach pain", "top_n": 5 }
    Returns: { "results": [ { ...formulation fields... } ] }
    """
    body  = request.get_json(silent=True) or {}
    query = body.get("query", "").strip()
    top_n = int(body.get("top_n", 5))

    if not query:
        return jsonify({"error": "query field is required"}), 400

    results = _rec.recommend_formulations(query, top_n=top_n)
    return jsonify({"results": results})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "total_formulations": len(_df)})

@app.route("/dashboard", methods=["GET"])
def dashboard():
    return jsonify({
        "modules": [
            {
                "name": "Ayurveda Recommendation",
                "description": "Get formulations based on symptoms",
                "endpoint": "/recommend",
                "method": "POST"
            },
            {
                "name": "Dosha Detector",
                "description": "Find your body type (Vata, Pitta, Kapha)",
                "endpoint": "/dosha",
                "method": "POST"
            },
            {
                "name": "Diet Planner",
                "description": "Personalized diet based on dosha",
                "endpoint": "/diet",
                "method": "POST"
            }
        ]
    })
    
@app.route("/dosha", methods=["POST"])
def dosha():
    return jsonify({
        "dominant_dosha": "Vata",
        "message": "Dosha detection coming soon"
    })
        
@app.route("/diet", methods=["POST"])
def diet():
    return jsonify({
        "eat": ["warm food", "ghee", "rice"],
        "avoid": ["cold food", "dry snacks"]
    })


if __name__ == "__main__":
    app.run(port=5050, debug=False)
