import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

from hybrid_recommend import HybridRecommender, get_recommendations
from explainability import RecommendationExplainer
from evaluation import RecommenderEvaluator
from fairness_checks import check_bias_and_fairness

app = FastAPI(title="AI-Powered Recommendation System", version="1.1")

# Enable CORS middleware (adjust allowed origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Request Models
class RecommendationRequest(BaseModel):
    user_id: int
    top_n: int = 10

class ExplainRequest(BaseModel):
    user_id: int
    movie_id: int
    detail_level: str = "simple"  # Options: "simple" or "detailed"

class FairnessRequest(BaseModel):
    recommendations: List[int]

# New model for user preferences
class UserPreferencesRequest(BaseModel):
    user_id: int
    alpha: float  # Weight for the original relevance score
    beta: float   # Weight for the normalized popularity penalty

# Global object initialization for performance improvements
@app.on_event("startup")
def startup_event():
    try:
        # Initialize global objects and attach to app.state for reuse
        app.state.recommender = HybridRecommender(use_bert=True)
        app.state.explainer = RecommendationExplainer(advanced=True)
        app.state.evaluator = RecommenderEvaluator()
        # Initialize an in-memory store for user preferences
        app.state.user_preferences = {}  # key: user_id, value: {"alpha": ..., "beta": ...}
        logger.info("Global objects (recommender, explainer, evaluator, user_preferences) initialized successfully.")
    except Exception as e:
        logger.exception("Error during startup initialization: %s", e)
        raise

# Root Endpoint
@app.get("/")
def home() -> Dict[str, Any]:
    """
    Root endpoint providing basic API information.
    """
    return {"message": "Welcome to the AI-Powered Recommendation API 🚀"}

# Endpoint: Get Hybrid Recommendations
@app.get("/recommend/{user_id}")
def get_recommendations_endpoint(user_id: int, top_n: int = 10) -> Dict[str, Any]:
    """
    Retrieve hybrid recommendations for a given user.
    """
    try:
        recommender: HybridRecommender = app.state.recommender
        recommendations = recommender.hybrid_recommendation(user_id, top_n=top_n)
        return {
            "user_id": user_id,
            "recommendations": [
                {"movie_id": int(m_id), "title": title, "score": round(score, 4)}
                for m_id, title, score in recommendations
            ]
        }
    except Exception as e:
        logger.exception("Error in get_recommendations for user %s: %s", user_id, e)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# Endpoint: Explain a Recommendation
@app.post("/explain")
def get_explanation(request: ExplainRequest) -> Dict[str, Any]:
    """
    Provide an explanation for why a particular movie was recommended.
    """
    try:
        explainer: RecommendationExplainer = app.state.explainer
        explanation = explainer.explain_recommendation(
            user_history=[],  # In production, you might fetch the user's history
            recommended_movie_id=request.movie_id,
            detail_level=request.detail_level
        )
        return {
            "user_id": request.user_id,
            "movie_id": request.movie_id,
            "explanation": explanation
        }
    except Exception as e:
        logger.exception("Error in get_explanation: %s", e)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# Endpoint: Evaluate the Model
@app.get("/evaluate")
def evaluate() -> Dict[str, Any]:
    """
    Evaluate the recommendation model and return evaluation metrics.
    """
    try:
        evaluator: RecommenderEvaluator = app.state.evaluator
        evaluation_results = evaluator.evaluate_model()
        return {"evaluation_results": evaluation_results}
    except Exception as e:
        logger.exception("Error in evaluate: %s", e)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# Endpoint: Fairness Checks
@app.post("/fairness")
def fairness_checks(request: FairnessRequest) -> Dict[str, Any]:
    """
    Perform fairness checks on the provided list of recommended movie IDs.
    """
    try:
        metrics = check_bias_and_fairness(request.recommendations)
        return {"fairness_metrics": metrics}
    except Exception as e:
        logger.exception("Error in fairness_checks: %s", e)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# New Endpoint: Get User Preferences
@app.get("/user_preferences/{user_id}")
def get_user_preferences(user_id: int) -> Dict[str, Any]:
    """
    Retrieve the fairness and personalization preferences for a given user.
    """
    try:
        preferences = app.state.user_preferences.get(user_id)
        if preferences is None:
            # Return default preferences if none are set
            preferences = {"alpha": 1.0, "beta": 0.5}
        return {"user_id": user_id, "preferences": preferences}
    except Exception as e:
        logger.exception("Error retrieving preferences for user %s: %s", user_id, e)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# New Endpoint: Update User Preferences
@app.post("/user_preferences")
def update_user_preferences(preferences_request: UserPreferencesRequest) -> Dict[str, Any]:
    """
    Update the fairness and personalization preferences for a given user.
    """
    try:
        user_id = preferences_request.user_id
        # Store the preferences in the in-memory dictionary
        app.state.user_preferences[user_id] = {"alpha": preferences_request.alpha, "beta": preferences_request.beta}
        logger.info(f"Updated preferences for user {user_id}: {app.state.user_preferences[user_id]}")
        return {"message": "User preferences updated successfully", "preferences": app.state.user_preferences[user_id]}
    except Exception as e:
        logger.exception("Error updating preferences for user %s: %s", preferences_request.user_id, e)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
