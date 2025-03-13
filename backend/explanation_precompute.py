import argparse
import json
import logging
import pandas as pd
from pathlib import Path

from explainability import RecommendationExplainer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_user_movie_pairs(input_path: Path):
    """
    Load user-movie pairs from a CSV file.
    Expected CSV columns: user_id, movie_id.
    """
    try:
        df = pd.read_csv(input_path)
        if 'user_id' not in df.columns or 'movie_id' not in df.columns:
            raise ValueError("Input CSV must contain 'user_id' and 'movie_id' columns.")
        logger.info(f"Loaded {len(df)} user-movie pairs from {input_path}")
        return df[['user_id', 'movie_id']].to_dict(orient='records')
    except Exception as e:
        logger.exception(f"Failed to load user-movie pairs: {e}")
        raise


def precompute_explanations(pairs, detail_level, output_path: Path):
    """
    Precompute explanations for a list of user-movie pairs.

    Args:
        pairs (list): List of dicts with 'user_id' and 'movie_id'.
        detail_level (str): "simple" or "detailed" explanation level.
        output_path (Path): Path to save the precomputed explanations JSON.
    """
    # Initialize the recommendation explainer (advanced set to True for richer explanations)
    explainer = RecommendationExplainer(advanced=True)
    explanations = {}

    for pair in pairs:
        user_id = pair['user_id']
        movie_id = pair['movie_id']
        try:
            explanation = explainer.explain_recommendation(user_id, movie_id, detail_level=detail_level)
            key = f"user_{user_id}_movie_{movie_id}"
            explanations[key] = explanation
            logger.info(f"Precomputed explanation for User {user_id}, Movie {movie_id}")
        except Exception as e:
            logger.error(f"Error precomputing explanation for User {user_id}, Movie {movie_id}: {e}")
            explanations[f"user_{user_id}_movie_{movie_id}"] = {"error": str(e)}

    # Save explanations to JSON file
    try:
        with open(output_path, "w") as outfile:
            json.dump(explanations, outfile, indent=4)
        logger.info(f"Precomputed explanations saved to {output_path}")
    except Exception as e:
        logger.exception(f"Error saving explanations to file: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Precompute explanations for a list of user-movie pairs using the RecommendationExplainer."
    )
    parser.add_argument(
        "--input", type=str, default="",
        help="Path to CSV file containing user-movie pairs (columns: user_id, movie_id). If not provided, a default list will be used."
    )
    parser.add_argument(
        "--output", type=str, default="precomputed_explanations.json",
        help="Output JSON file path to save precomputed explanations."
    )
    parser.add_argument(
        "--detail_level", type=str, default="simple", choices=["simple", "detailed"],
        help="Explanation detail level: 'simple' or 'detailed'."
    )
    args = parser.parse_args()

    # Load user-movie pairs from input file or use default sample data
    if args.input:
        input_path = Path(args.input)
        pairs = load_user_movie_pairs(input_path)
    else:
        # Default sample pairs if no input file is provided
        logger.info("No input file provided. Using default sample user-movie pairs.")
        pairs = [
            {"user_id": 1, "movie_id": 1},
            {"user_id": 1, "movie_id": 2},
            {"user_id": 2, "movie_id": 1}
        ]

    output_path = Path(args.output)
    precompute_explanations(pairs, args.detail_level, output_path)


if __name__ == "__main__":
    main()
