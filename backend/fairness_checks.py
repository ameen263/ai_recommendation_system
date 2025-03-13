import pandas as pd
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define file paths (adjust as needed)
RATINGS_FILE_PATH = "data/u.data"
MOVIES_FILE_PATH = "data/u.item"  # Movies file path


def load_ratings(file_path: str) -> pd.DataFrame:
    """
    Load the ratings dataset from u.data.

    :param file_path: Path to the ratings file.
    :return: DataFrame containing the ratings.
    """
    try:
        ratings = pd.read_csv(
            file_path,
            sep="\t",
            names=["userId", "movieId", "rating", "timestamp"]
        )
        logger.info(f"Loaded ratings data from {file_path} with {len(ratings)} records.")
        return ratings
    except Exception as e:
        logger.error(f"Error loading ratings from {file_path}: {e}")
        raise


def load_movies(file_path: str) -> pd.DataFrame:
    """
    Load the movies dataset from u.item.

    :param file_path: Path to the movies file.
    :return: DataFrame containing the movies.
    """
    try:
        movies = pd.read_csv(
            file_path,
            sep="|",
            encoding="latin-1",
            header=None,
            usecols=[0, 1, 2],  # Select columns: movie_id, title, genres
            names=["movie_id", "title", "genres"],
        )
        logger.info(f"Loaded movies data from {file_path} with {len(movies)} records.")
        return movies
    except Exception as e:
        logger.error(f"Error loading movies from {file_path}: {e}")
        raise


def popularity_bias_score(recommendations: list) -> float:
    """
    Calculate the popularity bias score for the given recommendations.
    The score is defined as the ratio between the average rating count of recommended movies and the overall average rating count.

    :param recommendations: List of recommended movie IDs.
    :return: Popularity bias score.
    """
    try:
        # Ensure recommendations are integers
        recommendations = [int(movie_id) for movie_id in recommendations]

        # Compute movie popularity (number of ratings per movie)
        movie_popularity = ratings.groupby("movieId").size()

        # Safely get popularity for recommended movies
        rec_popularity = movie_popularity.reindex(recommendations).dropna()
        if rec_popularity.empty:
            logger.warning("No recommended movies found in ratings data.")
            return 0.0

        recommended_popularity = rec_popularity.mean()
        overall_popularity = movie_popularity.mean()

        if overall_popularity == 0:
            logger.warning("Overall popularity is zero; cannot compute bias score.")
            return 0.0

        bias_score = recommended_popularity / overall_popularity
        logger.info(f"Calculated popularity bias score: {bias_score:.4f}")
        return bias_score
    except Exception as e:
        logger.error(f"Error calculating popularity bias: {e}")
        return 0.0


def diversity_score(recommendations: list) -> float:
    """
    Calculate the diversity score for the given recommendations.
    Diversity is measured as the average number of unique genres per recommended movie.

    :param recommendations: List of recommended movie IDs.
    :return: Diversity score.
    """
    try:
        # Ensure recommendations are integers
        recommendations = [int(movie_id) for movie_id in recommendations]
        recommended_movies = movies[movies["movie_id"].isin(recommendations)]
        if recommended_movies.empty:
            logger.warning("No recommended movies found in movies data.")
            return 0.0

        # Calculate unique genres per movie
        unique_genres_per_movie = [
            len(set(genres.split("|"))) for genres in recommended_movies["genres"] if genres
        ]
        if not unique_genres_per_movie:
            logger.warning("No genre information available for recommended movies.")
            return 0.0

        diversity = np.mean(unique_genres_per_movie)
        logger.info(f"Calculated diversity score: {diversity:.4f}")
        return diversity
    except Exception as e:
        logger.error(f"Error calculating diversity score: {e}")
        return 0.0


def exposure_fairness_score(recommendations: list) -> float:
    """
    Calculate the exposure fairness score for the given recommendations.
    Exposure fairness is measured by the coefficient of variation (std/mean) of popularity among recommended movies.

    :param recommendations: List of recommended movie IDs.
    :return: Exposure fairness score.
    """
    try:
        # Ensure recommendations are integers
        recommendations = [int(movie_id) for movie_id in recommendations]
        movie_popularity = ratings.groupby("movieId").size()
        rec_popularity = movie_popularity.reindex(recommendations).dropna()
        if rec_popularity.empty:
            logger.warning("No recommended movies found in ratings data for exposure fairness.")
            return 0.0

        mean_popularity = rec_popularity.mean()
        if mean_popularity == 0:
            logger.warning("Mean popularity of recommended movies is zero; cannot compute exposure fairness.")
            return 0.0

        exposure_fairness = rec_popularity.std() / mean_popularity
        logger.info(f"Calculated exposure fairness score: {exposure_fairness:.4f}")
        return exposure_fairness
    except Exception as e:
        logger.error(f"Error calculating exposure fairness: {e}")
        return 0.0


def check_bias_and_fairness(recommendations: list) -> dict:
    """
    Perform all fairness checks for the given recommendations.

    :param recommendations: List of recommended movie IDs.
    :return: Dictionary containing fairness metrics.
    """
    try:
        popularity_bias = popularity_bias_score(recommendations)
        diversity = diversity_score(recommendations)
        exposure_fairness = exposure_fairness_score(recommendations)

        fairness_metrics = {
            "Popularity Bias Score": round(popularity_bias, 4),
            "Diversity Score": round(diversity, 4),
            "Exposure Fairness Score": round(exposure_fairness, 4),
        }
        logger.info(f"Fairness metrics: {fairness_metrics}")
        return fairness_metrics
    except Exception as e:
        logger.error(f"Error in fairness checks: {e}")
        return {}


def re_rank_recommendations(recommendations: list, predicted_scores: dict, lambda_factor: float = 0.5) -> list:
    """
    Re-rank recommendations to mitigate fairness issues by balancing accuracy with fairness.
    The adjusted score is computed as: adjusted_score = original_score - lambda_factor * normalized_popularity

    :param recommendations: List of recommended movie IDs.
    :param predicted_scores: Dictionary mapping movie IDs to their predicted scores.
    :param lambda_factor: Trade-off parameter controlling the influence of popularity on re-ranking.
    :return: List of re-ranked movie IDs.
    """
    try:
        movie_popularity = ratings.groupby("movieId").size()
        if movie_popularity.empty:
            logger.warning("Ratings data is empty; returning original recommendations.")
            return recommendations

        max_popularity = movie_popularity.max()
        adjusted_scores = {}

        for movie in recommendations:
            pop = movie_popularity.get(movie, 0)
            norm_pop = pop / max_popularity
            original_score = predicted_scores.get(movie, 0)
            adjusted_score = original_score - lambda_factor * norm_pop
            adjusted_scores[movie] = adjusted_score
            logger.debug(
                f"Movie {movie}: original_score={original_score:.4f}, norm_pop={norm_pop:.4f}, adjusted_score={adjusted_score:.4f}")

        re_ranked = sorted(adjusted_scores, key=adjusted_scores.get, reverse=True)
        logger.info(f"Re-ranked recommendations: {re_ranked}")
        return re_ranked
    except Exception as e:
        logger.error(f"Error in re-ranking recommendations: {e}")
        return recommendations


# Load datasets once so they can be reused in the functions below.
try:
    ratings = load_ratings(RATINGS_FILE_PATH)
    movies = load_movies(MOVIES_FILE_PATH)
except Exception as e:
    logger.error("Failed to load datasets; please check file paths and data integrity.")

if __name__ == "__main__":
    # Example usage:
    # Suppose we have a list of recommended movie IDs from a recommender system.
    recommended_movie_ids = [1, 2, 3, 4, 5]

    # Predicted scores for these movies (e.g., from a CF model) for re-ranking purposes.
    predicted_scores = {1: 4.5, 2: 4.2, 3: 4.0, 4: 3.8, 5: 3.5}

    fairness_metrics = check_bias_and_fairness(recommended_movie_ids)
    print("Fairness Metrics:", fairness_metrics)

    re_ranked = re_rank_recommendations(recommended_movie_ids, predicted_scores, lambda_factor=0.5)
    print("Re-ranked Recommendations:", re_ranked)
