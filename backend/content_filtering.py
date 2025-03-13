import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import List, Optional, Dict
import logging

# Import SentenceTransformer if available
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
    logging.warning("SentenceTransformer is not installed. BERT-based embeddings will not be available.")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class ContentBasedRecommender:
    def __init__(self, movies_file_path: str, use_bert: bool = False, use_tfidf: bool = True) -> None:
        """
        Initialize the content-based recommender system.

        Args:
            movies_file_path (str): Path to the movies data file.
            use_bert (bool): If True, use BERT-based embeddings.
            use_tfidf (bool): If True, use TF-IDF; otherwise, use CountVectorizer.
        """
        self.movies_file_path = movies_file_path
        self.movies: pd.DataFrame = pd.DataFrame()
        self.vectorizer: Optional[object] = None
        self.count_matrix: Optional[np.ndarray] = None
        self.cosine_sim: Optional[np.ndarray] = None
        self.use_bert = use_bert and (SentenceTransformer is not None)
        self.use_tfidf = use_tfidf
        self.bert_model = None

        if self.use_bert:
            self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("BERT-based embeddings enabled using SentenceTransformer.")

        self._initialize_system()

    def _initialize_system(self) -> None:
        """Initialize the recommendation system by loading data and computing the similarity matrix."""
        self.movies = self._load_movies()
        if self.movies.empty:
            logger.error("Movies dataset is empty. Check the movies file and its contents.")
        else:
            # Create an enriched combined text column with title and genres (add additional fields if needed)
            if 'genres' in self.movies.columns:
                self.movies['combined_text'] = (
                        self.movies['title'].astype(str) + " " +
                        self.movies['genres'].astype(str)
                )
            else:
                self.movies['combined_text'] = self.movies['title'].astype(str)

            # Compute similarity using BERT embeddings if enabled; otherwise use vectorization.
            if self.use_bert:
                self._compute_bert_embeddings()
            else:
                self._compute_similarity_matrix()

    def _load_movies(self) -> pd.DataFrame:
        """
        Load and preprocess the movies dataset.

        Returns:
            pd.DataFrame: Processed movies DataFrame.
        """
        try:
            if not os.path.exists(self.movies_file_path):
                raise FileNotFoundError(f"File not found: {self.movies_file_path}")
            # Assuming u.item format: movie_id, title, release_date, genres (selected columns)
            movies = pd.read_csv(
                self.movies_file_path,
                sep="|",
                encoding="latin-1",
                header=None,
                usecols=[0, 1, 2, 4],
                names=["movie_id", "title", "release_date", "genres"],
            )
            logger.info(f"Successfully loaded {len(movies)} movies from {self.movies_file_path}")
            return movies
        except Exception as e:
            logger.exception(f"Error loading movies from {self.movies_file_path}: {e}")
            return pd.DataFrame()

    def _compute_similarity_matrix(self) -> None:
        """Compute the cosine similarity matrix using TF–IDF or CountVectorizer."""
        try:
            if "combined_text" not in self.movies.columns or self.movies["combined_text"].isnull().all():
                raise ValueError("Combined text data is missing or empty in the dataset.")
            if self.use_tfidf:
                self.vectorizer = TfidfVectorizer(stop_words="english")
                logger.info("Using TF-IDF vectorizer for text representation.")
            else:
                self.vectorizer = CountVectorizer(stop_words="english")
                logger.info("Using CountVectorizer for text representation.")
            self.count_matrix = self.vectorizer.fit_transform(self.movies["combined_text"])
            self.cosine_sim = cosine_similarity(self.count_matrix, self.count_matrix)
            logger.info("Cosine similarity matrix computed using text vectorization.")
        except Exception as e:
            logger.exception(f"Error computing similarity matrix: {e}")
            self.cosine_sim = None

    def _compute_bert_embeddings(self) -> None:
        """Compute cosine similarity matrix using BERT-based embeddings."""
        try:
            if "combined_text" not in self.movies.columns or self.movies["combined_text"].isnull().all():
                raise ValueError("Combined text data is missing or empty in the dataset.")
            # Generate embeddings using the SentenceTransformer model
            embeddings = self.bert_model.encode(self.movies["combined_text"].tolist(), show_progress_bar=True)
            self.cosine_sim = cosine_similarity(embeddings, embeddings)
            logger.info("Cosine similarity matrix computed using BERT embeddings.")
        except Exception as e:
            logger.exception(f"Error computing BERT embeddings and similarity matrix: {e}")
            self.cosine_sim = None

    def get_similar_movies(self, movie_id: int, top_n: int = 10) -> List[Dict]:
        """
        Get the top_n most similar movies to the given movie_id.

        Args:
            movie_id (int): The ID of the movie to find similar movies for.
            top_n (int, optional): The number of similar movies to return. Defaults to 10.

        Returns:
            List[Dict]: A list of dictionaries, each containing details of a similar movie.
        """
        if self.cosine_sim is None:
            logger.error("Cosine similarity matrix not computed. Cannot process similar movies request.")
            return []
        try:
            movie_indices = self.movies.index[self.movies["movie_id"] == movie_id].tolist()
            if not movie_indices:
                logger.error(f"Movie ID {movie_id} not found in the dataset.")
                return []
            movie_idx = movie_indices[0]
            sim_scores = list(enumerate(self.cosine_sim[movie_idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            # Exclude the movie itself
            sim_scores = sim_scores[1:top_n + 1]

            similar_movies_list: List[Dict] = []
            for idx, sim_score in sim_scores:
                movie = self.movies.iloc[idx]
                movie_details = self._get_movie_details(movie)
                if movie_details:
                    movie_details['similarity_score'] = sim_score
                    similar_movies_list.append(movie_details)
            return similar_movies_list
        except Exception as e:
            logger.exception(f"Error getting similar movies for movie_id {movie_id}: {e}")
            return []

    def _get_movie_details(self, movie: pd.Series) -> Optional[Dict]:
        """
        Extract relevant details from a movie Series.

        Args:
            movie (pd.Series): Pandas Series containing movie data.

        Returns:
            Optional[Dict]: Dictionary with movie details or None if an error occurs.
        """
        try:
            return {
                "movie_id": int(movie.movie_id),
                "title": movie.title,
                "release_date": movie.release_date,
                "genres": movie.genres,
            }
        except Exception as e:
            logger.exception(f"Error getting movie details: {e}")
            return None

    def get_movie_by_id(self, movie_id: int) -> Optional[Dict]:
        """
        Get details for a specific movie by ID.

        Args:
            movie_id (int): Movie ID to look up.

        Returns:
            Optional[Dict]: Movie details or None if not found.
        """
        try:
            movie_rows = self.movies[self.movies["movie_id"] == movie_id]
            if movie_rows.empty:
                logger.error(f"Movie with ID {movie_id} not found.")
                return None
            movie = movie_rows.iloc[0]
            return self._get_movie_details(movie)
        except Exception as e:
            logger.exception(f"Error retrieving movie with ID {movie_id}: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    # Set use_bert=True to use BERT-based embeddings; otherwise, it will use TF-IDF/CountVectorizer.
    recommender = ContentBasedRecommender("data/u.item", use_bert=True, use_tfidf=False)
    example_movie_id = 1
    similar_movies = recommender.get_similar_movies(example_movie_id, top_n=5)
    if similar_movies:
        for movie in similar_movies:
            print(f"Title: {movie['title']}, Similarity Score: {movie['similarity_score']:.4f}")
