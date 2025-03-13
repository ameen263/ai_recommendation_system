import pandas as pd
import networkx as nx
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RecommendationExplainer:
    def __init__(self, movies_file_path: str = "data/u.item"):
        """
        Initialize the recommendation explainer that combines logical and graph-based explanations.

        Args:
            movies_file_path (str): Path to the movies dataset (u.item).
        """
        self.movies_file_path = movies_file_path
        self.movies_df = self._load_movies()
        self.movie_graph = self._build_movie_graph()

    def _load_movies(self) -> pd.DataFrame:
        """
        Load the movies' dataset.
        Assumes the file is pipe-separated with columns: movie_id, title, release_date, imdb_url, genres.

        Returns:
            pd.DataFrame: DataFrame containing movie information.
        """
        try:
            df = pd.read_csv(
                self.movies_file_path,
                sep="|",
                encoding="latin-1",
                header=None,
                names=["movie_id", "title", "release_date", "imdb_url", "genres"]
            )
            logger.info(f"Loaded {len(df)} movies from {self.movies_file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading movies: {e}")
            return pd.DataFrame()

    def _build_movie_graph(self) -> nx.Graph:
        """
        Build a graph connecting movies and genres.
        Each movie is added as a node with its title and genres, and connected to genre nodes.

        Returns:
            nx.Graph: The constructed graph.
        """
        G = nx.Graph()
        try:
            for _, row in self.movies_df.iterrows():
                movie_id = row['movie_id']
                title = row['title']
                genres_str = row.get('genres', "")
                genres = [genre.strip() for genre in genres_str.split("|") if genre.strip()]
                # Add movie node with attributes
                G.add_node(movie_id, title=title, type="movie", genres=genres)
                # Connect the movie to each of its genre nodes
                for genre in genres:
                    if not G.has_node(genre):
                        G.add_node(genre, type="genre")
                    G.add_edge(movie_id, genre)
            logger.info("Movie graph built successfully.")
        except Exception as e:
            logger.error(f"Error building movie graph: {e}")
        return G

    def _logical_explanation(self, user_history: List[int], recommended_movie_id: int) -> str:
        """
        Generate a rule-based explanation by comparing genres of the recommended movie with those of movies in the user's history.

        Args:
            user_history (List[int]): List of movie IDs the user has interacted with.
            recommended_movie_id (int): The recommended movie ID.

        Returns:
            str: A natural language explanation.
        """
        rec_movie = self.movies_df[self.movies_df['movie_id'] == recommended_movie_id]
        if rec_movie.empty:
            return "No explanation available for the recommended movie."

        rec_title = rec_movie['title'].values[0]
        rec_genres = set(rec_movie['genres'].values[0].split("|"))

        explanations = []
        if user_history:
            for hist_movie_id in user_history:
                hist_movie = self.movies_df[self.movies_df['movie_id'] == hist_movie_id]
                if hist_movie.empty:
                    continue
                hist_title = hist_movie['title'].values[0]
                hist_genres = set(hist_movie['genres'].values[0].split("|"))
                common_genres = rec_genres.intersection(hist_genres)
                if common_genres:
                    common_str = ", ".join(common_genres)
                    explanations.append(
                        f"You liked '{hist_title}', which shares the genres ({common_str}) with '{rec_title}'."
                    )
            if explanations:
                return " ".join(explanations)
        return f"'{rec_title}' is recommended based on its unique attributes and overall popularity among similar users."

    def _graph_explanation(self, user_history: List[int], recommended_movie_id: int) -> str:
        """
        Generate a graph-based explanation by using the movie graph.
        It finds the genres connected to the recommended movie and identifies common genres from the user's history.

        Args:
            user_history (List[int]): List of movie IDs the user has interacted with.
            recommended_movie_id (int): The recommended movie ID.

        Returns:
            str: A natural language explanation derived from the graph.
        """
        if not self.movie_graph.has_node(recommended_movie_id):
            return "No explanation available for the recommended movie."
        rec_attrs = self.movie_graph.nodes[recommended_movie_id]
        rec_title = rec_attrs.get("title", "the recommended movie")
        rec_genres = set(rec_attrs.get("genres", []))
        explanation_lines = []
        for hist_movie_id in user_history:
            if not self.movie_graph.has_node(hist_movie_id):
                continue
            hist_attrs = self.movie_graph.nodes[hist_movie_id]
            hist_title = hist_attrs.get("title", "a movie")
            hist_genres = set(hist_attrs.get("genres", []))
            common_genres = rec_genres.intersection(hist_genres)
            if common_genres:
                common_str = ", ".join(common_genres)
                explanation_lines.append(
                    f"You liked '{hist_title}', which shares the genres ({common_str}) with '{rec_title}'."
                )
        if explanation_lines:
            return " ".join(explanation_lines)
        else:
            return f"'{rec_title}' has unique genres that might expand your viewing experience."

    def explain_recommendation(self, user_history: List[int], recommended_movie_id: int,
                               detail_level: str = "simple") -> Dict[str, Any]:
        """
        Provide an explanation for a movie recommendation using both logical and graph-based approaches.
        Depending on the detail_level, return either a simple explanation (logical only) or a detailed explanation that includes both.

        Args:
            user_history (List[int]): List of movie IDs the user has interacted with.
            recommended_movie_id (int): The movie ID of the recommendation.
            detail_level (str): "simple" returns a concise explanation; "detailed" includes both methods.

        Returns:
            Dict[str, Any]: A dictionary containing the explanation(s) and possibly additional metadata.
        """
        explanation_logical = self._logical_explanation(user_history, recommended_movie_id)
        explanation_graph = self._graph_explanation(user_history, recommended_movie_id)

        if detail_level.lower() == "detailed":
            return {
                "logical_explanation": explanation_logical,
                "graph_explanation": explanation_graph,
                "combined": f"{explanation_logical} {explanation_graph}"
            }
        else:
            # Simple mode: use the logical explanation as the primary summary
            return {
                "summary": explanation_logical
            }


if __name__ == "__main__":
    # Example usage:
    explainer = RecommendationExplainer()
    # Assume user's history contains movies with IDs 1 and 2, and we want an explanation for movie ID 1.
    user_history = [1, 2]
    recommended_movie_id = 1
    simple_explanation = explainer.explain_recommendation(user_history, recommended_movie_id, detail_level="simple")
    detailed_explanation = explainer.explain_recommendation(user_history, recommended_movie_id, detail_level="detailed")

    print("Simple Explanation:")
    print(simple_explanation)
    print("\nDetailed Explanation:")
    print(detailed_explanation)
