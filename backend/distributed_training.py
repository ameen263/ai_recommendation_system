from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("DistributedTrainingALS") \
        .getOrCreate()

    try:
        # Define data path (assumes merged_data.csv is tab-separated with header)
        data_path = "backend/data/merged_data.csv"
        logger.info(f"Loading data from {data_path}...")
        # Load data; adjust options if necessary
        df = spark.read.option("delimiter", "\t") \
            .option("header", "true") \
            .csv(data_path)

        # Convert columns to appropriate data types
        df = df.withColumn("user_id", col("user_id").cast("integer")) \
            .withColumn("movie_id", col("movie_id").cast("integer")) \
            .withColumn("rating", col("rating").cast("float"))

        logger.info(f"Loaded {df.count()} records from the dataset.")

        # Split data into training (80%) and test (20%)
        train, test = df.randomSplit([0.8, 0.2], seed=42)
        logger.info("Data split into training and test sets.")

        # Initialize ALS model for collaborative filtering
        als = ALS(
            userCol="user_id",
            itemCol="movie_id",
            ratingCol="rating",
            nonnegative=True,
            implicitPrefs=False,
            coldStartStrategy="drop",  # Drop predictions for unknown users/items
            rank=100,  # Number of latent factors; can be tuned
            maxIter=10,
            regParam=0.1,
            seed=42
        )

        logger.info("Training the ALS model...")
        model = als.fit(train)
        logger.info("ALS model training completed.")

        # Evaluate the model using RMSE metric
        predictions = model.transform(test)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)
        logger.info(f"Distributed ALS model RMSE on test set: {rmse:.4f}")

        # Save the trained model
        model_save_path = "models/als_model"
        logger.info(f"Saving model to {model_save_path}...")
        model.save(model_save_path)
        logger.info("Model saved successfully.")

    except Exception as e:
        logger.exception(f"An error occurred during distributed training: {e}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
