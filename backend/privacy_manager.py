import sqlite3
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATABASE_PATH = Path("privacy.db")


class PrivacyManager:
    def __init__(self, db_path=DATABASE_PATH):
        """
        Initialize the PrivacyManager, connecting to the SQLite database and creating
        the user_consent table if it does not exist.
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """
        Create necessary tables for privacy management.
        """
        try:
            with self.conn:
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_consent (
                        user_id TEXT PRIMARY KEY,
                        consent INTEGER NOT NULL,   -- 1 for consent given, 0 for not given
                        timestamp TEXT NOT NULL
                    )
                """)
            logger.info("Privacy tables ensured in the database.")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise

    def record_consent(self, user_id, consent):
        """
        Record or update a user's consent status.

        Args:
            user_id (str): The unique identifier for the user.
            consent (bool): True if user gives consent, False otherwise.
        """
        timestamp = datetime.utcnow().isoformat()
        try:
            with self.conn:
                self.conn.execute("""
                    INSERT INTO user_consent (user_id, consent, timestamp)
                    VALUES (?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET 
                        consent=excluded.consent,
                        timestamp=excluded.timestamp
                """, (user_id, int(consent), timestamp))
            logger.info(f"Consent for user {user_id} recorded as {consent} at {timestamp}.")
        except Exception as e:
            logger.error(f"Error recording consent for user {user_id}: {e}")
            raise

    def get_consent(self, user_id):
        """
        Retrieve a user's consent status.

        Args:
            user_id (str): The unique identifier for the user.

        Returns:
            bool or None: True/False if consent record exists; None if not found.
        """
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT consent FROM user_consent WHERE user_id = ?", (user_id,))
            row = cur.fetchone()
            if row:
                consent_value = bool(row["consent"])
                logger.info(f"Retrieved consent for user {user_id}: {consent_value}.")
                return consent_value
            else:
                logger.info(f"No consent record found for user {user_id}.")
                return None
        except Exception as e:
            logger.error(f"Error retrieving consent for user {user_id}: {e}")
            raise

    def remove_user_data(self, user_id):
        """
        Remove a user's privacy-sensitive data.
        This method should be integrated with your system's data deletion workflow.

        Args:
            user_id (str): The unique identifier for the user.

        Note: In a full implementation, this function would delete user data from
        all data stores (ratings, interactions, etc.). Here, we simulate the process.
        """
        try:
            with self.conn:
                self.conn.execute("DELETE FROM user_consent WHERE user_id = ?", (user_id,))
            # TODO: Add deletion logic for other data (e.g., ratings, user profiles) as needed.
            logger.info(f"All privacy-related data for user {user_id} has been removed.")
        except Exception as e:
            logger.error(f"Error removing data for user {user_id}: {e}")
            raise

    def close(self):
        """
        Close the database connection.
        """
        self.conn.close()
        logger.info("Database connection closed.")


if __name__ == "__main__":
    # Example usage
    pm = PrivacyManager()

    # Record consent for a user
    user_id = "user_123"
    pm.record_consent(user_id, consent=True)

    # Retrieve the consent
    consent = pm.get_consent(user_id)
    print(f"Consent for {user_id}: {consent}")

    # Remove the user's data
    pm.remove_user_data(user_id)
    consent_after_deletion = pm.get_consent(user_id)
    print(f"Consent after deletion for {user_id}: {consent_after_deletion}")

    pm.close()
