import os
import re
import sqlite3
import bcrypt
import jwt
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends, Response
from pydantic import BaseModel, constr
from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app for authentication
app = FastAPI(title="Authentication API", version="1.1")

# Enable CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
SECRET_KEY = os.environ.get('SECRET_KEY', 'my_default_secret_key')
TOKEN_EXPIRY_MINUTES = int(os.environ.get('TOKEN_EXPIRY_MINUTES', 30))
DATABASE = 'users.db'


def get_db():
    """Establish and return a new database connection."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


# Update Pydantic models to include explicit consent for data processing.
class RegisterRequest(BaseModel):
    username: str
    password: constr(min_length=8)
    consent: bool  # True if user consents to data processing


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/register")
def register(request: RegisterRequest):
    """
    Register a new user.
    - Validates password complexity.
    - Requires explicit consent (GDPR compliance).
    - Stores hashed password and consent flag in the SQLite database.
    """
    username = request.username
    password = request.password
    consent = request.consent

    if not consent:
        raise HTTPException(
            status_code=400,
            detail="Registration requires consent to data processing as per GDPR."
        )

    # Check password complexity:
    pattern = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[\W_]).{8,}$')
    if not pattern.match(password):
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters long and include an uppercase letter, a lowercase letter, a digit, and a special character."
        )

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        conn = get_db()
        cursor = conn.cursor()
        # Assuming the users table has columns: id, username, password, consent, created_at
        cursor.execute(
            "INSERT INTO users (username, password, consent, created_at) VALUES (?, ?, ?, ?)",
            (username, hashed_password, int(consent), datetime.utcnow().isoformat())
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Username already exists")
    finally:
        conn.close()
    return {"message": "User registered successfully"}


@app.post("/login")
def login(request: LoginRequest):
    """
    Log in an existing user.
    - Checks credentials.
    - Returns a JWT token on success.
    """
    username = request.username
    password = request.password
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()

    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    stored_password = user["password"]
    if isinstance(stored_password, str):
        stored_password = stored_password.encode('utf-8')
    if not bcrypt.checkpw(password.encode('utf-8'), stored_password):
        raise HTTPException(status_code=401, detail="Invalid password")

    token = jwt.encode({
        "user_id": user["id"],
        "exp": datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRY_MINUTES)
    }, SECRET_KEY, algorithm="HS256")
    return {"message": "Login successful", "token": token}


# Dependency to verify JWT token (for protected endpoints)
def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# New endpoint: Export user data (GDPR right to data portability)
@app.get("/user_data")
def export_user_data(user_id: int, token_data: dict = Depends(lambda token: verify_token(token))):
    """
    Export a user's stored data.
    This endpoint allows users to retrieve all personal data held by the system.
    """
    if token_data.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access forbidden")

    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, consent, created_at FROM users WHERE id = ?", (user_id,))
        user_data = cursor.fetchone()
        conn.close()
        if user_data is None:
            raise HTTPException(status_code=404, detail="User data not found")

        # Convert SQLite Row to dictionary
        user_dict = dict(user_data)
        return {"user_data": user_dict}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting user data: {e}")


# New endpoint: Delete user data ("Right to be Forgotten")
@app.delete("/user_data")
def delete_user_data(user_id: int, token_data: dict = Depends(lambda token: verify_token(token))):
    """
    Delete a user's personal data.
    This endpoint implements the GDPR 'right to be forgotten', removing the user's data from the system.
    """
    if token_data.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access forbidden")

    try:
        conn = get_db()
        cursor = conn.cursor()
        # Delete the user's record; in production, extend this to other related tables as needed.
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        conn.close()
        return {"message": f"All personal data for user {user_id} has been deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting user data: {e}")


# Example protected route for testing
@app.get("/protected")
def protected_route(token_data: dict = Depends(lambda token: verify_token(token))):
    return {"message": "Access granted", "user": token_data}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("auth_api:app", host="0.0.0.0", port=8000, reload=True)
