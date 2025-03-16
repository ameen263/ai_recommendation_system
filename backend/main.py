from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import app as recommendation_app  # Recommendation API
from auth_api import app as auth_app  # Authentication API
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request


# Constants
APP_TITLE = "Unified AI Movie Recommendation API"
APP_VERSION = "1.0"
ALLOWED_ORIGINS = ["*"]  # Adjust for production
ALLOWED_CREDENTIALS = True
ALLOWED_METHODS = ["*"]
ALLOWED_HEADERS = ["*"]

# Initialize the app
app = FastAPI(title=APP_TITLE, version=APP_VERSION)

templates = Jinja2Templates(directory="../frontend/build")

app.mount('/static', StaticFiles(directory="../frontend/build/static"), 'static')


# Middleware configuration
def add_cors_middleware(application: FastAPI):
    application.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=ALLOWED_CREDENTIALS,
        allow_methods=ALLOWED_METHODS,
        allow_headers=ALLOWED_HEADERS,
    )



# Apply CORS middleware
add_cors_middleware(app)

# Mount sub-applications
app.mount("/api", recommendation_app)
app.mount("/auth", auth_app)



# # Define root endpoint
# @app.get("/")
# def get_root_endpoint():
#     """
#     Root endpoint providing basic API guidance.
#     """
#     return {
#         "message": "Welcome to the Unified API. Use /auth for authentication and /api for recommendations."
#     }

@app.get("/{rest_of_path:path}")
async def react_app(req: Request, rest_of_path: str):
    return templates.TemplateResponse('index.html', { 'request': req })


# Entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
