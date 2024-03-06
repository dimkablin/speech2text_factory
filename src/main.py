"""Main FastAPI entry point."""
from fastapi import FastAPI


app = FastAPI()


@app.get("/")
async def root() -> str:
    """Return a friendly HTTP greeting.

    Returns:
        str: small html page with microphone.
    """
    return "Hello World"


@app.get("/predict")
async def predict(name: str):
    """Predict function."""
    return {"message": f"Hello {name}"}
