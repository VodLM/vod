import os

import uvicorn
from fastapi import FastAPI

import src
from api import router

PORT = int(os.getenv("APP_PORT", "8080"))

app = FastAPI(
    title="ds-template-repo",
    description="Template repository for data science projects.",
    version=src.__version__,
)

app.include_router(router.router)


@app.get("/", summary="Health check")
async def root_handler() -> str:
    """Root endpoint for health check."""
    return "OK"


def main() -> None:
    """Main function to run ds-template-repo API."""
    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
