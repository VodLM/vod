import os

import uvicorn
from fastapi import FastAPI

import raffle_ds_research
from raffle_ds_research.api import router

PORT = int(os.getenv("APP_PORT", "8080"))

app = FastAPI(
    title="raffle-ds-research",
    description="Template repository for data science projects.",
    version=raffle_ds_research.__version__,
)

app.include_router(router.router)


@app.get("/", summary="Health check")
async def root_handler() -> str:
    """Root endpoint for health check."""
    return "OK"


def main() -> None:
    """Main function to run raffle-ds-research API."""
    uvicorn.run(app, host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
