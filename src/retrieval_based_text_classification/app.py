from fastapi import FastAPI
import uvicorn
import os
from loguru import logger
import traceback

from retrieval_based_text_classification.config import Config
from retrieval_based_text_classification.routers import base_router
from retrieval_based_text_classification.logging_config import LoggingSetup

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    # Initialize logging
    cfg = Config().get_config()
    worker_id = os.getpid()
    LoggingSetup.setup(cfg.app.log_dir, worker_pid=worker_id)
    logger.info("-" * 21 + "initilizing service" + "-" * 21)
    
    # Create FastAPI application
    app = FastAPI(
        title="Retrieval based Text Classification",
        description="Text Classification Service API"
    )
    
    # Initialize realtime voice conversion service
    app.state.cfg = cfg
    
    # Include routers
    logger.info("registering routers...")
    app.include_router(base_router)
    
    logger.info("-" * 21 + "service initialized" + "-" * 21)
    return app

def main() -> None:
    """Main function to run the FastAPI application."""
    app_config = Config().get_config().app
    logger.info(f"Starting RTC service on {app_config.host}:{app_config.port}")
    logger.info(f"Number of workers: {app_config.workers}")
    
    uvicorn.run(
        "retrieval_based_text_classification.app:create_app",
        host=app_config.host,
        port=app_config.port,
        workers=app_config.workers,
        factory=True,
        log_config=None  # Forbid default logging config
    )
    
if __name__ == "__main__":
    main()