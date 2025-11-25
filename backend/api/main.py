"""
Main FastAPI Application.

This is the entry point for the ML Monitoring System API.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import uvicorn
from dotenv import load_dotenv

# Import routes
from backend.api.routes import predictions, monitoring, models as models_routes

# Import core components
from backend.ml.model import PricePredictor
try:
    from backend.ml.drift_detector import DriftDetector
except ImportError:
    DriftDetector = None
    print("Warning: DriftDetector not available due to dependency issues")
from backend.monitoring.metrics import MetricsCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


# Application state
app_state = {
    'model': None,
    'drift_detector': None,
    'metrics_collector': None,
    'db_manager': None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler for startup and shutdown events.
    """
    # Startup
    logger.info("Starting ML Monitoring System API...")
    
    try:
        # Initialize metrics collector
        logger.info("Initializing metrics collector...")
        app_state['metrics_collector'] = MetricsCollector()
        
        # Initialize drift detector
        if DriftDetector is not None:
            logger.info("Initializing drift detector...")
            drift_threshold = float(os.getenv("DRIFT_THRESHOLD", "0.25"))
            warning_threshold = float(os.getenv("DRIFT_WARNING_THRESHOLD", "0.20"))
            app_state['drift_detector'] = DriftDetector(
                drift_threshold=drift_threshold,
                warning_threshold=warning_threshold
            )
        else:
            logger.warning("Drift detector skipped - evidently library not available")
            app_state['drift_detector'] = None
        
        # Initialize model (placeholder - load from MLflow in production)
        logger.info("Loading production model...")
        app_state['model'] = PricePredictor()
        # TODO: Load actual model from MLflow
        # app_state['model'].load_model("path/to/production/model.pkl")
        logger.warning("Using placeholder model - implement MLflow loading")
        
        # Initialize database manager
        logger.info("Initializing database connection...")
        try:
            from backend.data.db_manager import DatabaseManager
            app_state['db_manager'] = DatabaseManager()
            app_state['db_manager'].init_db()
            logger.info("✓ Database initialized")
        except Exception as e:
            logger.warning(f"Database initialization failed: {str(e)}")
            logger.warning("Continuing without database - some features will be limited")
            app_state['db_manager'] = None
        
        # Set dependencies in route modules
        predictions.set_model(app_state['model'])
        predictions.set_metrics_collector(app_state['metrics_collector'])
        predictions.set_db_manager(app_state['db_manager'])
        
        monitoring.set_drift_detector(app_state['drift_detector'])
        monitoring.set_metrics_collector(app_state['metrics_collector'])
        monitoring.set_db_manager(app_state['db_manager'])
        monitoring.set_model(app_state['model'])
        
        models_routes.set_db_manager(app_state['db_manager'])
        # models_routes.set_retrainer(retrainer)  # TODO: Initialize retrainer
        
        # Set model version info in metrics
        if app_state['metrics_collector']:
            app_state['metrics_collector'].update_model_version(
                "v1.0.0",
                metadata={
                    "algorithm": "XGBoost",
                    "status": "development"
                }
            )
        
        logger.info("✓ API startup complete")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down ML Monitoring System API...")
    
    try:
        # Close database connections
        if app_state['db_manager']:
            app_state['db_manager'].close()
            logger.info("✓ Database connections closed")
        
        # Save metrics state (if needed)
        # TODO: Implement metrics persistence if required
        
        logger.info("✓ API shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


# Create FastAPI app
app = FastAPI(
    title="ML Monitoring System API",
    description="Production ML model monitoring with drift detection and automated retraining",
    version="1.0.0",
    lifespan=lifespan
)


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request ID Middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to each request."""
    import uuid
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response


# Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = datetime.now()
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = (datetime.now() - start_time).total_seconds()
    
    # Log request
    logger.info(
        f"{request.method} {request.url.path} "
        f"- {response.status_code} "
        f"- {duration:.3f}s"
    )
    
    return response


# Error Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG", "false").lower() == "true" else "An error occurred",
            "timestamp": datetime.now().isoformat(),
            "request_id": getattr(request.state, "request_id", None)
        }
    )


# Include routers
app.include_router(predictions.router, prefix="/api/v1")
app.include_router(monitoring.router, prefix="/api/v1")
app.include_router(models_routes.router, prefix="/api/v1")


# Root endpoint
@app.get("/")
async def root():
    """
    API root endpoint with basic information.
    """
    return {
        "name": "ML Monitoring System API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "metrics": "/metrics",
            "predictions": "/api/v1/predict",
            "monitoring": "/api/v1/monitoring",
            "models": "/api/v1/models"
        }
    }


# Health check endpoint
@app.get("/health")
async def health():
    """
    Quick health check endpoint.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": app_state['model'] is not None,
        "drift_detector": app_state['drift_detector'] is not None,
        "metrics_collector": app_state['metrics_collector'] is not None
    }


# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    """
    Expose Prometheus metrics.
    """
    if app_state['metrics_collector']:
        metrics_data = app_state['metrics_collector'].expose_metrics()
        return Response(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST
        )
    else:
        return Response(
            content="# Metrics not available\n",
            media_type=CONTENT_TYPE_LATEST
        )


# Main entry point
if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "backend.api.main:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level="info"
    )
