"""AWS Lambda entrypoint — wraps the FastAPI app for Lambda / Function URL.

Mangum translates Lambda (API Gateway v2 / Function URL) events into ASGI
calls, so the same backend.main:app runs unchanged behind a Function URL.
"""
from mangum import Mangum

from backend.main import app

handler = Mangum(app)
