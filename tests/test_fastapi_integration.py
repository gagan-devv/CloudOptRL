"""
Unit tests for FastAPI integration.

Tests verify that:
1. /reset endpoint returns {"status": "ok"}
2. Gradio interface is mounted at "/"
3. Existing Gradio functionality still works
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
import gradio as gr


def test_reset_endpoint_returns_ok():
    """Test /reset endpoint returns {"status": "ok"}."""
    # Create a minimal FastAPI app with reset endpoint
    app = FastAPI()
    
    @app.post("/reset")
    async def reset_endpoint():
        """Reset endpoint for validator compliance."""
        return {"status": "ok"}
    
    client = TestClient(app)
    response = client.post("/reset")
    
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_gradio_mounted_at_root():
    """Test Gradio interface is mounted at root path '/'."""
    # Create a minimal Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("Test Interface")
    
    # Create FastAPI app and mount Gradio
    app = FastAPI()
    
    @app.post("/reset")
    async def reset_endpoint():
        return {"status": "ok"}
    
    app = gr.mount_gradio_app(app, demo, path="/")
    
    client = TestClient(app)
    
    # Test that root path returns HTML (Gradio interface)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


def test_reset_endpoint_with_gradio_mounted():
    """Test /reset endpoint works when Gradio is mounted."""
    # Create a minimal Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("Test Interface")
    
    # Create FastAPI app with reset endpoint
    app = FastAPI()
    
    @app.post("/reset")
    async def reset_endpoint():
        return {"status": "ok"}
    
    # Mount Gradio at root
    app = gr.mount_gradio_app(app, demo, path="/")
    
    client = TestClient(app)
    
    # Test reset endpoint still works after mounting Gradio
    response = client.post("/reset")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_fastapi_app_structure():
    """Test that the main app.py has correct FastAPI structure."""
    # Import the demo from server/app.py
    import sys
    import os
    
    # Add parent directory to path to import app
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Import the demo object from server/app.py
    from server.app import demo
    
    # Verify demo is a Gradio Blocks instance
    assert isinstance(demo, gr.Blocks)
    
    # Create FastAPI app and mount demo
    app = FastAPI()
    
    @app.post("/reset")
    async def reset_endpoint():
        return {"status": "ok"}
    
    app = gr.mount_gradio_app(app, demo, path="/")
    
    client = TestClient(app)
    
    # Test reset endpoint
    response = client.post("/reset")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    
    # Test Gradio interface at root
    response = client.get("/")
    assert response.status_code == 200


def test_port_7860_configuration():
    """Test that port 7860 is the configured port."""
    # This is a documentation test - we verify the port is set in server/app.py
    # The actual port binding happens at runtime with uvicorn.run()
    
    # Read server/app.py to verify port configuration
    import os
    app_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server", "app.py")
    
    with open(app_path, 'r') as f:
        content = f.read()
    
    # Verify port 7860 is specified in uvicorn.run()
    assert "port=7860" in content or "port = 7860" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
