#!/usr/bin/env python

import asyncio
import json
import os
import threading
from typing import Optional

import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from aider.coders import Coder
from aider.io import InputOutput


class StreamingCaptureIO(InputOutput):
    """Capture and queue IO for streaming responses."""
    
    def __init__(self, *args, **kwargs):
        self.queue = asyncio.Queue()
        self.current_response = []
        super().__init__(*args, **kwargs)
    
    def tool_output(self, msg, log_only=False):
        if not log_only:
            self.queue.put_nowait({"type": "tool_output", "content": msg})
        super().tool_output(msg, log_only=log_only)
    
    def tool_error(self, msg):
        self.queue.put_nowait({"type": "tool_error", "content": msg})
        super().tool_error(msg)
    
    def tool_warning(self, msg):
        self.queue.put_nowait({"type": "tool_warning", "content": msg})
        super().tool_warning(msg)
    
    def collect_response(self, chunk):
        """Collect assistant response chunks and queue them."""
        self.current_response.append(chunk)
        self.queue.put_nowait({"type": "assistant_chunk", "content": chunk})
    
    def get_full_response(self):
        """Get the concatenated response collected so far."""
        return "".join(self.current_response)
    
    def clear_response(self):
        """Clear the stored response."""
        self.current_response = []


class HeadlessServer:
    """Headless server for Aider using Starlette."""
    
    def __init__(self, coder: Coder, host: str = "0.0.0.0", port: int = 8080):
        self.coder = coder
        self.host = host
        self.port = port
        self.io = StreamingCaptureIO(
            pretty=False,
            yes=True,
            dry_run=coder.io.dry_run,
            encoding=coder.io.encoding,
        )
        self.coder.commands.io = self.io
        self.coder.yield_stream = True
        self.coder.stream = True
        
        # Handle Dreamlab specific base URL path
        self.base_url_path = os.environ.get("BASE_URL_PATH", "")
        self.app = self.create_app()
    
    def create_app(self):
        """Create the Starlette application with routes."""
        # Determine route prefix for Dreamlab environment
        prefix = ""
        if self.base_url_path:
            prefix = f"/coder/{self.port}/{self.base_url_path}"
        
        routes = [
            Route(f"{prefix}/", endpoint=self.status_endpoint, methods=["GET"]),
            Route(f"{prefix}/chat", endpoint=self.chat_endpoint, methods=["POST"]),
            Route(f"{prefix}/files", endpoint=self.files_endpoint, methods=["GET"]),
            Route(f"{prefix}/add_file", endpoint=self.add_file_endpoint, methods=["POST"]),
            Route(f"{prefix}/remove_file", endpoint=self.remove_file_endpoint, methods=["POST"]),
        ]
        
        # Configure CORS middleware
        middleware = [
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],  # In production, you should specify exact origins
                allow_methods=["GET", "POST"],
                allow_headers=["*"],
            )
        ]
        
        return Starlette(routes=routes, middleware=middleware)
    
    async def status_endpoint(self, request: Request):
        """Status endpoint to check if the server is running and provide API documentation."""
        # Generate simple API documentation
        docs = {
            "endpoints": {
                "/": "GET - Get server status and API documentation",
                "/chat": "POST - Send a message to the LLM and get a streaming response. Body: {'message': 'your message'}",
                "/files": "GET - List files in chat context and all available files",
                "/add_file": "POST - Add a file to the chat context. Body: {'filename': 'path/to/file.py'}",
                "/remove_file": "POST - Remove a file from the chat context. Body: {'filename': 'path/to/file.py'}",
            }
        }
        
        return JSONResponse({
            "status": "ok",
            "version": self.coder.__class__.__module__,
            "files_in_chat": self.coder.get_inchat_relative_files(),
            "documentation": docs,
        })
    
    async def stream_response(self, message: str):
        """Stream the response from the coder to the client."""
        # Clear any previous response
        self.io.clear_response()
        
        # Run the coder in a separate thread to not block the event loop
        def run_coder():
            try:
                self.coder._get_partial_response(message, yield_callback=self.io.collect_response)
                self.io.queue.put_nowait({"type": "done", "content": self.io.get_full_response()})
                
                # Add information about edits
                if hasattr(self.coder, "aider_edited_files") and self.coder.aider_edited_files:
                    self.io.queue.put_nowait({
                        "type": "edit_info",
                        "files": self.coder.aider_edited_files,
                        "commit_hash": getattr(self.coder, "last_aider_commit_hash", None),
                        "commit_message": getattr(self.coder, "last_aider_commit_message", None),
                    })
            except Exception as e:
                self.io.queue.put_nowait({"type": "error", "content": str(e)})
        
        # Start the coder thread
        thread = threading.Thread(target=run_coder)
        thread.daemon = True
        thread.start()
        
        # Stream the responses from the queue
        async def event_generator():
            while True:
                try:
                    item = await self.io.queue.get()
                    yield f"data: {json.dumps(item)}\n\n"
                    
                    # If done signal is received, end the stream
                    if item["type"] == "done":
                        break
                except asyncio.CancelledError:
                    break
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )
    
    async def chat_endpoint(self, request: Request):
        """Chat endpoint to send messages to the coder."""
        data = await request.json()
        message = data.get("message", "")
        
        if not message:
            return JSONResponse({"error": "Message is required"}, status_code=400)
        
        return await self.stream_response(message)
    
    async def files_endpoint(self, request: Request):
        """Endpoint to get files information."""
        return JSONResponse({
            "files_in_chat": self.coder.get_inchat_relative_files(),
            "all_files": self.coder.get_all_relative_files(),
        })
    
    async def add_file_endpoint(self, request: Request):
        """Endpoint to add a file to the chat."""
        data = await request.json()
        filename = data.get("filename", "")
        
        if not filename:
            return JSONResponse({"error": "Filename is required"}, status_code=400)
        
        try:
            self.coder.add_rel_fname(filename)
            return JSONResponse({"status": "ok", "files_in_chat": self.coder.get_inchat_relative_files()})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)
    
    async def remove_file_endpoint(self, request: Request):
        """Endpoint to remove a file from the chat."""
        data = await request.json()
        filename = data.get("filename", "")
        
        if not filename:
            return JSONResponse({"error": "Filename is required"}, status_code=400)
        
        try:
            self.coder.drop_rel_fname(filename)
            return JSONResponse({"status": "ok", "files_in_chat": self.coder.get_inchat_relative_files()})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)
    
    def run(self):
        """Run the server."""
        uvicorn.run(
            self.app,
            host=self.host,
            port=int(self.port),
            log_level="info"
        )


def launch_headless(coder: Coder, host: Optional[str] = None, port: Optional[int] = None):
    """Launch the headless server."""
    # Use environment variables if provided (for Dreamlab)
    if not host:
        host = "0.0.0.0"
    
    if not port:
        port = int(os.environ.get("CODER_PORT", 8080))
    
    # Dreamlab-specific settings (similar to GUI mode)
    base_url_path = os.environ.get("BASE_URL_PATH", "")
    
    print(f"Starting headless server on {host}:{port}")
    if base_url_path:
        print(f"Using base URL path: coder/{port}/{base_url_path}")
    print("CONTROL-C to exit...")
    
    server = HeadlessServer(coder, host, port)
    server.run()