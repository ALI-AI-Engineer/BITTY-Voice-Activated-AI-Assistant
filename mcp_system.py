import os
import psutil
import shutil
import asyncio
import subprocess
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from starlette.responses import FileResponse

# Setup logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

# ==========================
# Config
# ==========================
API_KEY = os.getenv("MCP_API_KEY", "test-key")
MAX_READ_BYTES = 5 * 1024 * 1024  # 5MB

app = FastAPI(title="MCP System Server")

# ==========================
# Auth Middleware
# ==========================

def api_key_auth(api_key: str = None):
    """Extract API key from query parameters and validate it."""
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ==========================
# Schemas
# ==========================
class ShellReq(BaseModel):
    cmd: str
    confirm: bool = False
    timeout: int = 10

class FileOpReq(BaseModel):
    src: str
    dst: Optional[str] = None
    confirm: bool = False

# ==========================
# Utils
# ==========================
def ensure_allowed(path: str):
    if ".." in path or path.startswith("/etc") or path.startswith("C:\\Windows"):
        raise HTTPException(403, "Access denied")

# ==========================
# Routes
# ==========================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "MCP System Server",
        "features": ["CPU", "Processes", "Files", "Shell"],
        "api_key_required": True,
        "version": "2.0.0"
    }

@app.get("/cpu", dependencies=[Depends(api_key_auth)])
async def cpu():
    """Get CPU and memory usage."""
    try:
        logger.info("CPU request received")
        return await asyncio.to_thread(
            lambda: {
                "cpu": psutil.cpu_percent(0.5),
                "mem": psutil.virtual_memory().percent
            }
        )
    except Exception as e:
        logger.error(f"CPU request failed: {e}")
        raise HTTPException(500, f"Failed to get CPU info: {e}")

@app.get("/ps", dependencies=[Depends(api_key_auth)])
async def ps():
    """Get list of running processes."""
    try:
        logger.info("Process list request received")
        return await asyncio.to_thread(
            lambda: [
                {"pid": p.pid, "name": p.name(), "cpu": p.cpu_percent(), "mem": p.memory_percent()}
                for p in psutil.process_iter(attrs=None)
            ]
        )
    except Exception as e:
        logger.error(f"Process list request failed: {e}")
        raise HTTPException(500, f"Failed to get process list: {e}")

@app.post("/open_app", dependencies=[Depends(api_key_auth)])
async def open_app(path: str):
    """Open an application by path."""
    try:
        ensure_allowed(path)
        logger.info(f"Opening application: {path}")
        await asyncio.to_thread(subprocess.Popen, [path])
        return {"ok": True, "path": path}
    except Exception as e:
        logger.error(f"Failed to open app {path}: {e}")
        raise HTTPException(400, str(e))

@app.post("/shell", dependencies=[Depends(api_key_auth)])
async def shell(req: ShellReq):
    """Run a shell command."""
    try:
        if not req.confirm:
            logger.info(f"Shell command needs confirmation: {req.cmd}")
            return {"needs_confirmation": True, "message": f"Confirm to run: {req.cmd}"}
        
        logger.info(f"Executing shell command: {req.cmd}")
        proc = await asyncio.create_subprocess_shell(
            req.cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=req.timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise HTTPException(400, "Command timed out")
        
        return {
            "stdout": out.decode(),
            "stderr": err.decode(),
            "returncode": proc.returncode
        }
    except Exception as e:
        logger.error(f"Shell command failed: {e}")
        raise HTTPException(400, str(e))

@app.get("/read_file", dependencies=[Depends(api_key_auth)])
async def read_file(path: str, max_bytes: int = MAX_READ_BYTES):
    """Read a file."""
    try:
        ensure_allowed(path)
        logger.info(f"Reading file: {path}")

        def _read():
            if not os.path.exists(path) or not os.path.isfile(path):
                raise HTTPException(404, "File not found")
            size = os.path.getsize(path)
            if size > max_bytes:
                return {"ok": False, "message": f"File too large ({size} bytes)."}
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return {"path": os.path.abspath(path), "size": size, "content": f.read()}

        return await asyncio.to_thread(_read)
    except Exception as e:
        logger.error(f"File read failed: {e}")
        raise HTTPException(500, f"Failed to read file: {e}")

@app.post("/write_file", dependencies=[Depends(api_key_auth)])
async def write_file(path: str, content: str):
    """Write content to a file."""
    try:
        ensure_allowed(path)
        logger.info(f"Writing file: {path}")

        def _write():
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"ok": True, "path": os.path.abspath(path)}

        return await asyncio.to_thread(_write)
    except Exception as e:
        logger.error(f"File write failed: {e}")
        raise HTTPException(500, f"Failed to write file: {e}")

@app.post("/append_file", dependencies=[Depends(api_key_auth)])
async def append_file(path: str, content: str):
    """Append content to a file."""
    try:
        ensure_allowed(path)
        logger.info(f"Appending to file: {path}")

        def _append():
            with open(path, "a", encoding="utf-8") as f:
                f.write(content)
            return {"ok": True, "path": os.path.abspath(path)}

        return await asyncio.to_thread(_append)
    except Exception as e:
        logger.error(f"File append failed: {e}")
        raise HTTPException(500, f"Failed to append to file: {e}")

@app.post("/file_op", dependencies=[Depends(api_key_auth)])
async def file_op(req: FileOpReq):
    """Perform file operations."""
    try:
        ensure_allowed(req.src)
        if req.dst:
            ensure_allowed(req.dst)

        if not req.confirm:
            logger.info(f"File operation needs confirmation: {req.src}")
            return {"needs_confirmation": True, "message": f"Confirm to operate on: {req.src}"}

        logger.info(f"Performing file operation: {req.src} -> {req.dst}")

        def _op():
            if req.dst:
                if os.path.isdir(req.src):
                    shutil.copytree(req.src, req.dst, dirs_exist_ok=True)
                else:
                    shutil.copy(req.src, req.dst)
                return {"ok": True, "src": req.src, "dst": req.dst}
            else:
                os.remove(req.src)
                return {"ok": True, "deleted": req.src}

        return await asyncio.to_thread(_op)
    except Exception as e:
        logger.error(f"File operation failed: {e}")
        raise HTTPException(500, f"Failed to perform file operation: {e}")

@app.get("/download", dependencies=[Depends(api_key_auth)])
async def download(path: str):
    """Download a file."""
    try:
        ensure_allowed(path)
        if not os.path.exists(path):
            raise HTTPException(404, "File not found")
        logger.info(f"Downloading file: {path}")
        return FileResponse(path, filename=os.path.basename(path))
    except Exception as e:
        logger.error(f"File download failed: {e}")
        raise HTTPException(500, f"Failed to download file: {e}")

@app.get("/info")
async def get_info():
    """Get server information."""
    return {
        "title": "MCP System Server",
        "description": "System operations server for CPU, processes, files, and shell commands",
        "version": "2.0.0",
        "endpoints": {
            "/health": "Health check",
            "/cpu": "Get CPU and memory usage",
            "/ps": "Get running processes",
            "/open_app": "Open application",
            "/shell": "Execute shell command",
            "/read_file": "Read file content",
            "/write_file": "Write file content",
            "/append_file": "Append to file",
            "/file_op": "File operations (copy/move/delete)",
            "/download": "Download file",
            "/info": "Server information"
        },
        "async_compatible": True
    }