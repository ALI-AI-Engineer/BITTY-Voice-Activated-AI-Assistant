# start_servers.py
import subprocess
import sys
import os
import signal
import time
import requests

def run_server(name, file, port):
    """Run each MCP server with uvicorn in a subprocess."""
    print(f" Starting {name} on port {port}...")
    
    return subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        f"{file}:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--reload"   # auto-reload for dev
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def check_server_health(port, timeout=10):
    """Check if server is responding."""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print("\n Shutting down all MCP servers...")
    for p in processes:
        p.terminate()
        p.wait()
    print(" All servers stopped.")
    sys.exit(0)

if __name__ == "__main__":
    processes = []
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # System MCP
        processes.append(run_server("System MCP", "mcp_system", 8000))
        time.sleep(2)  # Give server time to start

        # Web MCP
        processes.append(run_server("Web MCP", "mcp_web", 8010))
        time.sleep(2)

        # Productivity MCP
        processes.append(run_server("Productivity MCP", "mcp_productivity", 8020))
        time.sleep(2)

        print("  All MCP servers started:")
        print("   - System MCP       -> http://localhost:8000")
        print("   - Web MCP          -> http://localhost:8010")
        print("   - Productivity MCP -> http://localhost:8020")
        
        
        # Check server health
        print("\n Checking server health...")
        for name, port in [("System", 8000), ("Web", 8010), ("Productivity", 8020)]:
            if check_server_health(port):
                print(f"    {name} MCP: Healthy")
            else:
                print(f"    {name} MCP: Not responding")

        print("\n Servers are running. Press Ctrl+C to stop all servers.")
        
        # Keep running until stopped
        for p in processes:
            p.wait()

    except KeyboardInterrupt:
        print("\n Shutting down all MCP servers...")
        for p in processes:
            p.terminate()
            p.wait()
        print(" All servers stopped.")
    except Exception as e:
        print(f" Error: {e}")
        for p in processes:
            p.terminate()