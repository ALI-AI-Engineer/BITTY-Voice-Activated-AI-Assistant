#!/usr/bin/env python3
"""
Test MCP server connectivity and functionality
"""

import asyncio
import httpx
import json

async def test_mcp_connectivity():
    """Test all MCP servers."""
    servers = {
        "System": {
            "url": "http://127.0.0.1:8000",
            "tests": [
                ("/health", "GET"),
                ("/cpu?api_key=ad78f658653f3626f76ee0cc69c55e1b58821c7dc0008480771eb2d6b2320423", "GET"),
                ("/ps?api_key=ad78f658653f3626f76ee0cc69c55e1b58821c7dc0008480771eb2d6b2320423", "GET")
            ]
        },
        "Web": {
            "url": "http://127.0.0.1:8010",
            "tests": [
                ("/health", "GET"),
                ("/search?query=python", "GET")
            ]
        },
        "Productivity": {
            "url": "http://127.0.0.1:8020",
            "tests": [
                ("/health", "GET"),
                ("/test", "GET")
            ]
        }
    }
    
    results = {}
    
    for server_name, config in servers.items():
        print(f"\n Testing {server_name} MCP Server...")
        server_results = []
        
        for endpoint, method in config["tests"]:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    if method == "GET":
                        response = await client.get(f"{config['url']}{endpoint}")
                    else:
                        response = await client.post(f"{config['url']}{endpoint}")
                    
                    if response.status_code == 200:
                        print(f"    {endpoint}: OK")
                        server_results.append(f" {endpoint}")
                    else:
                        print(f"    {endpoint}: Status {response.status_code}")
                        server_results.append(f" {endpoint} (Status: {response.status_code})")
                        
            except Exception as e:
                print(f"    {endpoint}: {e}")
                server_results.append(f" {endpoint} (Error: {e})")
        
        results[server_name] = server_results
    
    return results

async def test_email_functionality():
    """Test email sending functionality."""
    print("\n Testing Email Functionality...")
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Test email send
            email_data = {
                "to": "aliakbar477977@gmail.com",
                "subject": "Test Email",
                "message": "This is a test email from Bitty for Checking If Servers are up and running  or Not."
            }
            
            response = await client.post(
                "http://127.0.0.1:8020/email/send",
                json=email_data
            )
            
            result = response.json()
            print(f"   Email send result: {result}")
            
            return result
            
    except Exception as e:
        print(f"    Email test failed: {e}")
        return {"error": str(e)}

async def main():
    """Main test function."""
    print(" MCP Server Connectivity Test")
    print("=" * 50)
    
    # Test basic connectivity
    connectivity_results = await test_mcp_connectivity()
    
    # Test email functionality
    email_result = await test_email_functionality()
    
    # Summary
    print("\n Test Summary:")
    print("=" * 50)
    
    for server, results in connectivity_results.items():
        print(f"\n{server} MCP:")
        for result in results:
            print(f"   {result}")
    
    print(f"\nEmail Test:")
    print(f"   {email_result}")

if __name__ == "__main__":
    asyncio.run(main())