"""
Minimal Modal test - Hello World
"""

import modal

app = modal.App("test-hello-world")


@app.function()
def hello(name: str = "World") -> str:
    return f"Hello, {name}!"


@app.local_entrypoint()
def main():
    print("Running hello locally...")
    result = hello.local("Local")
    print(f"Local result: {result}")
    
    print("\nRunning hello remotely...")
    result = hello.remote("Modal")
    print(f"Remote result: {result}")
