"""
start.py — Launch the Immunisation Adviser (FastAPI backend + browser).

Usage:
    python start.py
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"
PROJECT_DIR = Path(__file__).parent


def check_embeddings():
    chunks_file = PROJECT_DIR / "data" / "chunks_with_embeddings.json"
    if not chunks_file.exists():
        print("ERROR: data/chunks_with_embeddings.json not found.")
        print("Run the following first:")
        print("  python -m ingestion.csv_to_chunks")
        print("  python -m ingestion.embed_and_index data/chunks_raw.json --local")
        sys.exit(1)


def main():
    check_embeddings()

    print(f"Starting Immunisation Adviser at {URL} ...")

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "api.main:app",
            "--host", HOST,
            "--port", str(PORT),
            "--reload",
        ],
        cwd=str(PROJECT_DIR),
    )

    # Wait for server to be ready
    import urllib.request
    for _ in range(20):
        try:
            urllib.request.urlopen(f"{URL}/health", timeout=1)
            break
        except Exception:
            time.sleep(0.5)

    webbrowser.open(URL)
    print(f"Opened {URL} in browser. Press Ctrl+C to stop.")

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
