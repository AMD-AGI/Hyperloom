"""Entry point for: python -m hyperloom.agents.kernel"""
import sys
from pathlib import Path

def main():
    print(f"Kernel agent tools at: {Path(__file__).parent / 'tools'}")
    print("Use: python -m hyperloom.agents.kernel.tools.<tool_name>")

if __name__ == "__main__":
    main()
