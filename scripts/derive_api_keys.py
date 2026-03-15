#!/usr/bin/env python3
"""Derive Polymarket CLOB API credentials from a private key.

Usage:
    python scripts/derive_api_keys.py

The script will:
  1. Read POLYMARKET_PRIVATE_KEY from .env (or prompt you)
  2. Derive API Key, Secret, and Passphrase
  3. Automatically update your .env file with the credentials
"""

import os
import sys
import re
from pathlib import Path

# Resolve project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def load_private_key() -> str:
    """Try to read POLYMARKET_PRIVATE_KEY from .env or environment."""
    # Check .env file first
    if ENV_FILE.exists():
        content = ENV_FILE.read_text()
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("POLYMARKET_PRIVATE_KEY=") and not line.endswith("="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    return val

    # Check environment variable
    val = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
    if val:
        return val

    # Prompt user
    print("⚠️  POLYMARKET_PRIVATE_KEY not found in .env or environment.")
    val = input("Enter your Polymarket private key (0x...): ").strip()
    if not val:
        print("❌ No private key provided. Exiting.")
        sys.exit(1)
    return val


def derive_credentials(private_key: str) -> dict:
    """Derive API credentials using py-clob-client."""
    try:
        from py_clob_client.client import ClobClient  # type: ignore[import-untyped]
    except ImportError:
        print("❌ py-clob-client not installed. Installing...")
        os.system(f"{sys.executable} -m pip install py-clob-client")
        from py_clob_client.client import ClobClient  # type: ignore[import-untyped]

    print("🔑 Deriving API credentials from private key...")
    client = ClobClient(
        host="https://clob.polymarket.com",
        key=private_key,
        chain_id=137,  # Polygon mainnet
    )

    creds = client.derive_api_key()
    return {
        "POLYMARKET_API_KEY": creds.api_key,
        "POLYMARKET_API_SECRET": creds.api_secret,
        "POLYMARKET_API_PASSPHRASE": creds.api_passphrase,
    }


def update_env_file(creds: dict) -> None:
    """Update or append credentials in .env file."""
    if not ENV_FILE.exists():
        print(f"❌ .env file not found at {ENV_FILE}")
        print("   Creating one from .env.example...")
        example = PROJECT_ROOT / ".env.example"
        if example.exists():
            ENV_FILE.write_text(example.read_text())
        else:
            ENV_FILE.write_text("")

    content = ENV_FILE.read_text()

    for key, value in creds.items():
        # Replace existing line or append
        pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
        new_line = f"{key}={value}"
        if pattern.search(content):
            content = pattern.sub(new_line, content)
        else:
            content = content.rstrip("\n") + f"\n{new_line}\n"

    ENV_FILE.write_text(content)


def main() -> None:
    print("=" * 60)
    print("  Polymarket CLOB API Key Derivation Tool")
    print("=" * 60)
    print()

    private_key = load_private_key()
    masked = private_key[:6] + "..." + private_key[-4:]
    print(f"  Using private key: {masked}")
    print()

    creds = derive_credentials(private_key)

    print()
    print("✅ Credentials derived successfully!")
    print()
    print(f"  POLYMARKET_API_KEY        = {creds['POLYMARKET_API_KEY']}")
    print(f"  POLYMARKET_API_SECRET     = {creds['POLYMARKET_API_SECRET'][:12]}...")
    print(f"  POLYMARKET_API_PASSPHRASE = {creds['POLYMARKET_API_PASSPHRASE'][:12]}...")
    print()

    update_env_file(creds)
    print(f"✅ .env file updated at: {ENV_FILE}")
    print()
    print("Done! You can now run the bot.")


if __name__ == "__main__":
    main()
