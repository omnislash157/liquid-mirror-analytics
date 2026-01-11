#!/usr/bin/env python3
"""
API Key Management for Liquid Mirror Analytics.

Usage:
    python manage_keys.py create --tenant driscoll --name "Enterprise Bot OTLP"
    python manage_keys.py list --tenant driscoll
    python manage_keys.py revoke --key lm_abc12345
    python manage_keys.py verify --key lm_abc12345_secrethere
"""

import argparse
import hashlib
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load .env from enterprise_bot
load_dotenv(Path("C:/Users/mthar/projects/enterprise_bot/.env"))


def get_connection():
    """Get database connection."""
    conn_str = os.getenv("AZURE_PG_CONNECTION_STRING")
    if not conn_str:
        print("ERROR: AZURE_PG_CONNECTION_STRING not set")
        sys.exit(1)
    return psycopg2.connect(conn_str)


def generate_api_key() -> tuple[str, str, str]:
    """
    Generate a new API key.

    Returns:
        (full_key, prefix, key_hash)

    Format: lm_{prefix}_{secret}
        - prefix: 8 chars, used for identification
        - secret: 32 chars, the actual secret
    """
    prefix = secrets.token_hex(4)  # 8 chars
    secret = secrets.token_hex(16)  # 32 chars
    full_key = f"lm_{prefix}_{secret}"
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    return full_key, f"lm_{prefix}", key_hash


def create_key(tenant_slug: str, name: str, scopes: list[str] = None) -> str:
    """Create a new API key for a tenant."""
    if scopes is None:
        scopes = ["ingest", "read"]

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Find tenant
            cur.execute(
                "SELECT id FROM liquid_mirror.tenants WHERE slug = %s",
                (tenant_slug,)
            )
            tenant = cur.fetchone()

            if not tenant:
                # Create tenant if doesn't exist
                print(f"Tenant '{tenant_slug}' not found, creating...")
                cur.execute(
                    """
                    INSERT INTO liquid_mirror.tenants (slug, name)
                    VALUES (%s, %s)
                    RETURNING id
                    """,
                    (tenant_slug, tenant_slug.title())
                )
                tenant = cur.fetchone()
                conn.commit()

            tenant_id = tenant['id']

            # Generate key
            full_key, prefix, key_hash = generate_api_key()

            # Insert key
            import json
            cur.execute(
                """
                INSERT INTO liquid_mirror.api_keys
                (tenant_id, key_prefix, key_hash, name, scopes)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (tenant_id, prefix, key_hash, name, json.dumps(scopes))
            )
            conn.commit()

            print(f"\nAPI Key created successfully!")
            print(f"=" * 60)
            print(f"Tenant:  {tenant_slug}")
            print(f"Name:    {name}")
            print(f"Scopes:  {', '.join(scopes)}")
            print(f"Prefix:  {prefix}")
            print(f"=" * 60)
            print(f"\nFULL API KEY (save this, it won't be shown again):\n")
            print(f"  {full_key}")
            print(f"\n" + "=" * 60)
            print(f"\nAdd to your .env:")
            print(f"  OTLP_API_KEY={full_key}")
            print()

            return full_key

    finally:
        conn.close()


def list_keys(tenant_slug: str = None) -> None:
    """List API keys, optionally filtered by tenant."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if tenant_slug:
                cur.execute(
                    """
                    SELECT k.*, t.slug as tenant_slug
                    FROM liquid_mirror.api_keys k
                    JOIN liquid_mirror.tenants t ON k.tenant_id = t.id
                    WHERE t.slug = %s
                    ORDER BY k.created_at DESC
                    """,
                    (tenant_slug,)
                )
            else:
                cur.execute(
                    """
                    SELECT k.*, t.slug as tenant_slug
                    FROM liquid_mirror.api_keys k
                    JOIN liquid_mirror.tenants t ON k.tenant_id = t.id
                    ORDER BY t.slug, k.created_at DESC
                    """
                )

            keys = cur.fetchall()

            if not keys:
                print("No API keys found.")
                return

            print(f"\n{'Prefix':<15} {'Tenant':<15} {'Name':<25} {'Active':<8} {'Last Used':<20}")
            print("-" * 85)

            for key in keys:
                last_used = key['last_used_at'].strftime("%Y-%m-%d %H:%M") if key['last_used_at'] else "Never"
                active = "Yes" if key['is_active'] else "No"
                print(f"{key['key_prefix']:<15} {key['tenant_slug']:<15} {(key['name'] or '-'):<25} {active:<8} {last_used:<20}")

            print()

    finally:
        conn.close()


def revoke_key(key_prefix: str) -> None:
    """Revoke an API key by prefix."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE liquid_mirror.api_keys
                SET is_active = false
                WHERE key_prefix = %s
                RETURNING id
                """,
                (key_prefix,)
            )

            if cur.rowcount == 0:
                print(f"No key found with prefix: {key_prefix}")
            else:
                conn.commit()
                print(f"Key {key_prefix} has been revoked.")

    finally:
        conn.close()


def verify_key(full_key: str) -> None:
    """Verify an API key is valid."""
    if not full_key.startswith("lm_"):
        print("Invalid key format (must start with 'lm_')")
        return

    key_hash = hashlib.sha256(full_key.encode()).hexdigest()

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT k.*, t.slug as tenant_slug
                FROM liquid_mirror.api_keys k
                JOIN liquid_mirror.tenants t ON k.tenant_id = t.id
                WHERE k.key_hash = %s
                """,
                (key_hash,)
            )

            key = cur.fetchone()

            if not key:
                print("INVALID: Key not found")
                return

            if not key['is_active']:
                print("INVALID: Key has been revoked")
                return

            if key['expires_at'] and key['expires_at'] < datetime.now(timezone.utc):
                print("INVALID: Key has expired")
                return

            print(f"VALID")
            print(f"  Tenant: {key['tenant_slug']}")
            print(f"  Name:   {key['name']}")
            print(f"  Scopes: {', '.join(key['scopes'])}")

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Liquid Mirror API Key Management")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create
    create_parser = subparsers.add_parser("create", help="Create a new API key")
    create_parser.add_argument("--tenant", "-t", required=True, help="Tenant slug")
    create_parser.add_argument("--name", "-n", required=True, help="Key name/description")
    create_parser.add_argument("--scopes", "-s", nargs="+", default=["ingest", "read"], help="Scopes (default: ingest read)")

    # list
    list_parser = subparsers.add_parser("list", help="List API keys")
    list_parser.add_argument("--tenant", "-t", help="Filter by tenant slug")

    # revoke
    revoke_parser = subparsers.add_parser("revoke", help="Revoke an API key")
    revoke_parser.add_argument("--key", "-k", required=True, help="Key prefix (e.g., lm_abc12345)")

    # verify
    verify_parser = subparsers.add_parser("verify", help="Verify an API key")
    verify_parser.add_argument("--key", "-k", required=True, help="Full API key")

    args = parser.parse_args()

    if args.command == "create":
        create_key(args.tenant, args.name, args.scopes)
    elif args.command == "list":
        list_keys(args.tenant)
    elif args.command == "revoke":
        revoke_key(args.key)
    elif args.command == "verify":
        verify_key(args.key)


if __name__ == "__main__":
    main()
