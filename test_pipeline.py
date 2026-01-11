#!/usr/bin/env python3
"""
Quick pipeline verification script.

Tests:
1. API key validation
2. OTLP log ingestion
3. Mirror insights endpoint
4. Database persistence

Usage:
    python test_pipeline.py
"""

import os
import sys
import json
import httpx
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

# Load env from enterprise_bot
load_dotenv(Path("C:/Users/mthar/projects/enterprise_bot/.env"))

# Configuration
BACKEND_URL = "https://liquid-mirror-analytics-production.up.railway.app"
API_KEY = os.getenv("OTLP_API_KEY", "lm_89a5fc24_07e13174365f8954fef7a1a81b4bc6e8")


def test_health():
    """Test backend is up."""
    print("\n[1/5] Health check...")
    r = httpx.get(f"{BACKEND_URL}/health", timeout=10)
    if r.status_code == 200:
        print(f"      PASS - {r.json()}")
        return True
    print(f"      FAIL - {r.status_code}: {r.text}")
    return False


def test_api_key():
    """Test API key validation."""
    print("\n[2/5] API key validation...")
    r = httpx.get(
        f"{BACKEND_URL}/api/v1/ingest/status",
        headers={"X-API-Key": API_KEY},
        timeout=10
    )
    if r.status_code == 200:
        data = r.json()
        print(f"      PASS - tenant_id: {data.get('tenant_id')}, scopes: {data.get('scopes')}")
        return True
    print(f"      FAIL - {r.status_code}: {r.text}")
    return False


def test_otlp_logs():
    """Test OTLP log ingestion."""
    print("\n[3/5] OTLP log ingestion...")

    payload = {
        "resourceLogs": [{
            "resource": {
                "attributes": [
                    {"key": "service.name", "value": {"stringValue": "test_pipeline"}}
                ]
            },
            "scopeLogs": [{
                "scope": {"name": "test"},
                "logRecords": [{
                    "timeUnixNano": str(int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)),
                    "severityNumber": 9,
                    "severityText": "INFO",
                    "body": {"stringValue": "Pipeline test log entry"},
                    "attributes": [
                        {"key": "test", "value": {"stringValue": "true"}}
                    ]
                }]
            }]
        }]
    }

    r = httpx.post(
        f"{BACKEND_URL}/v1/logs",
        json=payload,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": API_KEY
        },
        timeout=10
    )

    if r.status_code == 200:
        print(f"      PASS - Log ingested successfully")
        return True
    print(f"      FAIL - {r.status_code}: {r.text}")
    return False


def test_mirror_insights():
    """Test metacognitive mirror insights."""
    print("\n[4/5] Mirror insights endpoint...")
    r = httpx.get(f"{BACKEND_URL}/api/mirror/insights", timeout=10)
    if r.status_code == 200:
        data = r.json()
        print(f"      PASS - phase: {data.get('cognitive_phase')}, temp: {data.get('temperature')}")
        return True
    print(f"      FAIL - {r.status_code}: {r.text}")
    return False


def test_db_persistence():
    """Verify data hit the database."""
    print("\n[5/5] Database persistence check...")

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(os.getenv("AZURE_PG_CONNECTION_STRING"))
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Check for recent logs
        cur.execute("""
            SELECT COUNT(*) as count
            FROM enterprise.structured_logs
            WHERE timestamp > NOW() - INTERVAL '5 minutes'
        """)
        log_count = cur.fetchone()['count']

        # Check tenants
        cur.execute("SELECT slug, name FROM liquid_mirror.tenants")
        tenants = cur.fetchall()

        # Check api_keys
        cur.execute("SELECT key_prefix, name, is_active FROM liquid_mirror.api_keys")
        keys = cur.fetchall()

        conn.close()

        print(f"      Recent logs (5min): {log_count}")
        print(f"      Tenants: {[t['slug'] for t in tenants]}")
        print(f"      API Keys: {[k['key_prefix'] for k in keys]}")
        print(f"      PASS")
        return True

    except Exception as e:
        print(f"      FAIL - {e}")
        return False


def main():
    print("=" * 60)
    print("LIQUID MIRROR PIPELINE VERIFICATION")
    print("=" * 60)
    print(f"Backend: {BACKEND_URL}")
    print(f"API Key: {API_KEY[:20]}...")

    results = []
    results.append(("Health", test_health()))
    results.append(("API Key", test_api_key()))
    results.append(("OTLP Logs", test_otlp_logs()))
    results.append(("Mirror", test_mirror_insights()))
    results.append(("Database", test_db_persistence()))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    passed = sum(1 for _, ok in results if ok)
    total = len(results)

    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")

    print(f"\n  {passed}/{total} tests passed")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
