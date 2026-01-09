"""Run liquid_mirror schema migration."""
import os
from pathlib import Path
import psycopg2
from dotenv import load_dotenv

# Load .env from enterprise_bot
load_dotenv(Path("C:/Users/mthar/projects/enterprise_bot/.env"))

conn = psycopg2.connect(os.getenv("AZURE_PG_CONNECTION_STRING"))
cur = conn.cursor()

print("Running liquid_mirror schema migration...")

# 1. Create schema
print("1. Creating schema...")
cur.execute("CREATE SCHEMA IF NOT EXISTS liquid_mirror;")
conn.commit()

# 2. Create tables
print("2. Creating tables...")

cur.execute("""
CREATE TABLE IF NOT EXISTS liquid_mirror.tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(255) UNIQUE,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    is_active BOOLEAN DEFAULT true
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS liquid_mirror.api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    tenant_source VARCHAR(50) NOT NULL DEFAULT 'liquid_mirror',
    key_prefix VARCHAR(8) NOT NULL,
    key_hash VARCHAR(64) NOT NULL UNIQUE,
    name VARCHAR(255),
    scopes JSONB DEFAULT '["ingest", "read"]',
    rate_limit_per_minute INTEGER DEFAULT 1000,
    created_at TIMESTAMPTZ DEFAULT now(),
    last_used_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT true
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS liquid_mirror.query_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    user_email VARCHAR(255),
    session_id VARCHAR(255),
    trace_id VARCHAR(64),
    query_text TEXT NOT NULL,
    query_embedding VECTOR(1024),
    department VARCHAR(100),
    response_time_ms DOUBLE PRECISION,
    tokens_input INTEGER,
    tokens_output INTEGER,
    model_used VARCHAR(100),
    complexity_score DOUBLE PRECISION,
    intent_type VARCHAR(50),
    specificity_score DOUBLE PRECISION,
    temporal_urgency VARCHAR(20),
    is_multi_part BOOLEAN DEFAULT false,
    cognitive_phase VARCHAR(20),
    semantic_drift_magnitude DOUBLE PRECISION,
    drift_signal VARCHAR(30),
    query_cluster_id INTEGER,
    memory_ids_retrieved TEXT[],
    retrieval_scores DOUBLE PRECISION[],
    created_at TIMESTAMPTZ DEFAULT now()
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS liquid_mirror.session_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    user_email VARCHAR(255),
    session_id VARCHAR(255),
    trace_id VARCHAR(64),
    event_data JSONB DEFAULT '{}',
    error_type VARCHAR(100),
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS liquid_mirror.insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    insight_type VARCHAR(50) NOT NULL,
    severity VARCHAR(10) NOT NULL,
    title VARCHAR(255),
    description TEXT NOT NULL,
    metrics JSONB DEFAULT '{}',
    suggested_action TEXT,
    estimated_impact TEXT,
    related_user_email VARCHAR(255),
    related_session_id VARCHAR(255),
    related_query_ids UUID[],
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by VARCHAR(255),
    resolved_at TIMESTAMPTZ,
    resolved_by VARCHAR(255),
    resolution_notes TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS liquid_mirror.memory_temperatures (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    memory_id VARCHAR(255) NOT NULL,
    temperature DOUBLE PRECISION NOT NULL,
    access_count INTEGER DEFAULT 0,
    last_access_at TIMESTAMPTZ,
    burst_intensity DOUBLE PRECISION,
    burst_detected_at TIMESTAMPTZ,
    community_id INTEGER,
    co_accessed_memories TEXT[],
    snapshot_time TIMESTAMPTZ DEFAULT now(),
    UNIQUE(tenant_id, memory_id, snapshot_time)
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS liquid_mirror.cognitive_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    phase VARCHAR(20) NOT NULL,
    phase_confidence DOUBLE PRECISION,
    temperature DOUBLE PRECISION,
    focus_score DOUBLE PRECISION,
    stability_score DOUBLE PRECISION,
    access_entropy DOUBLE PRECISION,
    query_entropy DOUBLE PRECISION,
    drift_magnitude DOUBLE PRECISION,
    drift_signal VARCHAR(30),
    dominant_topics JSONB DEFAULT '[]',
    query_count_window INTEGER,
    user_email VARCHAR(255),
    session_id VARCHAR(255),
    snapshot_time TIMESTAMPTZ DEFAULT now()
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS liquid_mirror.predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    user_email VARCHAR(255),
    session_id VARCHAR(255),
    context_memory_ids TEXT[],
    predicted_memory_ids TEXT[] NOT NULL,
    confidence_scores DOUBLE PRECISION[] NOT NULL,
    actual_accessed_ids TEXT[],
    accuracy DOUBLE PRECISION,
    validated_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now()
);
""")

conn.commit()
print("   Tables created.")

# 3. Create indexes
print("3. Creating indexes...")
indexes = [
    "CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON liquid_mirror.api_keys(key_prefix);",
    "CREATE INDEX IF NOT EXISTS idx_api_keys_tenant ON liquid_mirror.api_keys(tenant_id);",
    "CREATE INDEX IF NOT EXISTS idx_query_events_tenant ON liquid_mirror.query_events(tenant_id);",
    "CREATE INDEX IF NOT EXISTS idx_query_events_created ON liquid_mirror.query_events(created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_query_events_session ON liquid_mirror.query_events(tenant_id, session_id);",
    "CREATE INDEX IF NOT EXISTS idx_query_events_user ON liquid_mirror.query_events(tenant_id, user_email);",
    "CREATE INDEX IF NOT EXISTS idx_query_events_phase ON liquid_mirror.query_events(tenant_id, cognitive_phase);",
    "CREATE INDEX IF NOT EXISTS idx_session_events_tenant ON liquid_mirror.session_events(tenant_id);",
    "CREATE INDEX IF NOT EXISTS idx_session_events_created ON liquid_mirror.session_events(created_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_session_events_type ON liquid_mirror.session_events(tenant_id, event_type);",
    "CREATE INDEX IF NOT EXISTS idx_session_events_session ON liquid_mirror.session_events(tenant_id, session_id);",
    "CREATE INDEX IF NOT EXISTS idx_insights_tenant ON liquid_mirror.insights(tenant_id);",
    "CREATE INDEX IF NOT EXISTS idx_insights_severity ON liquid_mirror.insights(tenant_id, severity);",
    "CREATE INDEX IF NOT EXISTS idx_insights_type ON liquid_mirror.insights(tenant_id, insight_type);",
    "CREATE INDEX IF NOT EXISTS idx_memory_temp_tenant ON liquid_mirror.memory_temperatures(tenant_id);",
    "CREATE INDEX IF NOT EXISTS idx_cognitive_tenant ON liquid_mirror.cognitive_snapshots(tenant_id);",
    "CREATE INDEX IF NOT EXISTS idx_cognitive_time ON liquid_mirror.cognitive_snapshots(tenant_id, snapshot_time DESC);",
    "CREATE INDEX IF NOT EXISTS idx_cognitive_phase ON liquid_mirror.cognitive_snapshots(tenant_id, phase);",
    "CREATE INDEX IF NOT EXISTS idx_predictions_tenant ON liquid_mirror.predictions(tenant_id);",
    "CREATE INDEX IF NOT EXISTS idx_predictions_time ON liquid_mirror.predictions(tenant_id, created_at DESC);",
]
for idx in indexes:
    cur.execute(idx)
conn.commit()
print("   Indexes created.")

# 4. Enable RLS
print("4. Enabling RLS...")
rls_tables = [
    "query_events", "session_events", "insights",
    "memory_temperatures", "cognitive_snapshots", "predictions"
]
for table in rls_tables:
    cur.execute(f"ALTER TABLE liquid_mirror.{table} ENABLE ROW LEVEL SECURITY;")
conn.commit()
print("   RLS enabled.")

# 5. Create RLS policies
print("5. Creating RLS policies...")
for table in rls_tables:
    cur.execute(f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_policies
                WHERE policyname = 'tenant_isolation'
                AND tablename = '{table}'
                AND schemaname = 'liquid_mirror'
            ) THEN
                CREATE POLICY tenant_isolation ON liquid_mirror.{table}
                FOR ALL USING (tenant_id = current_setting('app.current_tenant', true)::uuid);
            END IF;
        END $$;
    """)
conn.commit()
print("   RLS policies created.")

# 6. Create helper functions
print("6. Creating helper functions...")
cur.execute("""
CREATE OR REPLACE FUNCTION liquid_mirror.set_tenant_context(p_tenant_id UUID)
RETURNS VOID AS $$
BEGIN
    PERFORM set_config('app.current_tenant', p_tenant_id::text, false);
END;
$$ LANGUAGE plpgsql;
""")

cur.execute("""
CREATE OR REPLACE FUNCTION liquid_mirror.get_current_tenant()
RETURNS UUID AS $$
BEGIN
    RETURN current_setting('app.current_tenant', true)::uuid;
END;
$$ LANGUAGE plpgsql;
""")
conn.commit()
print("   Functions created.")

# 7. Verify
print("\n7. Verifying migration...")
cur.execute("""
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'liquid_mirror' AND table_type = 'BASE TABLE'
ORDER BY table_name
""")
tables = [row[0] for row in cur.fetchall()]
print(f"   Tables: {tables}")

cur.close()
conn.close()

print("\nMigration completed successfully!")
print(f"Created {len(tables)} tables in liquid_mirror schema.")
