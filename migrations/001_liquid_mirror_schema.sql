-- Liquid Mirror Analytics - Multi-Tenant Schema
-- Migration: 001_liquid_mirror_schema.sql
-- Date: 2026-01-09
-- Description: Create liquid_mirror schema with RLS for multi-tenant analytics
--
-- ROLLBACK: See bottom of file for rollback SQL

-- =============================================================================
-- SCHEMA
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS liquid_mirror;

-- =============================================================================
-- TENANTS (for standalone customers not in enterprise.tenants)
-- =============================================================================

CREATE TABLE liquid_mirror.tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(255) UNIQUE,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    is_active BOOLEAN DEFAULT true
);

COMMENT ON TABLE liquid_mirror.tenants IS 'Standalone Liquid Mirror customers (enterprise_bot uses enterprise.tenants)';

-- =============================================================================
-- API KEYS (ingestion authentication)
-- =============================================================================

CREATE TABLE liquid_mirror.api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    tenant_source VARCHAR(50) NOT NULL DEFAULT 'liquid_mirror', -- 'enterprise' or 'liquid_mirror'
    key_prefix VARCHAR(8) NOT NULL, -- First 8 chars for identification (lm_xxx...)
    key_hash VARCHAR(64) NOT NULL UNIQUE, -- SHA-256 hash of full key
    name VARCHAR(255),
    scopes JSONB DEFAULT '["ingest", "read"]',
    rate_limit_per_minute INTEGER DEFAULT 1000,
    created_at TIMESTAMPTZ DEFAULT now(),
    last_used_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT true
);

CREATE INDEX idx_api_keys_prefix ON liquid_mirror.api_keys(key_prefix);
CREATE INDEX idx_api_keys_tenant ON liquid_mirror.api_keys(tenant_id);

COMMENT ON TABLE liquid_mirror.api_keys IS 'API keys for ingestion auth. tenant_source indicates where tenant_id lives.';

-- =============================================================================
-- QUERY EVENTS (the core analytics table)
-- =============================================================================

CREATE TABLE liquid_mirror.query_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,

    -- Identity
    user_email VARCHAR(255),
    session_id VARCHAR(255),
    trace_id VARCHAR(64),

    -- Query data
    query_text TEXT NOT NULL,
    query_embedding VECTOR(1024),
    department VARCHAR(100),

    -- Performance
    response_time_ms DOUBLE PRECISION,
    tokens_input INTEGER,
    tokens_output INTEGER,
    model_used VARCHAR(100),

    -- Heuristics (from existing analytics)
    complexity_score DOUBLE PRECISION,
    intent_type VARCHAR(50),
    specificity_score DOUBLE PRECISION,
    temporal_urgency VARCHAR(20),
    is_multi_part BOOLEAN DEFAULT false,

    -- MetacognitiveMirror enrichment
    cognitive_phase VARCHAR(20), -- exploration/exploitation/learning/consolidation/idle/crisis
    semantic_drift_magnitude DOUBLE PRECISION,
    drift_signal VARCHAR(30), -- topic_shift/semantic_expansion/semantic_collapse/etc
    query_cluster_id INTEGER,
    memory_ids_retrieved TEXT[], -- UUIDs of memories returned
    retrieval_scores DOUBLE PRECISION[],

    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_query_events_tenant ON liquid_mirror.query_events(tenant_id);
CREATE INDEX idx_query_events_created ON liquid_mirror.query_events(created_at DESC);
CREATE INDEX idx_query_events_session ON liquid_mirror.query_events(tenant_id, session_id);
CREATE INDEX idx_query_events_user ON liquid_mirror.query_events(tenant_id, user_email);
CREATE INDEX idx_query_events_phase ON liquid_mirror.query_events(tenant_id, cognitive_phase);
CREATE INDEX idx_query_events_drift ON liquid_mirror.query_events(tenant_id, drift_signal)
    WHERE drift_signal IS NOT NULL;

COMMENT ON TABLE liquid_mirror.query_events IS 'Query logs with cognitive analysis from MetacognitiveMirror';

-- =============================================================================
-- SESSION EVENTS (login, logout, errors, etc)
-- =============================================================================

CREATE TABLE liquid_mirror.session_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    event_type VARCHAR(50) NOT NULL, -- login/logout/error/department_switch/etc
    user_email VARCHAR(255),
    session_id VARCHAR(255),
    trace_id VARCHAR(64),
    event_data JSONB DEFAULT '{}',
    error_type VARCHAR(100),
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_session_events_tenant ON liquid_mirror.session_events(tenant_id);
CREATE INDEX idx_session_events_created ON liquid_mirror.session_events(created_at DESC);
CREATE INDEX idx_session_events_type ON liquid_mirror.session_events(tenant_id, event_type);
CREATE INDEX idx_session_events_session ON liquid_mirror.session_events(tenant_id, session_id);

COMMENT ON TABLE liquid_mirror.session_events IS 'Session lifecycle and error events';

-- =============================================================================
-- INSIGHTS (the magic - system-generated recommendations)
-- =============================================================================

CREATE TABLE liquid_mirror.insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,

    -- Insight classification
    insight_type VARCHAR(50) NOT NULL, -- performance_degradation/memory_concentration/cognitive_instability/semantic_collapse/etc
    severity VARCHAR(10) NOT NULL, -- info/warning/critical

    -- Content
    title VARCHAR(255),
    description TEXT NOT NULL,
    metrics JSONB DEFAULT '{}',
    suggested_action TEXT,
    estimated_impact TEXT,

    -- Context
    related_user_email VARCHAR(255),
    related_session_id VARCHAR(255),
    related_query_ids UUID[],

    -- Lifecycle
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by VARCHAR(255),
    resolved_at TIMESTAMPTZ,
    resolved_by VARCHAR(255),
    resolution_notes TEXT,

    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_insights_tenant ON liquid_mirror.insights(tenant_id);
CREATE INDEX idx_insights_severity ON liquid_mirror.insights(tenant_id, severity);
CREATE INDEX idx_insights_type ON liquid_mirror.insights(tenant_id, insight_type);
CREATE INDEX idx_insights_unacked ON liquid_mirror.insights(tenant_id, created_at DESC)
    WHERE acknowledged_at IS NULL;
CREATE INDEX idx_insights_critical ON liquid_mirror.insights(tenant_id, created_at DESC)
    WHERE severity = 'critical' AND acknowledged_at IS NULL;

COMMENT ON TABLE liquid_mirror.insights IS 'System-generated insights and recommendations from MetacognitiveMirror';

-- =============================================================================
-- MEMORY THERMODYNAMICS (temperature tracking)
-- =============================================================================

CREATE TABLE liquid_mirror.memory_temperatures (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    memory_id VARCHAR(255) NOT NULL, -- External memory reference

    -- Temperature metrics
    temperature DOUBLE PRECISION NOT NULL, -- 0-1 scale
    access_count INTEGER DEFAULT 0,
    last_access_at TIMESTAMPTZ,

    -- Burst detection
    burst_intensity DOUBLE PRECISION, -- NULL if no burst
    burst_detected_at TIMESTAMPTZ,

    -- Community clustering
    community_id INTEGER,
    co_accessed_memories TEXT[], -- Top co-accessed memory IDs

    snapshot_time TIMESTAMPTZ DEFAULT now(),

    UNIQUE(tenant_id, memory_id, snapshot_time)
);

CREATE INDEX idx_memory_temp_tenant ON liquid_mirror.memory_temperatures(tenant_id);
CREATE INDEX idx_memory_temp_hot ON liquid_mirror.memory_temperatures(tenant_id, temperature DESC);
CREATE INDEX idx_memory_temp_burst ON liquid_mirror.memory_temperatures(tenant_id, burst_intensity DESC NULLS LAST)
    WHERE burst_intensity IS NOT NULL;
CREATE INDEX idx_memory_temp_time ON liquid_mirror.memory_temperatures(tenant_id, snapshot_time DESC);

COMMENT ON TABLE liquid_mirror.memory_temperatures IS 'Memory access temperature and burst tracking';

-- =============================================================================
-- COGNITIVE SNAPSHOTS (system state over time)
-- =============================================================================

CREATE TABLE liquid_mirror.cognitive_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,

    -- Phase and state
    phase VARCHAR(20) NOT NULL, -- exploration/exploitation/learning/consolidation/idle/crisis
    phase_confidence DOUBLE PRECISION,

    -- Activity metrics
    temperature DOUBLE PRECISION, -- Overall activity level 0-1
    focus_score DOUBLE PRECISION, -- How concentrated attention is 0-1
    stability_score DOUBLE PRECISION, -- Phase stability 0-1

    -- Entropy metrics
    access_entropy DOUBLE PRECISION,
    query_entropy DOUBLE PRECISION,

    -- Drift metrics
    drift_magnitude DOUBLE PRECISION,
    drift_signal VARCHAR(30),

    -- Topics
    dominant_topics JSONB DEFAULT '[]', -- [{topic, strength}, ...]
    query_count_window INTEGER, -- Queries in analysis window

    -- Context
    user_email VARCHAR(255), -- If snapshot is user-specific
    session_id VARCHAR(255), -- If snapshot is session-specific

    snapshot_time TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_cognitive_tenant ON liquid_mirror.cognitive_snapshots(tenant_id);
CREATE INDEX idx_cognitive_time ON liquid_mirror.cognitive_snapshots(tenant_id, snapshot_time DESC);
CREATE INDEX idx_cognitive_phase ON liquid_mirror.cognitive_snapshots(tenant_id, phase);
CREATE INDEX idx_cognitive_crisis ON liquid_mirror.cognitive_snapshots(tenant_id, snapshot_time DESC)
    WHERE phase = 'crisis';
CREATE INDEX idx_cognitive_user ON liquid_mirror.cognitive_snapshots(tenant_id, user_email, snapshot_time DESC)
    WHERE user_email IS NOT NULL;

COMMENT ON TABLE liquid_mirror.cognitive_snapshots IS 'Point-in-time cognitive state captures';

-- =============================================================================
-- PREDICTIONS (Markov chain prefetch predictions)
-- =============================================================================

CREATE TABLE liquid_mirror.predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,

    -- Prediction context
    user_email VARCHAR(255),
    session_id VARCHAR(255),
    context_memory_ids TEXT[], -- What was accessed before prediction

    -- Predictions
    predicted_memory_ids TEXT[] NOT NULL,
    confidence_scores DOUBLE PRECISION[] NOT NULL,

    -- Validation (filled in after actual access)
    actual_accessed_ids TEXT[],
    accuracy DOUBLE PRECISION,
    validated_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_predictions_tenant ON liquid_mirror.predictions(tenant_id);
CREATE INDEX idx_predictions_time ON liquid_mirror.predictions(tenant_id, created_at DESC);
CREATE INDEX idx_predictions_unvalidated ON liquid_mirror.predictions(tenant_id, created_at)
    WHERE validated_at IS NULL;

COMMENT ON TABLE liquid_mirror.predictions IS 'Predictive prefetch predictions with validation tracking';

-- =============================================================================
-- ROW LEVEL SECURITY
-- =============================================================================

ALTER TABLE liquid_mirror.query_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE liquid_mirror.session_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE liquid_mirror.insights ENABLE ROW LEVEL SECURITY;
ALTER TABLE liquid_mirror.memory_temperatures ENABLE ROW LEVEL SECURITY;
ALTER TABLE liquid_mirror.cognitive_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE liquid_mirror.predictions ENABLE ROW LEVEL SECURITY;

-- Policies use current_setting('app.current_tenant') set by application
CREATE POLICY tenant_isolation ON liquid_mirror.query_events
    FOR ALL USING (tenant_id = current_setting('app.current_tenant', true)::uuid);

CREATE POLICY tenant_isolation ON liquid_mirror.session_events
    FOR ALL USING (tenant_id = current_setting('app.current_tenant', true)::uuid);

CREATE POLICY tenant_isolation ON liquid_mirror.insights
    FOR ALL USING (tenant_id = current_setting('app.current_tenant', true)::uuid);

CREATE POLICY tenant_isolation ON liquid_mirror.memory_temperatures
    FOR ALL USING (tenant_id = current_setting('app.current_tenant', true)::uuid);

CREATE POLICY tenant_isolation ON liquid_mirror.cognitive_snapshots
    FOR ALL USING (tenant_id = current_setting('app.current_tenant', true)::uuid);

CREATE POLICY tenant_isolation ON liquid_mirror.predictions
    FOR ALL USING (tenant_id = current_setting('app.current_tenant', true)::uuid);

-- =============================================================================
-- FUNCTIONS
-- =============================================================================

-- Function to set tenant context (call at start of each request)
CREATE OR REPLACE FUNCTION liquid_mirror.set_tenant_context(p_tenant_id UUID)
RETURNS VOID AS $$
BEGIN
    PERFORM set_config('app.current_tenant', p_tenant_id::text, false);
END;
$$ LANGUAGE plpgsql;

-- Function to get current tenant
CREATE OR REPLACE FUNCTION liquid_mirror.get_current_tenant()
RETURNS UUID AS $$
BEGIN
    RETURN current_setting('app.current_tenant', true)::uuid;
END;
$$ LANGUAGE plpgsql;

-- Function to generate API key (returns key, stores hash)
CREATE OR REPLACE FUNCTION liquid_mirror.create_api_key(
    p_tenant_id UUID,
    p_tenant_source VARCHAR DEFAULT 'enterprise',
    p_name VARCHAR DEFAULT NULL,
    p_scopes JSONB DEFAULT '["ingest", "read"]'
)
RETURNS TABLE(api_key TEXT, key_id UUID) AS $$
DECLARE
    v_key TEXT;
    v_prefix VARCHAR(8);
    v_hash VARCHAR(64);
    v_id UUID;
BEGIN
    -- Generate random key: lm_live_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
    v_key := 'lm_live_' || encode(gen_random_bytes(24), 'hex');
    v_prefix := substring(v_key, 1, 8);
    v_hash := encode(sha256(v_key::bytea), 'hex');

    INSERT INTO liquid_mirror.api_keys (tenant_id, tenant_source, key_prefix, key_hash, name, scopes)
    VALUES (p_tenant_id, p_tenant_source, v_prefix, v_hash, p_name, p_scopes)
    RETURNING id INTO v_id;

    RETURN QUERY SELECT v_key, v_id;
END;
$$ LANGUAGE plpgsql;

-- Function to validate API key and return tenant info
CREATE OR REPLACE FUNCTION liquid_mirror.validate_api_key(p_key TEXT)
RETURNS TABLE(
    tenant_id UUID,
    tenant_source VARCHAR,
    scopes JSONB,
    is_valid BOOLEAN
) AS $$
DECLARE
    v_hash VARCHAR(64);
BEGIN
    v_hash := encode(sha256(p_key::bytea), 'hex');

    RETURN QUERY
    SELECT
        ak.tenant_id,
        ak.tenant_source,
        ak.scopes,
        (ak.is_active AND (ak.expires_at IS NULL OR ak.expires_at > now())) as is_valid
    FROM liquid_mirror.api_keys ak
    WHERE ak.key_hash = v_hash;

    -- Update last_used_at
    UPDATE liquid_mirror.api_keys
    SET last_used_at = now()
    WHERE key_hash = v_hash;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- SEED DATA: enterprise_bot as first tenant
-- =============================================================================

-- Get enterprise_bot tenant ID from enterprise.tenants (Driscoll Foods)
-- If it doesn't exist, we'll handle in application layer
DO $$
DECLARE
    v_tenant_id UUID;
    v_api_key TEXT;
    v_key_id UUID;
BEGIN
    -- Try to get existing enterprise tenant (driscoll)
    SELECT id INTO v_tenant_id
    FROM enterprise.tenants
    WHERE slug = 'driscoll'
    LIMIT 1;

    IF v_tenant_id IS NOT NULL THEN
        -- Create API key for enterprise_bot
        SELECT * INTO v_api_key, v_key_id
        FROM liquid_mirror.create_api_key(
            v_tenant_id,
            'enterprise',
            'enterprise_bot_primary',
            '["ingest", "read", "admin"]'
        );

        RAISE NOTICE 'Created API key for enterprise_bot (driscoll): %', v_api_key;
        RAISE NOTICE 'Key ID: %', v_key_id;
        RAISE NOTICE '*** SAVE THIS KEY - IT WILL NOT BE SHOWN AGAIN ***';
    ELSE
        RAISE NOTICE 'No enterprise tenant found. Create API key manually after migration.';
    END IF;
END $$;

-- =============================================================================
-- ROLLBACK
-- =============================================================================
/*
-- To rollback this migration:

DROP FUNCTION IF EXISTS liquid_mirror.validate_api_key(TEXT);
DROP FUNCTION IF EXISTS liquid_mirror.create_api_key(UUID, VARCHAR, VARCHAR, JSONB);
DROP FUNCTION IF EXISTS liquid_mirror.get_current_tenant();
DROP FUNCTION IF EXISTS liquid_mirror.set_tenant_context(UUID);

DROP TABLE IF EXISTS liquid_mirror.predictions;
DROP TABLE IF EXISTS liquid_mirror.cognitive_snapshots;
DROP TABLE IF EXISTS liquid_mirror.memory_temperatures;
DROP TABLE IF EXISTS liquid_mirror.insights;
DROP TABLE IF EXISTS liquid_mirror.session_events;
DROP TABLE IF EXISTS liquid_mirror.query_events;
DROP TABLE IF EXISTS liquid_mirror.api_keys;
DROP TABLE IF EXISTS liquid_mirror.tenants;

DROP SCHEMA IF EXISTS liquid_mirror;
*/
