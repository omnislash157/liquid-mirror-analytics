# Liquid Mirror Analytics - Session Handoff

**Date:** 2026-01-09
**From:** Claude Opus 4.5 (enterprise_bot SDK session)
**Goal:** Multi-tenant analytics platform with predictive cognitive observability

---

## TL;DR

Liquid Mirror is not Datadog. It's a **cognitive AI observability platform** that tells you what's about to go wrong before it happens. The MetacognitiveMirror component is the moat - nothing else in the market does semantic drift detection, cognitive phase classification, or self-optimization recommendations.

**Status:** IMPLEMENTED - Schema, ingestion API, and dashboard endpoints are ready. Run migration and deploy.

---

## COMPLETED THIS SESSION

### 1. SQL Migration Created
**File:** `migrations/001_liquid_mirror_schema.sql`
- Full `liquid_mirror` schema with 7 tables
- RLS policies for tenant isolation
- API key auth with SHA-256 hashing
- Auto-generates API key for enterprise_bot (driscoll tenant)
- Rollback SQL included at bottom

### 2. Ingestion API Implemented
**File:** `src/liquid_mirror/ingest.py`

Endpoints:
- `POST /api/v1/ingest/query` - Log query events with cognitive analysis
- `POST /api/v1/ingest/session` - Log session events (login/logout/error)
- `POST /api/v1/ingest/batch` - Bulk ingestion
- `GET /api/v1/ingest/status` - Test API key validity

Auth: `X-API-Key` header with `lm_live_xxx` format keys

### 3. Dashboard Insights API Added
**File:** `src/liquid_mirror/routes.py` (new endpoints)

- `GET /api/analytics/insights/recent` - Recent insights with severity filter
- `GET /api/analytics/insights/critical` - Unacknowledged critical alerts
- `POST /api/analytics/insights/{id}/acknowledge` - Mark insight handled
- `GET /api/analytics/cognitive/snapshots` - Cognitive state over time
- `GET /api/analytics/cognitive/current` - Real-time cognitive phase

### 4. Main App Updated
**File:** `src/liquid_mirror/main.py`
- Version bumped to 0.3.0
- Ingest router mounted at `/api/v1/ingest`
- All endpoints documented in root `/`

---

## TO DEPLOY

```bash
# 1. Run the migration
psql $AZURE_PG_CONNECTION_STRING -f migrations/001_liquid_mirror_schema.sql

# 2. Note the API key printed (SAVE IT!)
# Output: Created API key for enterprise_bot (driscoll): lm_live_xxx...

# 3. Set the API key in enterprise_bot's env
# LIQUID_MIRROR_API_KEY=lm_live_xxx...

# 4. Start the service
uvicorn liquid_mirror.main:app --host 0.0.0.0 --port 8001
```

---

## What Liquid Mirror Has That Nobody Else Does

### 1. QueryArchaeologist
**Location:** `src/liquid_mirror/mirror/metacognitive_mirror.py:179`

Reconstructs cognitive intent from query patterns:
- Clusters queries by semantic similarity (embedding distance)
- Detects recurring patterns with frequency + recency scoring
- Calculates query entropy (high = exploration, low = exploitation)
- **Detects semantic drift** - topic shifts, vocabulary expansion/collapse

```python
drift_mag, drift_signal = archaeologist.detect_semantic_drift()
# Returns: DriftSignal.SEMANTIC_COLLAPSE  <- "User is stuck in a loop"
```

### 2. MemoryThermodynamics
**Location:** `src/liquid_mirror/mirror/metacognitive_mirror.py:356`

Tracks "temperature" of data access:
- Hot memories (frequently/recently accessed) vs cold memories
- **Burst detection** - sudden spikes in access indicate emergent importance
- Co-access graph - memories accessed together form communities
- Access entropy - how evenly distributed is attention

```python
bursts = thermodynamics.detect_bursts(time_window_hours=1.0)
# Returns: [("memory_123", 5.2)]  <- "This memory is 5.2x hotter than baseline"
```

### 3. CognitiveSeismograph
**Location:** `src/liquid_mirror/mirror/metacognitive_mirror.py:556`

Classifies cognitive phase from access patterns:
- **EXPLORATION** - Wide-ranging queries, high entropy
- **EXPLOITATION** - Focused queries, low entropy
- **LEARNING** - Building new structures
- **CONSOLIDATION** - Reviewing/connecting existing
- **IDLE** - Low activity
- **CRISIS** - Rapid, unfocused access. User is lost/confused.

```python
phase = seismograph._classify_phase(...)
# If phase == CognitivePhase.CRISIS: "ALERT: User appears lost or frustrated"
```

### 4. PredictivePrefetcher
**Location:** `src/liquid_mirror/mirror/metacognitive_mirror.py:745`

Markov chain predictions of future access:
- Learns transition probabilities from access sequences
- Predicts what memories will be needed next
- **Validates its own predictions** - tracks accuracy over time
- If accuracy > 60%: enable aggressive preloading

```python
prediction = prefetcher.predict_next_memories(current_sequence)
# Returns: [("memory_456", 0.72)]  <- "72% confident this is needed next"
```

### 5. ArchitecturalIntrospector
**Location:** `src/liquid_mirror/mirror/metacognitive_mirror.py:919`

The system examining itself:
- Detects slow query patterns (>2 sigma)
- Identifies memory concentration (top 10 memories = 50% traffic)
- Flags cognitive instability (rapid phase transitions)
- **Generates actionable recommendations**

```python
insights = introspector.analyze_system_health(...)
# Returns: MetacognitiveInsight(
#     severity="critical",
#     description="Semantic collapse detected",
#     suggested_action="URGENT: Investigate user workflow"
# )
```

---

## The Differentiator (Pitch Version)

**Datadog:** "Your p99 is 250ms" (you figure out why)
**Liquid Mirror:** "Your user is confused. Semantic collapse detected. They've asked the same question 5 ways. Here's the memory they need."

**Grafana:** Pretty dashboard you stare at
**Liquid Mirror:** "Cache these 5 memories and reduce latency 40%. I'm 72% confident they'll need memory_456 next. The system is in LEARNING phase - optimize for breadth not depth."

---

## Architecture Decision: Option C (Both Ingestion Paths)

### Path 1: OTEL Native (for any OTEL-instrumented app)
```
Any App -> OTEL Collector -> Liquid Mirror OTLP Receiver -> PostgreSQL + RLS
```
- Accept `X-Scope-OrgID` header for tenant routing
- Standard OTLP gRPC (:4317) and HTTP (:4318)
- Works with anything already using OpenTelemetry

### Path 2: Custom API (for enterprise_bot and legacy)
```
enterprise_bot -> POST /api/v1/ingest (API Key) -> Liquid Mirror -> PostgreSQL
```
- API key maps to tenant_id
- Custom JSON payload (current format)
- enterprise_bot is first tenant

---

## Proposed Schema: `liquid_mirror`

```sql
-- Use existing enterprise.tenants.id when available
-- Create new tenants in liquid_mirror.tenants when needed

CREATE SCHEMA liquid_mirror;

-- Tenants (for standalone customers, not enterprise_bot)
CREATE TABLE liquid_mirror.tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug VARCHAR NOT NULL UNIQUE,
    name VARCHAR NOT NULL,
    domain VARCHAR UNIQUE,
    created_at TIMESTAMPTZ DEFAULT now(),
    is_active BOOLEAN DEFAULT true
);

-- API Keys for ingestion auth
CREATE TABLE liquid_mirror.api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,  -- References enterprise.tenants OR liquid_mirror.tenants
    key_hash VARCHAR NOT NULL UNIQUE,
    name VARCHAR,
    scopes JSONB DEFAULT '["ingest"]',
    created_at TIMESTAMPTZ DEFAULT now(),
    last_used_at TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT true
);

-- Query events (multi-tenant with RLS)
CREATE TABLE liquid_mirror.query_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    user_email VARCHAR,
    session_id VARCHAR,
    query_text TEXT NOT NULL,
    query_embedding VECTOR(1024),
    department VARCHAR,
    response_time_ms DOUBLE PRECISION,
    tokens_input INTEGER,
    tokens_output INTEGER,
    model_used VARCHAR,
    -- Heuristics from existing analytics
    complexity_score DOUBLE PRECISION,
    intent_type VARCHAR,
    specificity_score DOUBLE PRECISION,
    -- MetacognitiveMirror fields
    cognitive_phase VARCHAR,  -- exploration/exploitation/learning/etc
    semantic_drift_magnitude DOUBLE PRECISION,
    drift_signal VARCHAR,
    query_cluster_id INTEGER,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Session events
CREATE TABLE liquid_mirror.session_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    event_type VARCHAR NOT NULL,
    user_email VARCHAR,
    session_id VARCHAR,
    event_data JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Mirror insights (the good stuff)
CREATE TABLE liquid_mirror.insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    insight_type VARCHAR NOT NULL,
    severity VARCHAR NOT NULL,  -- info/warning/critical
    description TEXT NOT NULL,
    metrics JSONB,
    suggested_action TEXT,
    estimated_impact TEXT,
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by VARCHAR,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Memory thermodynamics snapshots
CREATE TABLE liquid_mirror.memory_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    memory_id VARCHAR NOT NULL,
    temperature DOUBLE PRECISION,
    access_count INTEGER,
    burst_intensity DOUBLE PRECISION,
    community_id INTEGER,
    snapshot_time TIMESTAMPTZ DEFAULT now()
);

-- Cognitive state snapshots
CREATE TABLE liquid_mirror.cognitive_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    phase VARCHAR NOT NULL,
    temperature DOUBLE PRECISION,
    focus_score DOUBLE PRECISION,
    access_entropy DOUBLE PRECISION,
    query_entropy DOUBLE PRECISION,
    drift_magnitude DOUBLE PRECISION,
    stability_score DOUBLE PRECISION,
    dominant_topics JSONB,
    snapshot_time TIMESTAMPTZ DEFAULT now()
);

-- RLS Policies
ALTER TABLE liquid_mirror.query_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE liquid_mirror.session_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE liquid_mirror.insights ENABLE ROW LEVEL SECURITY;
ALTER TABLE liquid_mirror.memory_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE liquid_mirror.cognitive_snapshots ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation_query ON liquid_mirror.query_events
    USING (tenant_id = current_setting('app.current_tenant')::uuid);

CREATE POLICY tenant_isolation_session ON liquid_mirror.session_events
    USING (tenant_id = current_setting('app.current_tenant')::uuid);

CREATE POLICY tenant_isolation_insights ON liquid_mirror.insights
    USING (tenant_id = current_setting('app.current_tenant')::uuid);

CREATE POLICY tenant_isolation_memory ON liquid_mirror.memory_snapshots
    USING (tenant_id = current_setting('app.current_tenant')::uuid);

CREATE POLICY tenant_isolation_cognitive ON liquid_mirror.cognitive_snapshots
    USING (tenant_id = current_setting('app.current_tenant')::uuid);

-- Indexes
CREATE INDEX idx_query_events_tenant ON liquid_mirror.query_events(tenant_id);
CREATE INDEX idx_query_events_created ON liquid_mirror.query_events(created_at);
CREATE INDEX idx_query_events_session ON liquid_mirror.query_events(tenant_id, session_id);
CREATE INDEX idx_insights_tenant_severity ON liquid_mirror.insights(tenant_id, severity);
CREATE INDEX idx_insights_unacked ON liquid_mirror.insights(tenant_id) WHERE acknowledged_at IS NULL;
CREATE INDEX idx_cognitive_tenant_time ON liquid_mirror.cognitive_snapshots(tenant_id, snapshot_time);
```

---

## Next Session Tasks

### Phase 1: Schema + Migration
1. Review and approve SQL schema above
2. Create migration with rollback
3. Add enterprise_bot as first tenant (link to existing `enterprise.tenants.id`)
4. Generate API key for enterprise_bot

### Phase 2: Ingestion Pipeline
1. Create `/api/v1/ingest` endpoint with API key auth
2. Wire MetacognitiveMirror to process incoming events
3. Store insights to `liquid_mirror.insights` table
4. Background job for cognitive snapshot capture

### Phase 3: OTEL Integration (Optional)
1. Add OTLP receiver (grpc + http)
2. Map `X-Scope-OrgID` header to tenant_id
3. Transform OTEL spans/metrics to Liquid Mirror schema

### Phase 4: Dashboard API
1. GET `/api/v1/insights` - Current insights by severity
2. GET `/api/v1/cognitive-state` - Real-time cognitive phase
3. GET `/api/v1/predictions` - What will user need next
4. WebSocket `/ws/live` - Real-time insight stream

---

## Key Files

| File | Purpose |
|------|---------|
| `src/liquid_mirror/mirror/metacognitive_mirror.py` | The brain - 1441 lines of cognitive analysis |
| `src/liquid_mirror/service.py` | AnalyticsService - query/event logging |
| `src/liquid_mirror/heuristics.py` | Query complexity/intent analysis |
| `src/liquid_mirror/routes.py` | FastAPI routes (dashboard API) |
| `src/liquid_mirror/events/bus.py` | Pub/sub event bus |

---

## Questions for Next Session

1. **OTEL Priority:** Do we want OTEL ingestion in Phase 1, or is custom API enough to start?

2. **Tenant ID Strategy:** Use existing `enterprise.tenants.id` when available, or always create in `liquid_mirror.tenants`?

3. **Real-time vs Batch:** Should MetacognitiveMirror run on every query (real-time) or batch process every N minutes?

4. **Alert Delivery:** When we detect CRISIS or SEMANTIC_COLLAPSE, how do we notify? Webhook? Email? In-app?

---

## The Vision

Liquid Mirror is the **self-aware AI observability platform**. It doesn't just show you metrics - it tells you:

- What cognitive phase your users are in
- When they're about to get frustrated (semantic collapse)
- What data they'll need next (Markov prediction)
- How to optimize the system (architectural introspection)
- What's going wrong before it fails

This is observability that **thinks**. The mirror reflects not just what happened, but what it means and what to do about it.

---

**Written by Claude Opus 4.5 after reading the full MetacognitiveMirror implementation.**

*"The system watching itself think, so you don't have to."*
