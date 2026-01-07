<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    import { browser } from '$app/environment';
    import StatCard from '$lib/components/charts/StatCard.svelte';
    import LineChart from '$lib/components/charts/LineChart.svelte';
    import BarChart from '$lib/components/charts/BarChart.svelte';
    import DoughnutChart from '$lib/components/charts/DoughnutChart.svelte';
    import DateRangePicker from '$lib/components/charts/DateRangePicker.svelte';

    function getApiBase(): string {
        return import.meta.env.VITE_API_URL || 'http://localhost:8000';
    }

    // Dashboard state
    let loading = true;
    let error: string | null = null;
    let hours = 24;
    let selectedDepartment: string | null = null;
    
    // Dashboard data
    let summary = {
        total_queries: 0,
        refinement_rate: 0,
        avg_response_time_ms: 0,
        llm_cost_usd: 0,
        unique_users: 0
    };
    let refinementTrend: any[] = [];
    let departmentBreakdown: any[] = [];
    let knowledgeGaps: any[] = [];
    let recentQueries: any[] = [];
    let alertsFiring = 0;
    
    // Export menu
    let showExportMenu = false;
    let exporting = false;
    
    async function loadDashboard() {
        loading = true;
        error = null;
        
        try {
            const params = new URLSearchParams({ hours: hours.toString() });
            if (selectedDepartment) {
                params.set('department', selectedDepartment);
            }
            
            const res = await fetch(`${getApiBase()}/api/mirror/dashboard?${params}`);
            if (!res.ok) throw new Error('Failed to load dashboard');
            
            const data = await res.json();
            
            summary = data.summary;
            refinementTrend = data.refinement_trend;
            departmentBreakdown = data.department_breakdown;
            knowledgeGaps = data.knowledge_gaps;
            recentQueries = data.recent_queries;
            alertsFiring = data.alerts_firing;
            
        } catch (e) {
            error = e instanceof Error ? e.message : 'Unknown error';
        } finally {
            loading = false;
        }
    }
    
    async function exportDashboard(format: string) {
        exporting = true;
        showExportMenu = false;
        
        try {
            const res = await fetch(`${getApiBase()}/api/mirror/export`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ format, hours, department: selectedDepartment })
            });
            
            if (!res.ok) throw new Error('Export failed');
            
            // Handle file download
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `liquid-mirror-${new Date().toISOString().slice(0, 10)}.${format}`;
            a.click();
            URL.revokeObjectURL(url);
            
        } catch (e) {
            console.error('Export failed:', e);
        } finally {
            exporting = false;
        }
    }
    
    // Auto-refresh
    let refreshInterval: ReturnType<typeof setInterval>;
    
    onMount(() => {
        loadDashboard();
        refreshInterval = setInterval(loadDashboard, 60000); // Refresh every minute
    });
    
    onDestroy(() => {
        if (refreshInterval) clearInterval(refreshInterval);
    });
    
    // Reactive reload on filter change
    $: if (browser && (hours || selectedDepartment !== undefined)) {
        loadDashboard();
    }
    
    // Chart data transforms
    $: trendChartData = {
        labels: refinementTrend.map(r => new Date(r.hour).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })),
        datasets: [
            {
                label: 'Queries',
                data: refinementTrend.map(r => r.queries),
                borderColor: '#ffc800',
                backgroundColor: 'rgba(255, 200, 0, 0.1)',
                fill: true
            },
            {
                label: 'Refinements',
                data: refinementTrend.map(r => r.refinements),
                borderColor: '#ff6400',
                backgroundColor: 'rgba(255, 100, 0, 0.1)',
                fill: true
            }
        ]
    };
    
    $: deptChartData = {
        labels: departmentBreakdown.map(d => d.department),
        datasets: [{
            data: departmentBreakdown.map(d => d.queries),
            backgroundColor: [
                '#ffc800', '#00ff41', '#ff6400', '#00d4ff', '#ff0055',
                '#aa00ff', '#ffaa00', '#00ffaa'
            ]
        }]
    };
</script>

<svelte:head>
    <title>Liquid Mirror Analytics</title>
</svelte:head>

<div class="dashboard">
    <!-- Header -->
    <header class="dashboard-header">
        <div class="header-left">
            <h1>🪞 Liquid Mirror Analytics</h1>
            <p class="subtitle">Real-time intelligence dashboard</p>
        </div>
        <div class="header-right">
            <DateRangePicker bind:hours on:change={() => loadDashboard()} />
            
            <div class="export-container">
                <button 
                    class="export-btn"
                    class:loading={exporting}
                    on:click={() => showExportMenu = !showExportMenu}
                >
                    {#if exporting}
                        <span class="spinner"></span>
                    {:else}
                        ⬇ Export
                    {/if}
                </button>
                
                {#if showExportMenu}
                    <div class="export-menu">
                        <button on:click={() => exportDashboard('html')}>
                            🌐 HTML Snapshot
                        </button>
                        <button on:click={() => exportDashboard('xlsx')} disabled>
                            📊 Excel Report
                        </button>
                        <button on:click={() => exportDashboard('csv')} disabled>
                            📋 CSV Data
                        </button>
                        <button on:click={() => exportDashboard('pdf')} disabled>
                            📄 PDF Summary
                        </button>
                    </div>
                {/if}
            </div>
            
            {#if alertsFiring > 0}
                <div class="alerts-badge">
                    🔔 {alertsFiring} alert{alertsFiring > 1 ? 's' : ''}
                </div>
            {/if}
        </div>
    </header>
    
    {#if loading && !summary.total_queries}
        <div class="loading-state">
            <div class="spinner-large"></div>
            <p>Loading analytics...</p>
        </div>
    {:else if error}
        <div class="error-state">
            <p>⚠️ {error}</p>
            <button on:click={loadDashboard}>Retry</button>
        </div>
    {:else}
        <!-- Summary Stats -->
        <section class="stats-grid">
            <StatCard
                label="Total Queries"
                value={summary.total_queries.toLocaleString()}
                trend={12}
                icon="📊"
            />
            <StatCard
                label="Refinement Rate"
                value={`${summary.refinement_rate}%`}
                trend={summary.refinement_rate > 15 ? -5 : 3}
                icon="🔄"
                warning={summary.refinement_rate > 15}
            />
            <StatCard
                label="Avg Response"
                value={`${summary.avg_response_time_ms}ms`}
                trend={-8}
                icon="⚡"
            />
            <StatCard
                label="LLM Cost"
                value={`$${summary.llm_cost_usd.toFixed(2)}`}
                trend={5}
                icon="💰"
            />
        </section>
        
        <!-- Charts Row -->
        <section class="charts-row">
            <div class="chart-card">
                <h3>Query Volume & Refinements</h3>
                <LineChart data={trendChartData} height={250} />
            </div>
            <div class="chart-card">
                <h3>Department Breakdown</h3>
                <DoughnutChart data={deptChartData} height={250} />
            </div>
        </section>
        
        <!-- Knowledge Gaps -->
        <section class="knowledge-gaps">
            <h3>🕳️ Knowledge Gaps <span class="hint">(Topics where users struggle)</span></h3>
            
            {#if knowledgeGaps.length === 0}
                <p class="empty">No significant knowledge gaps detected. Great job! 🎉</p>
            {:else}
                <table>
                    <thead>
                        <tr>
                            <th>Query Pattern</th>
                            <th>Refinements</th>
                            <th>Departments</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {#each knowledgeGaps as gap}
                            <tr>
                                <td class="query-cell">{gap.query}</td>
                                <td>
                                    <span class="badge warning">{gap.refinement_count}x</span>
                                </td>
                                <td>{gap.departments.join(', ')}</td>
                                <td>
                                    <button class="action-btn">View</button>
                                </td>
                            </tr>
                        {/each}
                    </tbody>
                </table>
                <p class="gap-hint">📌 These topics need better documentation</p>
            {/if}
        </section>
        
        <!-- Recent Queries -->
        <section class="recent-queries">
            <div class="section-header">
                <h3>📜 Recent Queries</h3>
                <a href="/admin/liquid-mirror/queries" class="view-all">View All →</a>
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>User</th>
                        <th>Dept</th>
                        <th>Query</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {#each recentQueries.slice(0, 10) as query}
                        <tr>
                            <td class="time-cell">
                                {new Date(query.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                            </td>
                            <td>{query.user_email.split('@')[0]}</td>
                            <td>{query.department || '-'}</td>
                            <td class="query-cell">
                                {query.query?.slice(0, 50)}{query.query?.length > 50 ? '...' : ''}
                            </td>
                            <td class="status-cell">
                                {#if query.was_refined}
                                    <span class="badge warning" title="User rephrased">⚠️</span>
                                {/if}
                                {#if query.was_copied}
                                    <span class="badge success" title="Response copied">📋</span>
                                {/if}
                                {#if query.rating === 'up'}
                                    <span class="badge success">👍</span>
                                {:else if query.rating === 'down'}
                                    <span class="badge error">👎</span>
                                {/if}
                                {#if !query.was_refined && !query.was_copied && !query.rating}
                                    <span class="badge neutral">✓</span>
                                {/if}
                            </td>
                        </tr>
                    {/each}
                </tbody>
            </table>
        </section>
    {/if}
</div>

<style>
    .dashboard {
        max-width: 1400px;
        margin: 0 auto;
    }
    
    .dashboard-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(255, 200, 0, 0.2);
    }
    
    .dashboard-header h1 {
        font-size: 1.5rem;
        color: #ffc800;
        margin: 0;
    }
    
    .subtitle {
        color: #888;
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }
    
    .header-right {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .export-container {
        position: relative;
    }
    
    .export-btn {
        padding: 0.5rem 1rem;
        background: rgba(255, 200, 0, 0.1);
        border: 1px solid rgba(255, 200, 0, 0.3);
        border-radius: 6px;
        color: #ffc800;
        cursor: pointer;
        font-size: 0.9rem;
        transition: all 0.2s;
    }
    
    .export-btn:hover {
        background: rgba(255, 200, 0, 0.2);
    }
    
    .export-menu {
        position: absolute;
        top: 100%;
        right: 0;
        margin-top: 0.5rem;
        background: #1a1a20;
        border: 1px solid rgba(255, 200, 0, 0.2);
        border-radius: 8px;
        padding: 0.5rem;
        min-width: 180px;
        z-index: 100;
    }
    
    .export-menu button {
        display: block;
        width: 100%;
        padding: 0.5rem 0.75rem;
        background: none;
        border: none;
        color: #e0e0e0;
        text-align: left;
        cursor: pointer;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    
    .export-menu button:hover:not(:disabled) {
        background: rgba(255, 200, 0, 0.1);
    }
    
    .export-menu button:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
    
    .alerts-badge {
        padding: 0.5rem 0.75rem;
        background: rgba(255, 0, 85, 0.2);
        border: 1px solid rgba(255, 0, 85, 0.4);
        border-radius: 6px;
        color: #ff0055;
        font-size: 0.85rem;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .charts-row {
        display: grid;
        grid-template-columns: 2fr 1fr;
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .chart-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 200, 0, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
    }
    
    .chart-card h3 {
        font-size: 1rem;
        color: #ffc800;
        margin: 0 0 1rem 0;
    }
    
    .knowledge-gaps, .recent-queries {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 200, 0, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .knowledge-gaps h3, .recent-queries h3 {
        font-size: 1rem;
        color: #ffc800;
        margin: 0 0 1rem 0;
    }
    
    .hint {
        color: #888;
        font-weight: normal;
        font-size: 0.85rem;
    }
    
    .section-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .view-all {
        color: #ffc800;
        font-size: 0.85rem;
        text-decoration: none;
    }
    
    .view-all:hover {
        text-decoration: underline;
    }
    
    table {
        width: 100%;
        border-collapse: collapse;
    }
    
    th, td {
        padding: 0.75rem;
        text-align: left;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    th {
        color: #888;
        font-weight: 500;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    td {
        color: #e0e0e0;
        font-size: 0.9rem;
    }
    
    .query-cell {
        max-width: 300px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    
    .time-cell {
        color: #888;
        font-size: 0.85rem;
    }
    
    .status-cell {
        display: flex;
        gap: 0.25rem;
    }
    
    .badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
    }
    
    .badge.warning {
        background: rgba(255, 100, 0, 0.2);
        color: #ff6400;
    }
    
    .badge.success {
        background: rgba(0, 255, 65, 0.2);
        color: #00ff41;
    }
    
    .badge.error {
        background: rgba(255, 0, 85, 0.2);
        color: #ff0055;
    }
    
    .badge.neutral {
        background: rgba(255, 255, 255, 0.1);
        color: #888;
    }
    
    .action-btn {
        padding: 0.25rem 0.5rem;
        background: rgba(255, 200, 0, 0.1);
        border: 1px solid rgba(255, 200, 0, 0.3);
        border-radius: 4px;
        color: #ffc800;
        font-size: 0.8rem;
        cursor: pointer;
    }
    
    .gap-hint {
        margin-top: 1rem;
        color: #888;
        font-size: 0.85rem;
        font-style: italic;
    }
    
    .empty {
        color: #888;
        text-align: center;
        padding: 2rem;
    }
    
    .loading-state, .error-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 4rem;
        color: #888;
    }
    
    .spinner-large {
        width: 48px;
        height: 48px;
        border: 3px solid rgba(255, 200, 0, 0.2);
        border-top-color: #ffc800;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
        margin-bottom: 1rem;
    }
    
    .spinner {
        width: 16px;
        height: 16px;
        border: 2px solid rgba(255, 200, 0, 0.2);
        border-top-color: #ffc800;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    @media (max-width: 1200px) {
        .stats-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .charts-row {
            grid-template-columns: 1fr;
        }
    }
    
    @media (max-width: 768px) {
        .dashboard-header {
            flex-direction: column;
            gap: 1rem;
        }
        
        .header-right {
            width: 100%;
            justify-content: space-between;
        }
        
        .stats-grid {
            grid-template-columns: 1fr;
        }
    }
</style>