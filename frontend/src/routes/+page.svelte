<script lang="ts">
	import { analyticsStore } from '$lib/stores/analytics';
	import StatCard from '$lib/components/charts/StatCard.svelte';
	import LineChart from '$lib/components/charts/LineChart.svelte';
	import DoughnutChart from '$lib/components/charts/DoughnutChart.svelte';

	$: overview = $analyticsStore.overview;
	$: queriesByHour = $analyticsStore.queriesByHour;
	$: categories = $analyticsStore.categories;
</script>

<div class="space-y-6">
	<div class="flex items-center justify-between">
		<h2 class="text-2xl font-bold text-white">Analytics Overview</h2>
		<a
			href="/dashboard"
			class="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 text-white rounded-lg transition-colors"
		>
			Full Dashboard
		</a>
	</div>

	<!-- Stats Grid -->
	<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
		<StatCard
			title="Active Users"
			value={overview?.active_users ?? 0}
			icon="users"
			color="cyan"
		/>
		<StatCard
			title="Total Queries"
			value={overview?.total_queries ?? 0}
			icon="messages"
			color="blue"
		/>
		<StatCard
			title="Avg Response"
			value="{overview?.avg_response_time_ms ?? 0}ms"
			icon="clock"
			color="green"
		/>
		<StatCard
			title="Error Rate"
			value="{overview?.error_rate_percent ?? 0}%"
			icon="alert"
			color={overview?.error_rate_percent > 5 ? 'red' : 'green'}
		/>
	</div>

	<!-- Charts -->
	<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
		<div class="bg-surface-800 rounded-xl p-6 border border-surface-700">
			<h3 class="text-lg font-semibold text-white mb-4">Queries Over Time</h3>
			<div class="h-64">
				<LineChart data={queriesByHour} />
			</div>
		</div>

		<div class="bg-surface-800 rounded-xl p-6 border border-surface-700">
			<h3 class="text-lg font-semibold text-white mb-4">Query Categories</h3>
			<div class="h-64">
				<DoughnutChart data={categories} />
			</div>
		</div>
	</div>

	<!-- Quick Links -->
	<div class="bg-surface-800 rounded-xl p-6 border border-surface-700">
		<h3 class="text-lg font-semibold text-white mb-4">Quick Actions</h3>
		<div class="grid grid-cols-1 md:grid-cols-3 gap-4">
			<a
				href="/dashboard"
				class="p-4 bg-surface-700 hover:bg-surface-600 rounded-lg transition-colors"
			>
				<div class="text-cyan-400 font-medium">Full Dashboard</div>
				<div class="text-surface-300 text-sm mt-1">Deep analytics with heuristics breakdown</div>
			</a>
			<button
				on:click={() => analyticsStore.fetchDashboard()}
				class="p-4 bg-surface-700 hover:bg-surface-600 rounded-lg transition-colors text-left"
			>
				<div class="text-cyan-400 font-medium">Refresh Data</div>
				<div class="text-surface-300 text-sm mt-1">Reload analytics from API</div>
			</button>
			<a
				href="/dashboard#realtime"
				class="p-4 bg-surface-700 hover:bg-surface-600 rounded-lg transition-colors"
			>
				<div class="text-cyan-400 font-medium">Live Sessions</div>
				<div class="text-surface-300 text-sm mt-1">Monitor active user sessions</div>
			</a>
		</div>
	</div>
</div>
