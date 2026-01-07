<script lang="ts">
	import '../app.css';
	import { onMount } from 'svelte';
	import { analyticsStore } from '$lib/stores/analytics';

	let loading = true;

	onMount(async () => {
		// Initialize analytics store
		await analyticsStore.fetchDashboard();
		loading = false;
	});
</script>

<div class="min-h-screen bg-surface-900">
	<!-- Header -->
	<header class="border-b border-surface-700 bg-surface-800/50 backdrop-blur-sm sticky top-0 z-50">
		<div class="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
			<div class="flex items-center gap-3">
				<div class="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
					<span class="text-white font-bold text-sm">LM</span>
				</div>
				<h1 class="text-xl font-semibold text-white">Liquid Mirror Analytics</h1>
			</div>
			<nav class="flex items-center gap-4">
				<a href="/" class="text-surface-200 hover:text-white transition-colors">Overview</a>
				<a href="/dashboard" class="text-surface-200 hover:text-white transition-colors">Dashboard</a>
			</nav>
		</div>
	</header>

	<!-- Main content -->
	<main class="max-w-7xl mx-auto px-4 py-6">
		{#if loading}
			<div class="flex items-center justify-center h-64">
				<div class="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-500"></div>
			</div>
		{:else}
			<slot />
		{/if}
	</main>
</div>
