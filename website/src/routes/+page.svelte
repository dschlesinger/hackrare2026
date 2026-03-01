<script lang='ts'>
    import { enhance } from '$app/forms';
    import * as Tabs from "$lib/components/ui/tabs/index.js";

    type Item = {
        id: number;
        value: string;
        type: 'gene' | 'pheno';
        present: boolean;
    };

    let geneInput = $state('');
    let phenoInput = $state('');
    let items: Item[] = $state([]);
    let nextId = $state(0);
    let submitted = $state(false);

    let tests = $state(null);
    let diseases = $state(null);
    let total_information = $state(null);

    function addItem(type: 'gene' | 'pheno') {
        const value = type === 'gene' ? geneInput.trim() : phenoInput.trim();
        if (!value) return;
        items = [...items, { id: nextId++, value, type, present: true }];
        if (type === 'gene') geneInput = '';
        else phenoInput = '';
    }

    function deleteItem(id: number) {
        items = items.filter(item => item.id !== id);
    }

    function toggleType(id: number) {
        items = items.map(item =>
            item.id === id ? { ...item, type: item.type === 'gene' ? 'pheno' : 'gene' } : item
        );
    }

    function togglePresent(id: number) {
        items = items.map(item =>
            item.id === id ? { ...item, present: !item.present } : item
        );
    }

    function getLists() {
        return {
            gene_tests: items.filter(i => i.type === 'gene' && i.present).map(i => i.value),
            pheno_tests: items.filter(i => i.type === 'pheno' && i.present).map(i => i.value),
            not_present_genes: items.filter(i => i.type === 'gene' && !i.present).map(i => i.value),
            not_present_pheno: items.filter(i => i.type === 'pheno' && !i.present).map(i => i.value),
        };
    }
</script>

<div class="p-6 max-w-2xl mx-auto space-y-6">

    <!-- Gene Input -->
    <div class="flex gap-2">
        <input
            class="flex-1 border rounded px-3 py-2 text-sm"
            placeholder="Add gene..."
            bind:value={geneInput}
            onkeydown={(e) => e.key === 'Enter' && addItem('gene')}
        />
        <button
            type="button"
            class="bg-blue-600 text-white px-4 py-2 rounded text-sm hover:bg-blue-700"
            onclick={() => addItem('gene')}
        >
            Add Gene
        </button>
    </div>

    <!-- Pheno Input -->
    <div class="flex gap-2">
        <input
            class="flex-1 border rounded px-3 py-2 text-sm"
            placeholder="Add phenotype..."
            bind:value={phenoInput}
            onkeydown={(e) => e.key === 'Enter' && addItem('pheno')}
        />
        <button
            type="button"
            class="bg-purple-600 text-white px-4 py-2 rounded text-sm hover:bg-purple-700"
            onclick={() => addItem('pheno')}
        >
            Add Pheno
        </button>
    </div>

    <!-- Item List -->
    {#if items.length > 0}
        <ul class="space-y-2">
            {#each items as item (item.id)}
                <li class="flex items-center justify-between border rounded px-4 py-2 text-sm {item.present ? 'bg-green-50' : 'bg-red-50'}">
                    <div class="flex items-center gap-3">
                        <span class="font-mono text-xs px-2 py-0.5 rounded-full {item.type === 'gene' ? 'bg-blue-100 text-blue-700' : 'bg-purple-100 text-purple-700'}">
                            {item.type}
                        </span>
                        <span>{item.value}</span>
                        {#if !item.present}
                            <span class="text-xs text-red-400 italic">absent</span>
                        {/if}
                    </div>
                    <div class="flex gap-1">
                        <button
                            type="button"
                            title="{item.present ? 'Mark absent' : 'Mark present'}"
                            class="px-2 py-1 rounded text-xs border hover:bg-gray-100"
                            onclick={() => togglePresent(item.id)}
                        >
                            {item.present ? '✓' : '✗'}
                        </button>
                        <button
                            type="button"
                            title="Switch to {item.type === 'gene' ? 'pheno' : 'gene'}"
                            class="px-2 py-1 rounded text-xs border hover:bg-gray-100"
                            onclick={() => toggleType(item.id)}
                        >
                            {item.type === 'gene' ? '→ pheno' : '→ gene'}
                        </button>
                        <button
                            type="button"
                            title="Delete"
                            class="px-2 py-1 rounded text-xs border text-red-500 hover:bg-red-50"
                            onclick={() => deleteItem(item.id)}
                        >
                            ✕
                        </button>
                    </div>
                </li>
            {/each}
        </ul>
    {:else}
        <p class="text-sm text-gray-400 italic">No items added yet.</p>
    {/if}

    <!-- Submit Form -->
    <form
        action="/api/infer"
        method="POST"
        use:enhance={() => {
            submitted = true;
            return async ({ result }) => {
                if (result.type === 'success') {
                    tests = result.data.tests;
                    diseases = result.data.diseases;
                    total_information = result.data.total_information;
                }
                submitted = false;
            };
        }}
    >
        <input type="hidden" name="gene_tests" value={JSON.stringify(getLists().gene_tests)} />
        <input type="hidden" name="pheno_tests" value={JSON.stringify(getLists().pheno_tests)} />
        <input type="hidden" name="not_present_genes" value={JSON.stringify(getLists().not_present_genes)} />
        <input type="hidden" name="not_present_pheno" value={JSON.stringify(getLists().not_present_pheno)} />

        <button
            type="submit"
            class="w-full bg-green-600 text-white py-2 rounded hover:bg-green-700 text-sm font-medium disabled:opacity-50"
            disabled={items.length === 0 || submitted}
        >
            {submitted ? 'Loading...' : 'Submit'}
        </button>
    </form>

    <!-- Results -->
    {#if tests || diseases}
        Total Information: {total_information ? total_information : 'No info on total_information'}
        <Tabs.Root value="tests">
            <Tabs.List>
                <Tabs.Trigger value="tests">Tests</Tabs.Trigger>
                <Tabs.Trigger value="diseases">Diseases</Tabs.Trigger>
            </Tabs.List>

            <Tabs.Content value="tests">
                {#if tests}
                    <div class="border rounded p-4 flex flex-col gap-y-2 mt-4">
                        {#each Object.keys(tests).toSorted((a, b) => tests[b] - tests[a]) as t}
                            <div>{t}: {tests[t].toFixed(2)}</div>
                        {/each}
                    </div>
                {/if}
            </Tabs.Content>

            <Tabs.Content value="diseases">
                {#if diseases}
                    Number of possible diseases: {Object.keys(diseases).length}
                    <div class="border rounded p-4 flex flex-col gap-y-2 mt-4">
                        {#each Object.keys(diseases).toSorted((a, b) => diseases[b] - diseases[a]) as t}
                            <div>{t}: {diseases[t].toFixed(2)}</div>
                        {/each}
                    </div>
                {/if}
            </Tabs.Content>
        </Tabs.Root>
    {/if}

</div>