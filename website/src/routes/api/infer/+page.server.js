export const actions = {
    default: async ({ request }) => {
        const formData = await request.formData();

        const gene_tests = JSON.parse(formData.get('gene_tests'));
        const pheno_tests = JSON.parse(formData.get('pheno_tests'));
        const not_present_genes = JSON.parse(formData.get('not_present_genes'));
        const not_present_pheno = JSON.parse(formData.get('not_present_pheno'));

        const response = await fetch('http://localhost:8000/inference', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ gene_tests, pheno_tests, not_present_genes, not_present_pheno })
        });

        const data = await response.json();

        const { tests, diseases, total_information } = data;
        return { tests, diseases, total_information };
    }
};