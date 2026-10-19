// FUNCIÓN PRINCIPAL DEL MOTOR DE SUGERENCIA
function runSuggestionEngine(data, minEffectiveness, alphaVeto) {
    // 1. FILTRADO: Solo viables por MRP y que superen efectividad mínima
    let viableActions = data.filter(item =>
        item.ssScore >= alphaVeto &&
        (window.CASP12 ? CASP12.passes(item.effectiveness, minEffectiveness) : item.effectiveness >= minEffectiveness)
    );

    if (viableActions.length < 2) {
        console.warn("No hay suficientes alternativas viables para desempatar.");
        return null;
    }

    // 2. PREPARACIÓN DE MATRIZ [SS, E]
    const matrix = viableActions.map(a => [a.ssScore, a.effectiveness / 100]);

    // 3. CÁLCULO DE PESOS CRITIC (Endógenos)
    const weights = calculateCriticWeights(matrix);

    // 4. GOAL PROGRAMMING: Minimizar Z
    // Z = w1 * ((4 - SS)/4) + w2 * ((1 - E)/1)
    const results = viableActions.map((action, index) => {
        const ssFall = (4 - action.ssScore) / 4;
        const eFall = (1 - (action.effectiveness / 100));

        const zScore = (weights[0] * ssFall) + (weights[1] * eFall);

        return {
            name: action.name,
            ssScore: action.ssScore.toFixed(3),
            satIndex: action.effectiveness + "%",
            zValue: zScore.toFixed(4)
        };
    });

    // Ordenar por Z (de menor a mayor penalización)
    return results.sort((a, b) => a.zValue - b.zValue);
}

// CÁLCULO DE PESOS CRITIC PARA 2 COLUMNAS
function calculateCriticWeights(matrix) {
    const n = matrix.length;

    // Normalización Min-Max (0-1)
    const mins = [Math.min(...matrix.map(r => r[0])), Math.min(...matrix.map(r => r[1]))];
    const maxs = [Math.max(...matrix.map(r => r[0])), Math.max(...matrix.map(r => r[1]))];

    const normMatrix = matrix.map(r => [
        (maxs[0] - mins[0]) === 0 ? 1 : (r[0] - mins[0]) / (maxs[0] - mins[0]),
        (maxs[1] - mins[1]) === 0 ? 1 : (r[1] - mins[1]) / (maxs[1] - mins[1])
    ]);

    // Desviación Típica (σ)
    const sigmas = [0, 1].map(col => {
        const mean = normMatrix.reduce((a, b) => a + b[col], 0) / n;
        return Math.sqrt(normMatrix.reduce((a, b) => a + Math.pow(b[col] - mean, 2), 0) / n);
    });

    // Correlación de Pearson (r)
    let r = 0;
    if (n > 1) {
        const meanSS = normMatrix.reduce((a, b) => a + b[0], 0) / n;
        const meanE = normMatrix.reduce((a, b) => a + b[1], 0) / n;
        const num = normMatrix.reduce((a, b) => a + (b[0] - meanSS) * (b[1] - meanE), 0);
        const den = Math.sqrt(normMatrix.reduce((a, b) => a + Math.pow(b[0] - meanSS, 2), 0)) * Math.sqrt(normMatrix.reduce((a, b) => a + Math.pow(b[1] - meanE, 2), 0));
        r = den === 0 ? 0 : num / den;
    }

    // Cantidad de Información (Cj)
    const c = sigmas.map(s => s * (1 - r));
    const sumC = c[0] + c[1];

    // Pesos Finales
    return sumC === 0 ? [0.5, 0.5] : [c[0] / sumC, c[1] / sumC];
}
document.getElementById('toggleGP').addEventListener('click', function () {
    const dot = document.getElementById('toggleDot');
    const container = document.getElementById('suggestionContainer');
    const isEnabled = this.classList.contains('bg-purple-600');

    if (!isEnabled) {
        // ACTIVAR
        this.classList.replace('bg-gray-200', 'bg-purple-600');
        dot.classList.replace('translate-x-1', 'translate-x-6');
        container.classList.remove('hidden');

        // Ejecutar motor (Suponiendo que 'currentResults' es tu array de datos actual)
        const ranking = runSuggestionEngine(currentResults, minRequiredEff, currentVeto);
        renderGPTable(ranking);
    } else {
        // DESACTIVAR
        this.classList.replace('bg-purple-600', 'bg-gray-200');
        dot.classList.replace('translate-x-6', 'translate-x-1');
        container.classList.add('hidden');
    }
});

function renderGPTable(data) {
    const body = document.getElementById('gpTableBody');
    body.innerHTML = "";

    if (!data) {
        body.innerHTML = `<tr><td colspan="4" class="px-6 py-8 text-center text-gray-400 italic text-sm">No hay suficientes alternativas que superen tanto el Veto como la Efectividad mínima.</td></tr>`;
        return;
    }

    data.forEach((row, index) => {
        const isWinner = index === 0;
        body.innerHTML += `
            <tr class="${isWinner ? 'bg-purple-50/30' : ''} hover:bg-gray-50 transition-colors">
                <td class="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">
                    ${isWinner ? 'THE BEST' : ''}${row.name}
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-center text-gray-600 font-mono">${row.ssScore}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-center text-gray-600 font-mono">${row.satIndex}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-center font-bold ${isWinner ? 'text-purple-700' : 'text-gray-500'} font-mono bg-purple-50/50">
                    ${row.zValue}
                </td>
            </tr>
        `;
    });
}