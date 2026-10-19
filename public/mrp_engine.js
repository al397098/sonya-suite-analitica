/**
 * SONYA v4.0 - Motor MRP Nativo en JavaScript
 * Ejecución 100% en el cliente (0 latencia, sin Render)
 */
const MRPEngine = {
    ALPHA_DEFAULT: [0.0, 1.0, 2.5, 3.0, 4.0], // Escala alpha (q0 a q4)
    LR_DEFAULT: 1.0, // LÍNEA ROJA DE VETO (Si es < 1.0, muere)

    // Umbrales de referencia q0 (Inaceptable) -> q4 (Ideal)
    UMBRALES: {
        "Eco": { q0: 100, q1: 500, q2: 1000, q3: 3000, q4: 6000 },
        "Amb": { q0: 10,  q1: 30,  q2: 50,   q3: 70,   q4: 90 },
        "Soc": { q0: 50000, q1: 100000, q2: 300000, q3: 600000, q4: 900000 }
    },

    // Función matemática de estandarización por interpolación lineal
    aplicarEstandarizacion: function(valor, u, alpha) {
        const q = [u.q0, u.q1, u.q2, u.q3, u.q4];
        
        // Peor que inaceptable (Extrapolación inferior)
        if (valor <= q[0]) {
            let den = q[1] - q[0];
            if (Math.abs(den) < 1e-10) return alpha[0];
            return alpha[0] - ((alpha[1] - alpha[0]) / den) * (q[0] - valor);
        }

        // Interpolación entre tramos
        for (let k = 1; k <= 4; k++) {
            if (valor <= q[k]) {
                let den = q[k] - q[k-1];
                if (Math.abs(den) < 1e-10) return (alpha[k-1] + alpha[k]) / 2;
                return alpha[k-1] + ((alpha[k] - alpha[k-1]) / den) * (valor - q[k-1]);
            }
        }

        // Mejor que ideal (Extrapolación superior)
        let den = q[4] - q[3];
        if (Math.abs(den) < 1e-10) return alpha[4];
        return alpha[4] + ((alpha[4] - alpha[3]) / den) * (valor - q[4]);
    },

    // Cálculo de Sostenibilidad Fuerte
    calcularSS: function(eco, amb, soc) {
        const s_eco = this.aplicarEstandarizacion(eco, this.UMBRALES["Eco"], this.ALPHA_DEFAULT);
        const s_amb = this.aplicarEstandarizacion(amb, this.UMBRALES["Amb"], this.ALPHA_DEFAULT);
        const s_soc = this.aplicarEstandarizacion(soc, this.UMBRALES["Soc"], this.ALPHA_DEFAULT);

        // AGREGACIÓN DE LEONTIEF: El Veredicto es el valor más bajo
        const ss = Math.min(s_eco, s_amb, s_soc);
        const viable = ss >= this.LR_DEFAULT;

        // Diagnóstico de Fallo
        let pilar_fallido = "Ninguno";
        if (!viable) {
            if (ss === s_eco) pilar_fallido = "Económico";
            else if (ss === s_amb) pilar_fallido = "Ambiental";
            else if (ss === s_soc) pilar_fallido = "Social";
        }

        return {
            ss: parseFloat(ss.toFixed(2)),
            detalles: {
                eco: parseFloat(s_eco.toFixed(2)),
                amb: parseFloat(s_amb.toFixed(2)),
                soc: parseFloat(s_soc.toFixed(2))
            },
            viable: viable,
            pilar_fallido: pilar_fallido
        };
    }
};