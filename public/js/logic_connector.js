import { db, auth } from './firebase-config.js';
import {
    collection, getDocs, addDoc, deleteDoc, updateDoc, doc, query, where, writeBatch
} from "https://www.gstatic.com/firebasejs/10.8.0/firebase-firestore.js";
import { onAuthStateChanged } from "https://www.gstatic.com/firebasejs/10.8.0/firebase-auth.js";

// ─── SHARE CASP-12 · Effectiveness Index (EI) ───────────────────────
// Raw scale 12–48 pts  →  EI (%) = (raw − 12) / 36 · 100   (52 % ≈ 30.7 pts)
// In Firestore the EI is stored in the legacy field `satisfaccion` (in %); null = NS.
const CASP12 = {
    MIN: 12, MAX: 48, SPAN: 36,
    FALLBACK_TAU: 52,   // only if the EX-ANTE baseline has no defined EI
    EPS: 1e-9,          // floating-point tolerance so that "equal" counts as passing
    rawToPct(raw) { return (raw - 12) / 36 * 100; },
    pctToRaw(pct) { return 12 + (pct / 100) * 36; },
    isNS(v) {
        return v === null || v === undefined || v === '' ||
            String(v).trim().toUpperCase() === 'NS' || !Number.isFinite(parseFloat(v));
    },
    // User input → number (%) | 'NS'.  Accepts "78", "78%", "30.7pts", "NS", "".
    parse(input) {
        if (this.isNS(input)) return 'NS';
        const s = String(input).trim().toUpperCase().replace(',', '.');
        const n = parseFloat(s);
        const pct = /PTS?$/.test(s) ? this.rawToPct(n) : n;
        return Math.min(100, Math.max(0, Math.round(pct * 100) / 100));
    },
    // GATE 1 (Expert A): EI >= tau_EI  (greater OR EQUAL; never strict)
    passes(ei, tau) {
        return !this.isNS(ei) && parseFloat(ei) >= parseFloat(tau) - this.EPS;
    },
    // Default tau_EI = EI of the active Action's EX-ANTE baseline (A1)
    defaultTau(alts) {
        const base = (alts || []).filter(a => a.tipo === 'EX-ANTE');
        const a1 = base.find(a => /^A1$/i.test(String(a.nombre || a.accion || '').trim())) || base[0];
        return (a1 && !this.isNS(a1.satisfaccion)) ? parseFloat(a1.satisfaccion) : this.FALLBACK_TAU;
    },
    fmt(ei) {
        return this.isNS(ei) ? 'NS'
            : `${+parseFloat(ei).toFixed(1)}% · ${this.pctToRaw(parseFloat(ei)).toFixed(1)} pts`;
    },
    fmtTau(tau) { return this.fmt(tau); }
};
window.CASP12 = CASP12;

const BATCH_LIMIT = 450; // Firestore allows 500 writes per batch; keep headroom

class SonyaConnector {
    constructor() {
        this.uid = null; // Unique ID of the signed-in user
    }

    // ─── Singleton & Auth Gatekeeper ─────────────────────────
    // A single shared promise guarantees that only ONE onAuthStateChanged
    // listener is ever registered for initialisation, and it is detached
    // as soon as the first auth state resolves (no listener leak).
    static init() {
        if (window.__sonyaConnectorPromise) return window.__sonyaConnectorPromise;

        const inApp = window.location.pathname.includes('/public/');

        window.__sonyaConnectorPromise = new Promise((resolve) => {
            const unsubscribe = onAuthStateChanged(auth, (user) => {
                unsubscribe(); // one-shot: detach immediately
                const instance = new SonyaConnector();

                if (user) {
                    instance.uid = user.uid;
                    window.__sonyaConnector = instance;
                    // Single lightweight watcher: if the session ends elsewhere, leave the app.
                    if (inApp) {
                        onAuthStateChanged(auth, (u) => {
                            if (!u) window.location.href = '../index.html';
                        });
                    }
                    resolve(instance);
                } else if (inApp) {
                    window.location.href = '../index.html';
                } else {
                    resolve(instance); // On index.html the login UI handles the signed-out state
                }
            });
        });

        return window.__sonyaConnectorPromise;
    }

    // ─── Private per-user paths: usuarios/{uid}/acciones | alternativas ──
    getAccionesRef() { return collection(db, "usuarios", this.uid, "acciones"); }
    getAlternativasRef() { return collection(db, "usuarios", this.uid, "alternativas"); }
    getAccionDoc(ufId) { return doc(db, "usuarios", this.uid, "acciones", ufId); }
    getAltDoc(altId) { return doc(db, "usuarios", this.uid, "alternativas", altId); }

    // ─── Async public API ────────────────────────────────────
    async getUFs() {
        try {
            const ufs = [];
            const querySnapshot = await getDocs(this.getAccionesRef());
            querySnapshot.forEach((d) => {
                const data = d.data();
                ufs.push({ id: d.id, label: data.label || d.id, unit: data.unit || 'Units', count: data.count || 0 });
            });
            return ufs;
        } catch (error) { console.error(error); return []; }
    }

    async addUF(ufData) {
        try {
            const docRef = await addDoc(this.getAccionesRef(), { label: ufData.label, unit: ufData.unit || 'Units', count: 0 });
            return { id: docRef.id, ...ufData };
        } catch (error) { console.error(error); throw error; }
    }

    // Cascade delete using Firestore write batches.
    // Up to BATCH_LIMIT alternatives + the Action itself are removed atomically in one
    // batch; larger Actions are chunked and the Action document is deleted in the LAST
    // batch, so an interrupted run never leaves orphaned alternatives without a parent.
    async removeUF(ufId) {
        try {
            const q = query(this.getAlternativasRef(), where("ufId", "==", ufId));
            const snap = await getDocs(q);
            const refs = snap.docs.map(d => d.ref);

            for (let i = 0; i < refs.length; i += BATCH_LIMIT) {
                const batch = writeBatch(db);
                refs.slice(i, i + BATCH_LIMIT).forEach(r => batch.delete(r));
                const isLast = i + BATCH_LIMIT >= refs.length;
                if (isLast) batch.delete(this.getAccionDoc(ufId));
                await batch.commit();
            }
            if (refs.length === 0) {
                const batch = writeBatch(db);
                batch.delete(this.getAccionDoc(ufId));
                await batch.commit();
            }
            return true;
        } catch (error) {
            console.error("Error deleting Action:", error);
            throw error;
        }
    }

    async getAlternatives(ufId) {
        try {
            const alts = [];
            const q = query(this.getAlternativasRef(), where("ufId", "==", ufId));
            const querySnapshot = await getDocs(q);
            querySnapshot.forEach((d) => {
                const data = d.data();
                alts.push({
                    id: d.id, ufId: data.ufId, nombre: data.nombre, tipo: data.tipo || 'EX-POST',
                    eco: parseFloat(data.eco) || 0, amb: parseFloat(data.amb) || 0, soc: parseFloat(data.soc) || 0,
                    satisfaccion: CASP12.isNS(data.satisfaccion) ? 'NS' : parseFloat(data.satisfaccion), source: 'cloud'
                });
            });
            return alts;
        } catch (error) { console.error(error); return []; }
    }

    async addAlternative(ufId, altData) {
        try {
            const ei = CASP12.parse(altData.satisfaccion ?? altData.efectividad ?? altData.effectiveness);
            const flatAlt = {
                ufId: ufId,
                nombre: altData.nombre || altData.label || 'New Alternative',
                tipo: altData.tipo || 'EX-POST',
                eco: parseFloat(altData.eco ?? altData.scores?.eco) || 0,
                amb: parseFloat(altData.amb ?? altData.scores?.amb) || 0,
                soc: parseFloat(altData.soc ?? altData.scores?.soc) || 0,
                satisfaccion: ei === 'NS' ? null : ei
            };
            const docRef = await addDoc(this.getAlternativasRef(), flatAlt);
            return { id: docRef.id, ...flatAlt, satisfaccion: flatAlt.satisfaccion ?? 'NS', source: 'cloud' };
        } catch (error) { console.error(error); throw error; }
    }

    async updateEffectiveness(altId, ei) {
        try {
            const v = CASP12.parse(ei);
            await updateDoc(this.getAltDoc(altId), { satisfaccion: v === 'NS' ? null : v });
            return true;
        } catch (error) { console.error(error); throw error; }
    }

    async removeAlternative(altId) {
        try {
            await deleteDoc(this.getAltDoc(altId));
            return true;
        } catch (error) { console.error(error); throw error; }
    }
}
window.SonyaConnector = SonyaConnector;
