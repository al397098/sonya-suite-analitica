#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sonya_api.py — API Flask para PROYECTO SONYA v3
Sirve de puente entre el frontend HTML interactivo y sonya_motor_v3.py
Soporta: Simulaciones múltiples (arrays), edición BBDD, y parámetros Alpha/Q dinámicos.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import json as json_lib

from sonya_motor_v3 import (
    GestorBBDD,
    construir_dataframe_todas_acciones,
    calcular_critic_topsis,
    calcular_critic_mrp,
    calcular_veredicto_topsis,
    calcular_veredicto_mrp,
    calcular_umbrales_percentiles
)

app = Flask(__name__)
CORS(app)


# ── Utilidad: convertir numpy a tipos Python nativos ──────────────────────
def limpia(obj):
    if isinstance(obj, dict):
        return {k: limpia(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [limpia(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ── GESTIÓN DE BBDD ───────────────────────────────────────────────────────

@app.route('/acciones', methods=['GET'])
def get_acciones():
    acciones = GestorBBDD.cargar_acciones()
    resultado = []
    for nombre, datos in acciones.items():
        resultado.append({
            'nombre': nombre,
            'tipo': datos['Tipo'],
            'eco': datos['Eco'],
            'amb': datos['Amb'],
            'soc': datos['Soc'],
        })
    return jsonify(resultado)


@app.route('/acciones', methods=['POST'])
def post_accion():
    acciones = GestorBBDD.cargar_acciones()
    data = request.json
    nombre = data.get('nombre', '').strip()
    if not nombre:
        return jsonify({'error': 'Nombre vacío'}), 400
    if GestorBBDD.accion_existe(nombre, acciones):
        return jsonify({'error': f"La acción '{nombre}' ya existe"}), 409
    acciones[nombre] = {
        'Eco': float(data.get('eco', 0)),
        'Amb': float(data.get('amb', 0)),
        'Soc': float(data.get('soc', 0)),
        'Tipo': data.get('tipo', 'EX-POST'),
    }
    GestorBBDD.guardar_acciones(acciones)
    return jsonify({'ok': True, 'nombre': nombre})


@app.route('/acciones/<nombre>', methods=['PUT'])
def put_accion(nombre):
    acciones = GestorBBDD.cargar_acciones()
    nombre_orig = GestorBBDD.obtener_nombre_original(nombre, acciones)
    if not nombre_orig:
        return jsonify({'error': f"Acción '{nombre}' no encontrada"}), 404
    data = request.json
    acciones[nombre_orig]['Eco'] = float(data.get('eco', acciones[nombre_orig]['Eco']))
    acciones[nombre_orig]['Amb'] = float(data.get('amb', acciones[nombre_orig]['Amb']))
    acciones[nombre_orig]['Soc'] = float(data.get('soc', acciones[nombre_orig]['Soc']))
    GestorBBDD.guardar_acciones(acciones)
    return jsonify({'ok': True, 'nombre': nombre_orig})


@app.route('/acciones/<nombre>', methods=['DELETE'])
def delete_accion(nombre):
    acciones = GestorBBDD.cargar_acciones()
    exito, msg = GestorBBDD.eliminar_accion(nombre, acciones)
    if not exito:
        return jsonify({'error': msg}), 404
    return jsonify({'ok': True, 'mensaje': msg})


# ── RANKINGS GLOBALES ─────────────────────────────────────────────────────

@app.route('/ranking/topsis', methods=['GET'])
def ranking_topsis():
    acciones = GestorBBDD.cargar_acciones()
    df = construir_dataframe_todas_acciones(acciones)
    df = calcular_critic_topsis(df)
    df = calcular_veredicto_topsis(df)

    pesos = df.attrs['pesos_critic']
    filas = []

    # Extraer variables matemáticas necesarias para el frontend
    std_dev = np.std((np.max(df[['Eco', 'Amb', 'Soc']].values.astype(float), axis=0) - df[
        ['Eco', 'Amb', 'Soc']].values.astype(float)) / np.maximum(
        np.max(df[['Eco', 'Amb', 'Soc']].values.astype(float), axis=0) - np.min(
            df[['Eco', 'Amb', 'Soc']].values.astype(float), axis=0), 1e-10), axis=0, ddof=0)

    for _, row in df.sort_values('CC', ascending=False).iterrows():
        filas.append({
            'accion': row['Acción'],
            'tipo': row['Tipo'],
            'cc': float(row['CC']),  # El frontend multiplica por 100
            'eco': float(row['Eco']),
            'amb': float(row['Amb']),
            'soc': float(row['Soc']),
            'veredicto': row['Veredicto'],
            # Cálculos internos (D+, D-) necesarios para la vista matemática
            'd_plus': getattr(row, 'D_plus', 0.0),
            'd_minus': getattr(row, 'D_minus', 0.0)
        })

    return jsonify(limpia({
        'ranking': filas,
        'pesos': pesos,
        'std': {'Eco': std_dev[0], 'Amb': std_dev[1], 'Soc': std_dev[2]}
    }))


@app.route('/ranking/mrp', methods=['GET'])
def ranking_mrp_v2():
    acciones = GestorBBDD.cargar_acciones()

    alpha_raw = request.args.get('alpha')
    lr_raw = request.args.get('lr')
    q_raw = request.args.get('q')

    alpha = json_lib.loads(alpha_raw) if alpha_raw else [0, 1, 2.5, 3, 4]
    lr = float(lr_raw) if lr_raw else 1.0
    q_custom = json_lib.loads(q_raw) if q_raw and q_raw not in ('', 'null') else None

    criterios = ['Eco', 'Amb', 'Soc']
    df = construir_dataframe_todas_acciones(acciones)

    # CRITIC pesos
    X = df[criterios].values.astype(float)
    Xmax = np.max(X, axis=0);
    Xmin = np.min(X, axis=0)
    den = Xmax - Xmin;
    den[den == 0] = 1
    Xnorm = (Xmax - X) / den
    std = np.std(Xnorm, axis=0, ddof=0)
    corr = np.corrcoef(Xnorm.T);
    corr = np.nan_to_num(corr)
    C = np.array([std[j] * np.sum(1 - corr[j, :]) for j in range(3)])
    W = C / np.sum(C)
    pesos_dict = {criterios[i]: float(W[i]) for i in range(3)}

    # Umbrales
    if q_custom:
        umbrales = {}
        for c in criterios:
            vals = q_custom.get(c, [])
            keys = ['q0_inaceptable', 'q1_reserva', 'q2_medio', 'q3_bueno', 'q4_ideal']
            umbrales[c] = {keys[i]: float(vals[i]) for i in range(5)} if len(vals) == 5 else \
            calcular_umbrales_percentiles(acciones)[c]
    else:
        umbrales = calcular_umbrales_percentiles(acciones)

    def mrp_eval(valor, crit):
        q_map = umbrales[crit]
        q = [q_map['q0_inaceptable'], q_map['q1_reserva'], q_map['q2_medio'], q_map['q3_bueno'], q_map['q4_ideal']]
        a = alpha
        if valor >= q[0]:
            den = q[0] - q[1]
            if abs(den) < 1e-10: return 0.0
            return a[0] - ((a[1] - a[0]) / den) * (valor - q[0])
        k = 1 if valor >= q[1] else 2 if valor >= q[2] else 3 if valor >= q[3] else 4 if valor >= q[4] else 5
        if k == 5:
            den = q[3] - q[4]
            if abs(den) < 1e-10: return a[4]
            return a[4] + ((a[4] - a[3]) / den) * (q[4] - valor)
        den = q[k - 1] - q[k]
        if abs(den) < 1e-10: return (a[k - 1] + a[k]) / 2
        return a[k - 1] + ((a[k] - a[k - 1]) / den) * (q[k - 1] - valor)

    filas = []
    for _, row in df.iterrows():
        se = mrp_eval(row['Eco'], 'Eco')
        sa = mrp_eval(row['Amb'], 'Amb')
        ss_ = mrp_eval(row['Soc'], 'Soc')
        SS = min(se, sa, ss_)
        filas.append(
            {'accion': row['Acción'], 'tipo': row['Tipo'], 'ss': round(float(SS), 4), 's_eco': round(float(se), 4),
             's_amb': round(float(sa), 4), 's_soc': round(float(ss_), 4), 'eco': float(row['Eco']),
             'amb': float(row['Amb']), 'soc': float(row['Soc']),
             'veredicto': '✅ APROBADA' if SS >= lr else '❌ DESCARTADA'})

    filas.sort(key=lambda x: x['ss'], reverse=True)

    umbrales_fmt = {}
    for c, vals in umbrales.items():
        umbrales_fmt[c] = [
            {'pct': '100%', 'label': 'Inaceptable', 'valor': round(vals['q0_inaceptable'], 4)},
            {'pct': '75%', 'label': 'Reserva', 'valor': round(vals['q1_reserva'], 4)},
            {'pct': '50%', 'label': 'Medio', 'valor': round(vals['q2_medio'], 4)},
            {'pct': '25%', 'label': 'Bueno', 'valor': round(vals['q3_bueno'], 4)},
            {'pct': '0%', 'label': 'Ideal', 'valor': round(vals['q4_ideal'], 4)},
        ]

    return jsonify(
        limpia({'ranking': filas, 'pesos': pesos_dict, 'umbrales': umbrales_fmt, 'alpha': alpha, 'linea_roja': lr}))


# ── SIMULADORES MÚLTIPLES ─────────────────────────────────────────────────

@app.route('/evaluar/topsis', methods=['POST'])
def evaluar_topsis_v2():
    from sonya_motor_v3 import construir_dataframe_trabajo

    data = request.json
    acciones = GestorBBDD.cargar_acciones()

    ex_ante = data.get('ex_ante')
    sat_ante = float(data.get('sat_ante', 50))
    ex_posts = data.get('ex_posts', [])

    if not GestorBBDD.accion_existe(ex_ante, acciones):
        ep_data = data.get('ante_data', {})
        acciones[ex_ante] = {'Eco': float(ep_data.get('eco', 0)), 'Amb': float(ep_data.get('amb', 0)),
                             'Soc': float(ep_data.get('soc', 0)), 'Tipo': 'EX-ANTE'}
    ex_ante = GestorBBDD.obtener_nombre_original(ex_ante, acciones)

    resultados = []
    for ep in ex_posts:
        nombre_ep = ep['nombre']
        sat_ep = float(ep.get('sat', 70))
        if not GestorBBDD.accion_existe(nombre_ep, acciones):
            acciones[nombre_ep] = {'Eco': float(ep.get('eco', 0)), 'Amb': float(ep.get('amb', 0)),
                                   'Soc': float(ep.get('soc', 0)), 'Tipo': 'EX-POST'}
        nombre_ep = GestorBBDD.obtener_nombre_original(nombre_ep, acciones)

        df = construir_dataframe_trabajo(acciones, ex_ante, sat_ante, nombre_ep, sat_ep)
        df = calcular_critic_topsis(df)
        df = calcular_veredicto_topsis(df)

        for _, row in df.iterrows():
            if row['Tipo'] == 'EX-POST':  # Solo devolver el resultado de la evaluada
                resultados.append({
                    'accion': row['Acción'],
                    'tipo': row['Tipo'],
                    'cc': float(row['CC']),
                    'eco': float(row['Eco']),
                    'amb': float(row['Amb']),
                    'soc': float(row['Soc']),
                    'veredicto': row['Veredicto'],
                })

    return jsonify(limpia({'resultado': resultados}))


@app.route('/evaluar/mrp', methods=['POST'])
def evaluar_mrp_v2():
    from sonya_motor_v3 import construir_dataframe_trabajo

    data = request.json
    acciones = GestorBBDD.cargar_acciones()

    ex_ante = data.get('ex_ante')
    sat_ante = float(data.get('sat_ante', 50))
    ex_posts = data.get('ex_posts', [])
    alpha_in = data.get('alpha', [0, 1, 2.5, 3, 4])
    lr = float(data.get('lr', 1.0))
    q_in = data.get('q')

    if not GestorBBDD.accion_existe(ex_ante, acciones):
        ep_data = data.get('ante_data', {})
        acciones[ex_ante] = {'Eco': float(ep_data.get('eco', 0)), 'Amb': float(ep_data.get('amb', 0)),
                             'Soc': float(ep_data.get('soc', 0)), 'Tipo': 'EX-ANTE'}
    ex_ante = GestorBBDD.obtener_nombre_original(ex_ante, acciones)

    if q_in:
        umbrales_custom = {}
        for c in ['Eco', 'Amb', 'Soc']:
            vals = q_in.get(c, [])
            keys = ['q0_inaceptable', 'q1_reserva', 'q2_medio', 'q3_bueno', 'q4_ideal']
            umbrales_custom[c] = {keys[i]: float(vals[i]) for i in range(5)} if len(vals) == 5 else None
    else:
        umbrales_custom = None

    resultados = []

    def aplicar_mrp_custom(valor, umbrales, criterio, alpha):
        q_map = umbrales[criterio]
        q = [q_map['q0_inaceptable'], q_map['q1_reserva'], q_map['q2_medio'], q_map['q3_bueno'], q_map['q4_ideal']]
        a = alpha
        if valor >= q[0]:
            den = q[0] - q[1]
            if abs(den) < 1e-10: return 0.0
            return a[0] - ((a[1] - a[0]) / den) * (valor - q[0])
        k = 1 if valor >= q[1] else 2 if valor >= q[2] else 3 if valor >= q[3] else 4 if valor >= q[4] else 5
        if k == 5:
            den = q[3] - q[4]
            if abs(den) < 1e-10: return a[4]
            return a[4] + ((a[4] - a[3]) / den) * (q[4] - valor)
        den = q[k - 1] - q[k]
        if abs(den) < 1e-10: return (a[k - 1] + a[k]) / 2
        return a[k - 1] + ((a[k] - a[k - 1]) / den) * (q[k - 1] - valor)

    umbrales_uso = umbrales_custom if umbrales_custom else calcular_umbrales_percentiles(acciones)

    for ep in ex_posts:
        nombre_ep = ep['nombre']
        sat_ep = float(ep.get('sat', 70))
        if not GestorBBDD.accion_existe(nombre_ep, acciones):
            acciones[nombre_ep] = {'Eco': float(ep.get('eco', 0)), 'Amb': float(ep.get('amb', 0)),
                                   'Soc': float(ep.get('soc', 0)), 'Tipo': 'EX-POST'}
        nombre_ep = GestorBBDD.obtener_nombre_original(nombre_ep, acciones)

        df = construir_dataframe_trabajo(acciones, ex_ante, sat_ante, nombre_ep, sat_ep)

        for _, row in df.iterrows():
            if row['Tipo'] == 'EX-POST':
                se = aplicar_mrp_custom(row['Eco'], umbrales_uso, 'Eco', alpha_in)
                sa = aplicar_mrp_custom(row['Amb'], umbrales_uso, 'Amb', alpha_in)
                ss_ = aplicar_mrp_custom(row['Soc'], umbrales_uso, 'Soc', alpha_in)
                SS = min(se, sa, ss_)
                resultados.append({
                    'accion': row['Acción'],
                    'tipo': row['Tipo'],
                    'ss': round(float(SS), 4),
                    's_eco': round(float(se), 4),
                    's_amb': round(float(sa), 4),
                    's_soc': round(float(ss_), 4),
                    'eco': float(row['Eco']),
                    'amb': float(row['Amb']),
                    'soc': float(row['Soc'])
                })

    return jsonify(limpia({'resultado': resultados}))


import os

if __name__ == '__main__':
    # Render asigna un puerto dinámico, por eso usamos os.environ.get
    port = int(os.environ.get("PORT", 5000))
    # Importante: host '0.0.0.0' para que sea visible externamente
    app.run(host='0.0.0.0', port=port)