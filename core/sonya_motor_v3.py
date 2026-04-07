#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PROYECTO SONYA v3 - Motor Dual: CRITIC+TOPSIS vs CRITIC+MRP
Arquitectura: Dos motores de evaluación con Sostenibilidad Fuerte
Universidad Jaume I (UJI)
================================================================================
Sistema mejorado con:
- Motor Dual (TOPSIS clásico vs MRP con Sostenibilidad Fuerte)
- Ranking dinámico de acciones
- Gestión de BBDD (crear, eliminar, consultar)
- Umbrales auto-calculados con percentiles
================================================================================
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PASO 0: GESTIÓN DE PERSISTENCIA (JSON)
# ============================================================================

class GestorBBDD:
    """Gestiona la persistencia de acciones y evaluaciones en JSON."""
    
    ARCHIVO_ACCIONES = "sonya_acciones.json"
    ARCHIVO_EVALUACIONES = "sonya_evaluaciones.json"
    
    @classmethod
    def cargar_acciones(cls) -> Dict:
        """Carga acciones desde JSON. Si no existe, crea BBDD histórica inicial."""
        
        if os.path.exists(cls.ARCHIVO_ACCIONES):
            with open(cls.ARCHIVO_ACCIONES, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            acciones_iniciales = {
                'Microbus': {
                    'Eco': 5920.13, 'Amb': 68.38, 'Soc': 960377.53,
                    'Tipo': 'EX-ANTE'
                },
                'Telematico': {
                    'Eco': 431.63, 'Amb': 5.73, 'Soc': 77257.43,
                    'Tipo': 'EX-POST'
                },
                'Fruteria local': {
                    'Eco': 0.71, 'Amb': 193.53, 'Soc': 481.19,
                    'Tipo': 'EX-ANTE'
                },
                'Fruta España': {
                    'Eco': 0.92, 'Amb': 261.46, 'Soc': 576.56,
                    'Tipo': 'EX-POST'
                }
            }
            cls.guardar_acciones(acciones_iniciales)
            return acciones_iniciales
    
    @classmethod
    def guardar_acciones(cls, acciones: Dict) -> None:
        """Guarda acciones a JSON."""
        with open(cls.ARCHIVO_ACCIONES, 'w', encoding='utf-8') as f:
            json.dump(acciones, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def cargar_evaluaciones(cls) -> List[Dict]:
        """Carga historial de evaluaciones desde JSON."""
        if os.path.exists(cls.ARCHIVO_EVALUACIONES):
            with open(cls.ARCHIVO_EVALUACIONES, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    @classmethod
    def guardar_evaluaciones(cls, evaluaciones: List[Dict]) -> None:
        """Guarda historial de evaluaciones a JSON."""
        with open(cls.ARCHIVO_EVALUACIONES, 'w', encoding='utf-8') as f:
            json.dump(evaluaciones, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def accion_existe(cls, nombre_accion: str, acciones: Dict) -> bool:
        """Valida si una acción existe en BBDD (case-insensitive)."""
        nombres_lower = {k.lower(): k for k in acciones.keys()}
        return nombre_accion.lower() in nombres_lower
    
    @classmethod
    def obtener_nombre_original(cls, nombre_accion: str, acciones: Dict) -> str:
        """Obtiene el nombre original de una acción."""
        nombres_lower = {k.lower(): k for k in acciones.keys()}
        return nombres_lower.get(nombre_accion.lower(), nombre_accion)
    
    @classmethod
    def eliminar_accion(cls, nombre_accion: str, acciones: Dict) -> Tuple[bool, str]:
        """
        Elimina una acción de la BBDD.
        
        Returns:
            (éxito, mensaje)
        """
        if not cls.accion_existe(nombre_accion, acciones):
            return False, f"⚠️ La acción '{nombre_accion}' no existe en la BBDD."
        
        nombre_original = cls.obtener_nombre_original(nombre_accion, acciones)
        del acciones[nombre_original]
        cls.guardar_acciones(acciones)
        
        return True, f"✅ Acción '{nombre_original}' eliminada de la BBDD permanentemente."


# ============================================================================
# PASO 1: CONSTRUCCIÓN DEL DATAFRAME DE TRABAJO
# ============================================================================

def construir_dataframe_trabajo(acciones: Dict, ex_ante_nombre: str, 
                                 sat_ex_ante: float, ex_post_nombre: str,
                                 sat_ex_post: float) -> pd.DataFrame:
    """
    Construye un DataFrame de trabajo para evaluación.
    
    Args:
        acciones: BBDD de acciones
        ex_ante_nombre: Nombre de la acción EX-ANTE
        sat_ex_ante: Satisfacción del EX-ANTE
        ex_post_nombre: Nombre de la acción EX-POST
        sat_ex_post: Satisfacción del EX-POST
    
    Returns:
        pd.DataFrame: DataFrame con las dos acciones
    """
    
    datos = []
    
    accion_ante = acciones[ex_ante_nombre]
    datos.append({
        'Acción': ex_ante_nombre,
        'Tipo': 'EX-ANTE',
        'Sat': sat_ex_ante,
        'Eco': accion_ante['Eco'],
        'Amb': accion_ante['Amb'],
        'Soc': accion_ante['Soc']
    })
    
    accion_post = acciones[ex_post_nombre]
    datos.append({
        'Acción': ex_post_nombre,
        'Tipo': 'EX-POST',
        'Sat': sat_ex_post,
        'Eco': accion_post['Eco'],
        'Amb': accion_post['Amb'],
        'Soc': accion_post['Soc']
    })
    
    return pd.DataFrame(datos)


def construir_dataframe_todas_acciones(acciones: Dict) -> pd.DataFrame:
    """Construye un DataFrame con TODAS las acciones para ranking."""
    
    datos = []
    
    for nombre, datos_accion in acciones.items():
        datos.append({
            'Acción': nombre,
            'Tipo': datos_accion['Tipo'],
            'Sat': 50,  # Satisfacción por defecto (no se usa en ranking)
            'Eco': datos_accion['Eco'],
            'Amb': datos_accion['Amb'],
            'Soc': datos_accion['Soc']
        })
    
    return pd.DataFrame(datos)


# ============================================================================
# MOTOR 1: CRITIC + TOPSIS (Clásico)
# ============================================================================

def calcular_critic_topsis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecuta CRITIC + TOPSIS sobre dos acciones (EX-ANTE vs EX-POST).
    """
    
    df_trabajo = df.copy()
    criterios_impacto = ['Eco', 'Amb', 'Soc']
    X = df_trabajo[criterios_impacto].values.astype(float)
    
    # NORMALIZACIÓN
    X_max = np.max(X, axis=0)
    X_min = np.min(X, axis=0)
    
    denominador = X_max - X_min
    denominador[denominador == 0] = 1
    
    X_norm = (X_max - X) / denominador
    
    # DESVIACIÓN TÍPICA
    std_columnas = np.std(X_norm, axis=0, ddof=0)
    
    # CORRELACIÓN
    correlacion = np.corrcoef(X_norm.T)
    correlacion = np.nan_to_num(correlacion)
    
    # INFORMACIÓN
    C = np.zeros(len(criterios_impacto))
    for j in range(len(criterios_impacto)):
        suma_anticorr = np.sum(1 - correlacion[j, :])
        C[j] = std_columnas[j] * suma_anticorr
    
    W = C / np.sum(C)
    pesos_dict = {criterios_impacto[i]: W[i] for i in range(len(criterios_impacto))}
    
    # TOPSIS
    V = X_norm * W
    A_plus = np.max(V, axis=0)
    A_minus = np.min(V, axis=0)
    
    D_plus = np.sqrt(np.sum((V - A_plus) ** 2, axis=1))
    D_minus = np.sqrt(np.sum((V - A_minus) ** 2, axis=1))
    
    suma_distancias = D_plus + D_minus
    suma_distancias[suma_distancias == 0] = 1e-10
    CC = D_minus / suma_distancias
    
    df_trabajo['CC'] = CC
    df_trabajo.attrs['pesos_critic'] = pesos_dict
    df_trabajo.attrs['motor'] = 'TOPSIS'
    
    return df_trabajo


# ============================================================================
# MOTOR 2: CRITIC + MRP CON SOSTENIBILIDAD FUERTE
# ============================================================================

def calcular_umbrales_percentiles(acciones: Dict) -> Dict:
    """
    Auto-calcula 5 umbrales (q) para cada criterio usando percentiles.
    
    q[0] = Inaceptable (Max, peor)
    q[1] = Reserva (75%)
    q[2] = Medio (50%)
    q[3] = Bueno (25%)
    q[4] = Ideal (Min, mejor)
    
    Args:
        acciones: BBDD completa
    
    Returns:
        Dict con umbrales para cada criterio
    """
    
    df_todas = construir_dataframe_todas_acciones(acciones)
    criterios = ['Eco', 'Amb', 'Soc']
    
    umbrales = {}
    
    for criterio in criterios:
        columna = df_todas[criterio].values
        
        umbrales[criterio] = {
            'q0_inaceptable': float(np.quantile(columna, 1.0)),    # Max (peor)
            'q1_reserva': float(np.quantile(columna, 0.75)),       # 75%
            'q2_medio': float(np.quantile(columna, 0.50)),         # Mediana
            'q3_bueno': float(np.quantile(columna, 0.25)),         # 25%
            'q4_ideal': float(np.quantile(columna, 0.0))           # Min (mejor)
        }
    
    return umbrales


def aplicar_mrp_estandarizacion(valor: float, umbrales: Dict, criterio: str) -> float:
    """
    Aplica MRP (Multiple Reference Point) estandarización.
    
    Escala: alpha = [0, 1, 2.5, 3, 4]
    
    Para MINIMIZACIÓN:
    - q[0] = Max (inaceptable/peor)
    - q[1] = 75% (reserva)
    - q[2] = 50% (medio)
    - q[3] = 25% (bueno)
    - q[4] = Min (ideal/mejor)
    
    Fórmula: s_j = alpha[k-1] + ((alpha[k] - alpha[k-1]) / (q[k-1] - q[k])) * (q[k-1] - valor)
    
    Args:
        valor: Valor a evaluar
        umbrales: Dict de umbrales para el criterio
        criterio: 'Eco', 'Amb' o 'Soc'
    
    Returns:
        Puntuación estandarizada (s_j)
    """
    
    alpha = [0, 1, 2.5, 3, 4]
    
    q = [
        umbrales[criterio]['q0_inaceptable'],  # Max
        umbrales[criterio]['q1_reserva'],       # 75%
        umbrales[criterio]['q2_medio'],         # 50%
        umbrales[criterio]['q3_bueno'],         # 25%
        umbrales[criterio]['q4_ideal']          # Min
    ]
    
    # Encontrar intervalo k
    if valor > q[0]:  # Peor que inaceptable (extrapolación superior)
        denominador = q[0] - q[1]
        if abs(denominador) < 1e-10:
            return 0.0
        s_j = alpha[0] - ((alpha[1] - alpha[0]) / denominador) * (valor - q[0])
        return s_j
    
    elif valor >= q[1]:  # Entre inaceptable y reserva (k=1)
        k = 1
    elif valor >= q[2]:  # Entre reserva y medio (k=2)
        k = 2
    elif valor >= q[3]:  # Entre medio y bueno (k=3)
        k = 3
    elif valor >= q[4]:  # Entre bueno e ideal (k=4)
        k = 4
    else:  # Mejor que ideal (extrapolación inferior)
        denominador = q[3] - q[4]
        if abs(denominador) < 1e-10:
            return 4.0
        s_j = alpha[4] + ((alpha[4] - alpha[3]) / denominador) * (q[4] - valor)
        return s_j
    
    # Interpolación lineal dentro de los intervalos
    # Fórmula: s_j = alpha[k-1] + ((alpha[k] - alpha[k-1]) / (q[k-1] - q[k])) * (q[k-1] - valor)
    denominador = q[k-1] - q[k]
    
    if abs(denominador) < 1e-10:  # Protección contra división por cero
        return (alpha[k-1] + alpha[k]) / 2.0
    
    s_j = alpha[k-1] + ((alpha[k] - alpha[k-1]) / denominador) * (q[k-1] - valor)
    
    return s_j


def calcular_critic_mrp(df: pd.DataFrame, acciones: Dict) -> pd.DataFrame:
    df_trabajo = df.copy()
    criterios_impacto = ['Eco', 'Amb', 'Soc']

    # ── CRITIC: pesos ──────────────────────────────────────────────────────
    X = df_trabajo[criterios_impacto].values.astype(float)
    X_max = np.max(X, axis=0)
    X_min = np.min(X, axis=0)
    denominador = X_max - X_min
    denominador[denominador == 0] = 1
    X_norm = (X_max - X) / denominador
    std_columnas = np.std(X_norm, axis=0, ddof=0)
    correlacion = np.corrcoef(X_norm.T)
    correlacion = np.nan_to_num(correlacion)
    C = np.zeros(len(criterios_impacto))
    for j in range(len(criterios_impacto)):
        C[j] = std_columnas[j] * np.sum(1 - correlacion[j, :])
    W = C / np.sum(C)
    pesos_dict = {criterios_impacto[i]: W[i] for i in range(len(criterios_impacto))}

    # ── MRP: umbrales y estandarización para TODAS las acciones ───────────
    umbrales = calcular_umbrales_percentiles(acciones)

    s_ecos, s_ambs, s_socs, ss_vals = [], [], [], []
    for _, row in df_trabajo.iterrows():
        se = aplicar_mrp_estandarizacion(row['Eco'], umbrales, 'Eco')
        sa = aplicar_mrp_estandarizacion(row['Amb'], umbrales, 'Amb')
        ss = aplicar_mrp_estandarizacion(row['Soc'], umbrales, 'Soc')
        s_ecos.append(se)
        s_ambs.append(sa)
        s_socs.append(ss)
        ss_vals.append(min(se, sa, ss))

    df_trabajo['s_eco'] = s_ecos
    df_trabajo['s_amb'] = s_ambs
    df_trabajo['s_soc'] = s_socs
    df_trabajo['SS'] = ss_vals

    df_trabajo.attrs['pesos_critic'] = pesos_dict
    df_trabajo.attrs['umbrales'] = umbrales
    df_trabajo.attrs['motor'] = 'MRP'

    return df_trabajo

# ============================================================================
# CÁLCULO DE VEREDICTOS
# ============================================================================

def calcular_veredicto_topsis(df: pd.DataFrame) -> pd.DataFrame:
    """Veredicto para motor TOPSIS."""
    
    df_trabajo = df.copy()
    veredictos = []
    
    for idx, row in df_trabajo.iterrows():
        cc = row['CC']
        sat = row['Sat']
        
        if cc < 0.50:
            veredicto = "❌ DESCARTADA"
        elif cc >= 0.50 and sat < 50:
            veredicto = "👻 OCULTA"
        elif cc >= 0.50 and sat >= 50:
            veredicto = "✅ APROBADA"
        else:
            veredicto = "⚠️ ERROR"
        
        veredictos.append(veredicto)
    
    df_trabajo['Veredicto'] = veredictos
    return df_trabajo


def calcular_veredicto_mrp(df: pd.DataFrame) -> pd.DataFrame:
    df_trabajo = df.copy()
    veredictos = []
    for _, row in df_trabajo.iterrows():
        ss  = row['SS']
        sat = row['Sat']
        if ss < 1.0:
            veredicto = "❌ DESCARTADA"
        elif ss >= 1.0 and sat < 50:
            veredicto = "👻 OCULTA"
        else:
            veredicto = "✅ APROBADA"
        veredictos.append(veredicto)
    df_trabajo['Veredicto'] = veredictos
    return df_trabajo

# ============================================================================
# MOSTRAR RESULTADOS
# ============================================================================

def mostrar_pesos_critic(pesos_dict: dict) -> None:
    """Muestra los pesos calculados por CRITIC."""
    
    print("\n" + "-"*70)
    print("  PESOS CRÍTICOS CALCULADOS POR CRITIC (Data-Driven)")
    print("-"*70)
    
    for criterio, peso in pesos_dict.items():
        porcentaje = peso * 100
        barra = "█" * int(porcentaje / 2) + "░" * (50 - int(porcentaje / 2))
        print(f"  {criterio:10s}: {porcentaje:6.2f}% {barra}")
    
    print("-"*70)


def mostrar_umbrales(umbrales: Dict) -> None:
    print("\n" + "="*70)
    print("  UMBRALES q AUTO-CALCULADOS (Percentiles de la BBDD)")
    print("="*70)
    etiquetas = [
        ("q(0)", "100% — Inaceptable (Máximo, peor valor)"),
        ("q(1)", " 75% — Reserva"),
        ("q(2)", " 50% — Medio (Mediana)"),
        ("q(3)", " 25% — Bueno"),
        ("q(4)", "  0% — Ideal (Mínimo, mejor valor)"),
    ]
    for criterio, valores in umbrales.items():
        vals_list = [
            valores['q0_inaceptable'],
            valores['q1_reserva'],
            valores['q2_medio'],
            valores['q3_bueno'],
            valores['q4_ideal'],
        ]
        print(f"\n  {criterio}:")
        for (q_label, pct_label), val in zip(etiquetas, vals_list):
            print(f"    {q_label} ({pct_label}): {val:.4f}")
    print("="*70)
def mostrar_alfa(alpha: list = None) -> None:
    if alpha is None:
        alpha = [0, 1, 2.5, 3, 4]
    print("\n" + "="*70)
    print("  ESCALA α (MRP) Y LÍNEA ROJA")
    print("="*70)
    etiquetas = ["Inaceptable", "Reserva    ", "Medio      ", "Bueno      ", "Ideal      "]
    for i, (a, et) in enumerate(zip(alpha, etiquetas)):
        marca = "  ◄── LÍNEA ROJA (mínimo para aprobar)" if i == 1 else ""
        print(f"    α[{i}] = {a:5.2f}  →  q({i}) {et}{marca}")
    print("="*70)
def mostrar_evaluacion_topsis(df: pd.DataFrame) -> None:
    """Muestra resultado de evaluación TOPSIS."""
    
    print("\n" + "="*100)
    print("  RESULTADO DE LA EVALUACIÓN (Motor CRITIC + TOPSIS)")
    print("="*100)
    
    tabla = df[['Acción', 'Tipo', 'CC', 'Sat', 'Eco', 'Amb', 'Soc', 'Veredicto']].copy()
    tabla['CC (%)'] = (tabla['CC'] * 100).round(2)
    tabla = tabla.drop('CC', axis=1)
    tabla = tabla[['Acción', 'Tipo', 'CC (%)', 'Amb', 'Soc', 'Eco', 'Veredicto']]
    print(tabla.to_string(index=False))
    print("="*100)


def mostrar_evaluacion_mrp(df: pd.DataFrame) -> None:
    """Muestra resultado de evaluación MRP."""
    
    print("\n" + "="*100)
    print("  RESULTADO DE LA EVALUACIÓN (Motor CRITIC + MRP + Sostenibilidad Fuerte)")
    print("="*100)

    tabla = df[['Acción', 'Tipo', 'SS', 's_amb', 's_soc', 's_eco', 'Veredicto']].copy()
    tabla['s_eco'] = tabla['s_eco'].round(4)
    tabla['s_amb'] = tabla['s_amb'].round(4)
    tabla['s_soc'] = tabla['s_soc'].round(4)
    tabla['SS'] = tabla['SS'].round(4)
    
    print(tabla.to_string(index=False))
    print("="*100)


def mostrar_ranking_acciones(acciones: Dict, motor: str = 'TOPSIS') -> None:
    print("\n" + "="*100)

    if motor == 'TOPSIS':
        print("  RANKING DE ACCIONES (Mayor a Menor CC - Motor CRITIC+TOPSIS)")
        df_todas = construir_dataframe_todas_acciones(acciones)
        df_todas = calcular_critic_topsis(df_todas)
        df_todas = calcular_veredicto_topsis(df_todas)
        pesos = df_todas.attrs['pesos_critic']
        mostrar_pesos_critic(pesos)
        tabla = df_todas[['Acción', 'Tipo', 'CC', 'Eco', 'Amb', 'Soc', 'Veredicto']].copy()
        tabla['CC (%)'] = (tabla['CC'] * 100).round(2)
        tabla = tabla.drop('CC', axis=1)
        tabla = tabla[['Acción', 'Tipo', 'CC (%)', 'Amb', 'Soc', 'Eco', 'Veredicto']]
        tabla = tabla.sort_values('CC (%)', ascending=False).reset_index(drop=True)

    elif motor == 'MRP':
        print("  RANKING DE ACCIONES (Mayor a Menor SS - Motor CRITIC+MRP)")
        df_todas = construir_dataframe_todas_acciones(acciones)
        df_todas = calcular_critic_mrp(df_todas, acciones)
        df_todas = calcular_veredicto_mrp(df_todas)
        pesos    = df_todas.attrs['pesos_critic']
        umbrales = df_todas.attrs['umbrales']
        mostrar_pesos_critic(pesos)
        mostrar_alfa()           # ← siempre visible
        mostrar_umbrales(umbrales)  # ← siempre visible
        tabla = df_todas[['Acción', 'Tipo', 's_eco', 's_amb', 's_soc', 'SS', 'Veredicto']].copy()
        tabla['s_eco'] = tabla['s_eco'].round(4)
        tabla['s_amb'] = tabla['s_amb'].round(4)
        tabla['s_soc'] = tabla['s_soc'].round(4)
        tabla['SS'] = tabla['SS'].round(4)
        tabla = tabla[['Acción', 'Tipo', 'SS', 's_amb', 's_soc', 's_eco', 'Veredicto']]
        tabla = tabla.sort_values('SS', ascending=False).reset_index(drop=True)
    print(tabla.to_string(index=False))
    print("="*100)

# ============================================================================
# INGRESO DE DATOS INTERACTIVO
# ============================================================================

def obtener_entero_valido(mensaje: str, minimo: int = 0, maximo: int = None) -> int:
    """Obtiene un entero válido del usuario."""
    while True:
        try:
            valor = int(input(mensaje))
            if valor < minimo:
                print(f"⚠️ El valor debe ser >= {minimo}")
                continue
            if maximo is not None and valor > maximo:
                print(f"⚠️ El valor debe ser <= {maximo}")
                continue
            return valor
        except ValueError:
            print("⚠️ Error: Introduce un número entero válido")


def obtener_float_valido(mensaje: str, minimo: float = 0.0) -> float:
    """Obtiene un float válido del usuario."""
    while True:
        try:
            valor = float(input(mensaje))
            if valor < minimo:
                print(f"⚠️ El valor debe ser >= {minimo}")
                continue
            return valor
        except ValueError:
            print("⚠️ Error: Introduce un número válido (usa punto para decimales)")


def listar_acciones(acciones: Dict) -> None:
    """Muestra lista de todas las acciones disponibles."""
    
    print("\n" + "-"*70)
    print("  ACCIONES DISPONIBLES EN LA BBDD")
    print("-"*70)
    
    for idx, (nombre, datos) in enumerate(acciones.items(), 1):
        print(f"  {idx}. {nombre} ({datos['Tipo']})")
        print(f"     Eco={datos['Eco']:.2f} | Amb={datos['Amb']:.2f} | Soc={datos['Soc']:.2f}")
    
    print("-"*70)


def ingresar_ex_ante(acciones: Dict) -> Tuple[str, float]:
    """Ingresa la acción EX-ANTE."""
    
    print(f"\n{'='*70}")
    print(f"  INGRESO DE SITUACIÓN INICIAL (EX-ANTE)")
    print(f"{'='*70}")
    
    listar_acciones(acciones)
    
    while True:
        nombre_ante = input("\n📝 Nombre de la situación inicial (EX-ANTE): ").strip()
        
        if not nombre_ante:
            print("⚠️ El nombre no puede estar vacío")
            continue
        
        if GestorBBDD.accion_existe(nombre_ante, acciones):
            nombre_original = GestorBBDD.obtener_nombre_original(nombre_ante, acciones)
            print(f"✅ Acción '{nombre_original}' encontrada.")
            
            sat = obtener_entero_valido(
                f"\n📊 Nivel de satisfacción para '{nombre_original}' (0-100): ",
                minimo=0, maximo=100
            )
            
            return nombre_original, sat
        
        else:
            print(f"⚠️ Acción '{nombre_ante}' no encontrada.")
            print(f"   Se agregará a la BBDD de forma permanente.\n")
            
            sat = obtener_entero_valido(
                f"  • Nivel de satisfacción (0-100): ",
                minimo=0, maximo=100
            )
            
            eco = obtener_float_valido(f"  • Impacto Económico (€): ", minimo=0.0)
            amb = obtener_float_valido(f"  • Impacto Ambiental (puntos): ", minimo=0.0)
            soc = obtener_float_valido(f"  • Impacto Social (puntos): ", minimo=0.0)
            
            acciones[nombre_ante] = {
                'Eco': eco, 'Amb': amb, 'Soc': soc,
                'Tipo': 'EX-ANTE'
            }
            GestorBBDD.guardar_acciones(acciones)
            
            print(f"\n✅ Situación '{nombre_ante}' agregada a la BBDD permanentemente.")
            
            return nombre_ante, sat


def ingresar_ex_post(acciones: Dict) -> Tuple[str, float]:
    """Ingresa la acción EX-POST."""
    
    print(f"\n{'='*70}")
    print(f"  INGRESO DE ACCIÓN DE MEJORA (EX-POST)")
    print(f"{'='*70}")
    
    listar_acciones(acciones)
    
    while True:
        nombre_post = input("\n📝 Nombre de la acción de mejora (EX-POST): ").strip()
        
        if not nombre_post:
            print("⚠️ El nombre no puede estar vacío")
            continue
        
        if GestorBBDD.accion_existe(nombre_post, acciones):
            nombre_original = GestorBBDD.obtener_nombre_original(nombre_post, acciones)
            print(f"✅ Acción '{nombre_original}' encontrada.")
            
            sat = obtener_entero_valido(
                f"\n📊 Nivel de satisfacción para '{nombre_original}' (0-100): ",
                minimo=0, maximo=100
            )
            
            return nombre_original, sat
        
        else:
            print(f"⚠️ Acción '{nombre_post}' no encontrada.")
            print(f"   Se agregará a la BBDD de forma permanente.\n")
            
            sat = obtener_entero_valido(
                f"  • Nivel de satisfacción (0-100): ",
                minimo=0, maximo=100
            )
            
            eco = obtener_float_valido(f"  • Impacto Económico (€): ", minimo=0.0)
            amb = obtener_float_valido(f"  • Impacto Ambiental (puntos): ", minimo=0.0)
            soc = obtener_float_valido(f"  • Impacto Social (puntos): ", minimo=0.0)
            
            acciones[nombre_post] = {
                'Eco': eco, 'Amb': amb, 'Soc': soc,
                'Tipo': 'EX-POST'
            }
            GestorBBDD.guardar_acciones(acciones)
            
            print(f"\n✅ Acción '{nombre_post}' agregada a la BBDD permanentemente.")
            
            return nombre_post, sat


def gestionar_bbdd(acciones: Dict) -> Dict:
    """
    Menú para gestionar la BBDD (ver, eliminar, etc).
    """
    
    while True:
        print(f"\n{'='*70}")
        print(f"  GESTIÓN DE LA BBDD")
        print(f"{'='*70}")
        print(f"  1. Ver todas las acciones")
        print(f"  2. Eliminar una acción")
        print(f"  3. Volver al menú principal")
        print(f"{'='*70}")
        
        opcion = input("Elige una opción (1-3): ").strip()
        
        if opcion == '1':
            listar_acciones(acciones)
        
        elif opcion == '2':
            listar_acciones(acciones)
            nombre_eliminar = input("\n📝 Nombre de la acción a eliminar: ").strip()
            
            exito, mensaje = GestorBBDD.eliminar_accion(nombre_eliminar, acciones)
            print(f"\n{mensaje}")
            
            if exito:
                acciones = GestorBBDD.cargar_acciones()  # Recargar
        
        elif opcion == '3':
            return acciones
        
        else:
            print("⚠️ Opción no válida. Intenta de nuevo.")
    
    return acciones


# ============================================================================
# FLUJO PRINCIPAL
# ============================================================================

def main():
    """Flujo principal del Proyecto SONYA v3."""
    
    print("\n" + "█"*100)
    print("█" + " "*98 + "█")
    print("█" + "  PROYECTO SONYA v3 - MOTOR DUAL: CRITIC+TOPSIS vs CRITIC+MRP".center(98) + "█")
    print("█" + "  Universidad Jaume I (UJI) | Sostenibilidad Fuerte".center(98) + "█")
    print("█" + " "*98 + "█")
    print("█"*100)
    
    # Cargar BBDD
    print("\n🔧 Cargando base de datos...")
    acciones = GestorBBDD.cargar_acciones()
    print(f"✅ BBDD cargada: {len(acciones)} acciones disponibles")
    
    # Loop principal
    while True:
        print(f"\n{'='*70}")
        print(f"  MENÚ PRINCIPAL")
        print(f"{'='*70}")
        print(f"  1. Motor CRITIC + TOPSIS (Evaluación Clásica)")
        print(f"  2. Motor CRITIC + MRP (Sostenibilidad Fuerte)")
        print(f"  3. Ver Ranking de Acciones")
        print(f"  4. Gestionar BBDD")
        print(f"  5. Salir")
        print(f"{'='*70}")
        
        opcion = input("Elige motor o acción (1-5): ").strip()
        
        # ====================================================================
        # OPCIÓN 1: CRITIC + TOPSIS
        # ====================================================================
        
        if opcion == '1':
            print("\n🔧 Has seleccionado: Motor CRITIC + TOPSIS")
            
            # Ingresar EX-ANTE y EX-POST
            nombre_ex_ante, sat_ex_ante = ingresar_ex_ante(acciones)
            acciones = GestorBBDD.cargar_acciones()
            
            nombre_ex_post, sat_ex_post = ingresar_ex_post(acciones)
            acciones = GestorBBDD.cargar_acciones()
            
            # Recalcular
            print("\n⚙️ RECALCULANDO motor CRITIC + TOPSIS...")
            
            df_evaluacion = construir_dataframe_trabajo(
                acciones, nombre_ex_ante, sat_ex_ante,
                nombre_ex_post, sat_ex_post
            )
            
            df_evaluacion = calcular_critic_topsis(df_evaluacion)
            df_evaluacion = calcular_veredicto_topsis(df_evaluacion)
            
            pesos = df_evaluacion.attrs['pesos_critic']
            mostrar_pesos_critic(pesos)
            mostrar_evaluacion_topsis(df_evaluacion)
        
        # ====================================================================
        # OPCIÓN 2: CRITIC + MRP
        # ====================================================================
        
        elif opcion == '2':
            print("\n🔧 Has seleccionado: Motor CRITIC + MRP (Sostenibilidad Fuerte)")
            
            # Ingresar EX-ANTE y EX-POST
            nombre_ex_ante, sat_ex_ante = ingresar_ex_ante(acciones)
            acciones = GestorBBDD.cargar_acciones()
            
            nombre_ex_post, sat_ex_post = ingresar_ex_post(acciones)
            acciones = GestorBBDD.cargar_acciones()
            
            # Recalcular
            print("\n⚙️ RECALCULANDO motor CRITIC + MRP...")
            
            df_evaluacion = construir_dataframe_trabajo(
                acciones, nombre_ex_ante, sat_ex_ante,
                nombre_ex_post, sat_ex_post
            )
            
            df_evaluacion = calcular_critic_mrp(df_evaluacion, acciones)
            df_evaluacion = calcular_veredicto_mrp(df_evaluacion)
            
            pesos = df_evaluacion.attrs['pesos_critic']
            umbrales = df_evaluacion.attrs['umbrales']
            
            mostrar_pesos_critic(pesos)
            mostrar_alfa()
            mostrar_umbrales(umbrales)
            mostrar_evaluacion_mrp(df_evaluacion)
        
        # ====================================================================
        # OPCIÓN 3: Ver Ranking
        # ====================================================================
        
        elif opcion == '3':
            print("\n¿Qué ranking deseas ver?")
            print("  1. Ranking por CC (Motor TOPSIS)")
            print("  2. Ranking por SS (Motor MRP)")
            
            tipo_ranking = input("\nElige (1-2): ").strip()
            
            if tipo_ranking == '1':
                mostrar_ranking_acciones(acciones, motor='TOPSIS')
            elif tipo_ranking == '2':
                mostrar_ranking_acciones(acciones, motor='MRP')
            else:
                print("⚠️ Opción no válida.")
        
        # ====================================================================
        # OPCIÓN 4: Gestionar BBDD
        # ====================================================================
        
        elif opcion == '4':
            acciones = gestionar_bbdd(acciones)
        
        # ====================================================================
        # OPCIÓN 5: Salir
        # ====================================================================
        
        elif opcion == '5':
            print("\n👋 Finalizando Proyecto SONYA v3. ¡Hasta luego!")
            break
        
        else:
            print("⚠️ Opción no válida. Intenta de nuevo.")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    main()
