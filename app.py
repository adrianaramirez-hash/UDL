# app.py
import streamlit as st
import pandas as pd

import encuesta_calidad
import observacion_clases
import aulas_virtuales
import indice_reprobacion
import evaluacion_docente
from examenes_departamentales import render_examenes_departamentales

# 🔴 IMPORT CAPACITACIÓN (CON DEBUG REAL)
try:
    import capacitacion_docente
    HAS_CAPACITACION_MOD = True
    ERROR_CAP = None
except Exception as e:
    capacitacion_docente = None
    HAS_CAPACITACION_MOD = False
    ERROR_CAP = e

# Otros módulos
try:
    import encuesta_calidad_f
    HAS_EC_F_MOD = True
except:
    encuesta_calidad_f = None
    HAS_EC_F_MOD = False

try:
    import bajas_retencion
    HAS_BAJAS_MOD = True
except:
    bajas_retencion = None
    HAS_BAJAS_MOD = False

try:
    import seguimiento_inscripciones
    HAS_SEGUIMIENTO_INS_MOD = True
except:
    seguimiento_inscripciones = None
    HAS_SEGUIMIENTO_INS_MOD = False


# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Dirección Académica",
    layout="wide"
)

# ============================================================
# SIMULACIÓN USUARIO
# ============================================================
if "user_rol" not in st.session_state:
    st.session_state["user_rol"] = "DG"
    st.session_state["user_modulos"] = {"ALL"}
    st.session_state["user_allow_all"] = True

ROL = st.session_state["user_rol"]

if ROL == "DG":
    vista = "Dirección General"
    carrera = None
else:
    vista = "Director de carrera"
    carrera = None

# ============================================================
# HEADER
# ============================================================
st.title("🎓 Dirección Académica")
st.caption("Ecosistema académico")
st.divider()

# ============================================================
# MENÚ
# ============================================================
with st.sidebar:
    st.markdown("### Navegación")

    SECCIONES = [
        "Encuesta de calidad",
        "Observación de clases",
        "Evaluación docente",
        "Capacitación Docente",
        "Índice de reprobación",
        "Exámenes departamentales",
        "Aulas virtuales",
        "Bajas / Retención",
        "Seguimiento de Inscripciones"
    ]

    seccion = st.selectbox("Selecciona módulo", SECCIONES)

# ============================================================
# ROUTER
# ============================================================

if seccion == "Encuesta de calidad":
    encuesta_calidad.render_encuesta_calidad(vista=vista, carrera=carrera)

elif seccion == "Observación de clases":
    observacion_clases.render_observacion_clases(vista=vista, carrera=carrera)

elif seccion == "Evaluación docente":
    evaluacion_docente.render_evaluacion_docente(vista=vista, carrera=carrera)

# 🔥 MÓDULO CLAVE (CORREGIDO)
elif seccion == "Capacitación Docente":

    if not HAS_CAPACITACION_MOD:
        st.error("❌ No se pudo importar el módulo")

        with st.expander("Ver error real"):
            st.exception(ERROR_CAP)

    else:
        if hasattr(capacitacion_docente, "render_capacitacion_docente"):
            capacitacion_docente.render_capacitacion_docente(
                vista=vista,
                carrera=carrera
            )

        elif hasattr(capacitacion_docente, "mostrar_modulo_capacitacion_docente"):
            capacitacion_docente.mostrar_modulo_capacitacion_docente()

        else:
            st.error("❌ El archivo no tiene función válida")

elif seccion == "Índice de reprobación":
    indice_reprobacion.render_indice_reprobacion(vista=vista, carrera=carrera)

elif seccion == "Exámenes departamentales":
    render_examenes_departamentales(
        "https://docs.google.com/spreadsheets/d/1GqlE9SOkSNCdA9mi65hk45uuLAao8GHHoresiyhRfQU/edit",
        vista=vista,
        carrera=carrera
    )

elif seccion == "Aulas virtuales":
    aulas_virtuales.mostrar(vista=vista, carrera=carrera)

elif seccion == "Bajas / Retención":
    if HAS_BAJAS_MOD:
        bajas_retencion.render_bajas_retencion(vista=vista, carrera=carrera)
    else:
        st.warning("Módulo no disponible")

elif seccion == "Seguimiento de Inscripciones":
    if HAS_SEGUIMIENTO_INS_MOD:
        seguimiento_inscripciones.render_seguimiento_inscripciones(
            vista=vista,
            carrera=carrera
        )
    else:
        st.warning("Módulo no disponible")
