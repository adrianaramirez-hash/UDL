import pandas as pd
import streamlit as st
import altair as alt
import gspread
import re

# =====================================================
# CONFIG
# =====================================================
SHEET_NAME_DEFAULT = "REPROBACION"
UMBRAL_REPROBACION_DEFAULT = 70

COLOR_AZUL = "#2F80ED"
COLOR_ROJO = "#D
