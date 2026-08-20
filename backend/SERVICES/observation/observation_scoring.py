import pandas as pd


def respuesta_a_puntos(valor):
    """
    Convierte la respuesta del instrumento de observación a puntaje.

    Reglas:
    - Sí / Si / X -> 3 puntos
    - Sin evidencia -> 2 puntos
    - No -> 1 punto
    - Valores numéricos -> se convierten a float
    - Valores inválidos o vacíos -> None
    """
    if pd.isna(valor):
        return None

    texto = str(valor).strip().lower()

    if texto in ("sí", "si", "x"):
        return 3

    if "sin evidencia" in texto or "sin evidencias" in texto:
        return 2

    if texto == "no":
        return 1

    try:
        return float(texto)
    except ValueError:
        return None


def clasificar_por_puntos(total_puntos):
    """
    Clasifica una observación o promedio docente según el puntaje obtenido.

    Reglas:
    - Consolidado -> 97 puntos o más
    - En proceso -> 76 a 96 puntos
    - No consolidado -> 75 puntos o menos
    """
    if pd.isna(total_puntos):
        return ""

    if total_puntos >= 97:
        return "Consolidado"

    if total_puntos >= 76:
        return "En proceso"

    return "No consolidado"


def asignar_corte(fecha, df_cortes):
    """
    Asigna a una fecha el corte de 30 días correspondiente.

    df_cortes debe contener:
    - Corte
    - Fecha_inicio
    - Fecha_fin
    """
    if pd.isna(fecha) or df_cortes.empty:
        return "Sin corte"

    for _, fila in df_cortes.iterrows():
        fecha_inicio = fila.get("Fecha_inicio")
        fecha_fin = fila.get("Fecha_fin")

        if (
            pd.notna(fecha_inicio)
            and pd.notna(fecha_fin)
            and fecha_inicio <= fecha <= fecha_fin
        ):
            return str(fila.get("Corte"))

    return "Sin corte"