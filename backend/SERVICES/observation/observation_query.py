import unicodedata

import pandas as pd


def _normalizar_texto(valor) -> str:
    """
    Normaliza texto para búsquedas flexibles:
    - convierte a minúsculas
    - elimina espacios extra
    - elimina acentos
    """
    if valor is None:
        return ""

    texto = str(valor).strip().lower()
    texto = " ".join(texto.split())

    texto = "".join(
        caracter
        for caracter in unicodedata.normalize(
            "NFD",
            texto,
        )
        if unicodedata.category(caracter) != "Mn"
    )

    return texto


def _buscar_columna(
    df: pd.DataFrame,
    opciones: list[str],
) -> str | None:
    """
    Encuentra una columna aceptando pequeñas diferencias
    en espacios y acentos.
    """
    columnas_normalizadas = {
        _normalizar_texto(columna): columna
        for columna in df.columns
    }

    for opcion in opciones:
        clave = _normalizar_texto(opcion)

        if clave in columnas_normalizadas:
            return columnas_normalizadas[clave]

    return None


def filtrar_observaciones(
    df: pd.DataFrame,
    docente: str | None = None,
    servicio: str | None = None,
    materia: str | None = None,
    corte: str | None = None,
    tipo: str | None = None,
    clasificacion: str | None = None,
) -> pd.DataFrame:
    """
    Motor base de filtros para consultas del Asesor SIA.

    Coincidencia parcial:
    - docente
    - materia

    Coincidencia exacta:
    - servicio
    - corte
    - tipo
    - clasificación
    """

    resultado = df.copy()

    if resultado.empty:
        return resultado

    docente_col = _buscar_columna(
        resultado,
        ["Nombre del docente"],
    )

    servicio_col = _buscar_columna(
        resultado,
        ["Indica el servicio"],
    )

    materia_col = _buscar_columna(
        resultado,
        ["Asignatura", "Materia"],
    )

    corte_col = _buscar_columna(
        resultado,
        ["Corte"],
    )

    tipo_col = _buscar_columna(
        resultado,
        ["Tipo de observación"],
    )

    clasificacion_col = _buscar_columna(
        resultado,
        ["Clasificación_observación"],
    )

    # Docente: búsqueda parcial
    if docente and docente_col:
        busqueda = _normalizar_texto(docente)

        resultado = resultado[
            resultado[docente_col]
            .astype(str)
            .apply(_normalizar_texto)
            .str.contains(
                busqueda,
                regex=False,
                na=False,
            )
        ]

    # Servicio: coincidencia exacta
    if servicio and servicio_col:
        busqueda = _normalizar_texto(servicio)

        resultado = resultado[
            resultado[servicio_col]
            .astype(str)
            .apply(_normalizar_texto)
            == busqueda
        ]

    # Materia: búsqueda parcial
    if materia and materia_col:
        busqueda = _normalizar_texto(materia)

        resultado = resultado[
            resultado[materia_col]
            .astype(str)
            .apply(_normalizar_texto)
            .str.contains(
                busqueda,
                regex=False,
                na=False,
            )
        ]

    # Corte: coincidencia exacta
    if corte and corte_col:
        busqueda = _normalizar_texto(corte)

        resultado = resultado[
            resultado[corte_col]
            .astype(str)
            .apply(_normalizar_texto)
            == busqueda
        ]

    # Tipo: coincidencia exacta
    if tipo and tipo_col:
        busqueda = _normalizar_texto(tipo)

        resultado = resultado[
            resultado[tipo_col]
            .astype(str)
            .apply(_normalizar_texto)
            == busqueda
        ]

    # Clasificación: coincidencia exacta
    if clasificacion and clasificacion_col:
        busqueda = _normalizar_texto(
            clasificacion
        )

        resultado = resultado[
            resultado[clasificacion_col]
            .astype(str)
            .apply(_normalizar_texto)
            == busqueda
        ]

    return resultado.copy()


def obtener_resumen_consulta(
    df: pd.DataFrame,
) -> dict:
    """
    Resume el resultado de cualquier consulta.
    """

    if df.empty:
        return {
            "observaciones": 0,
            "promedio": None,
            "consolidado": 0,
            "en_proceso": 0,
            "no_consolidado": 0,
        }

    clasificacion_col = _buscar_columna(
        df,
        ["Clasificación_observación"],
    )

    puntaje_col = _buscar_columna(
        df,
        ["Total_puntos_observación"],
    )

    consolidado = 0
    en_proceso = 0
    no_consolidado = 0

    if clasificacion_col:
        clasificaciones = (
            df[clasificacion_col]
            .astype(str)
            .apply(_normalizar_texto)
        )

        consolidado = int(
            (
                clasificaciones
                == _normalizar_texto("Consolidado")
            ).sum()
        )

        en_proceso = int(
            (
                clasificaciones
                == _normalizar_texto("En proceso")
            ).sum()
        )

        no_consolidado = int(
            (
                clasificaciones
                == _normalizar_texto("No consolidado")
            ).sum()
        )

    promedio = None

    if puntaje_col:
        valores = pd.to_numeric(
            df[puntaje_col],
            errors="coerce",
        ).dropna()

        if not valores.empty:
            promedio = float(
                round(
                    valores.mean(),
                    1,
                )
            )

    return {
        "observaciones": int(len(df)),
        "promedio": promedio,
        "consolidado": consolidado,
        "en_proceso": en_proceso,
        "no_consolidado": no_consolidado,
    }


def buscar_docentes(
    df: pd.DataFrame,
    nombre: str,
) -> list[dict]:
    """
    Busca docentes por coincidencia parcial del nombre.
    """

    docente_col = _buscar_columna(
        df,
        ["Nombre del docente"],
    )

    if not docente_col or not nombre.strip():
        return []

    busqueda = _normalizar_texto(nombre)

    coincidencias = df[
        df[docente_col]
        .astype(str)
        .apply(_normalizar_texto)
        .str.contains(
            busqueda,
            regex=False,
            na=False,
        )
    ].copy()

    if coincidencias.empty:
        return []

    servicio_col = _buscar_columna(
        coincidencias,
        ["Indica el servicio"],
    )

    resultado = []

    for docente, grupo in coincidencias.groupby(
        docente_col
    ):
        servicios = []

        if servicio_col:
            servicios = (
                grupo[servicio_col]
                .dropna()
                .astype(str)
                .str.strip()
                .loc[lambda serie: serie != ""]
                .drop_duplicates()
                .tolist()
            )

        resultado.append(
            {
                "docente": str(docente).strip(),
                "observaciones": int(len(grupo)),
                "servicios": servicios,
            }
        )

    return resultado


def listar_materias(
    df: pd.DataFrame,
) -> list[dict]:
    """
    Devuelve las asignaturas y su número de observaciones.
    """

    materia_col = _buscar_columna(
        df,
        ["Asignatura", "Materia"],
    )

    if not materia_col or df.empty:
        return []

    conteo = (
        df[materia_col]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda serie: serie != ""]
        .value_counts()
    )

    return [
        {
            "materia": materia,
            "observaciones": int(cantidad),
        }
        for materia, cantidad in conteo.items()
    ]


def listar_servicios(
    df: pd.DataFrame,
) -> list[dict]:
    """
    Devuelve servicios/carreras y número de observaciones.
    """

    servicio_col = _buscar_columna(
        df,
        ["Indica el servicio"],
    )

    if not servicio_col or df.empty:
        return []

    conteo = (
        df[servicio_col]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda serie: serie != ""]
        .value_counts()
    )

    return [
        {
            "servicio": servicio,
            "observaciones": int(cantidad),
        }
        for servicio, cantidad in conteo.items()
    ]


def obtener_contexto_consulta(
    df: pd.DataFrame,
    docente: str | None = None,
    servicio: str | None = None,
    materia: str | None = None,
    corte: str | None = None,
    tipo: str | None = None,
    clasificacion: str | None = None,
) -> dict:
    """
    Función principal para entregar información estructurada
    al futuro Asesor SIA conversacional.

    La IA deberá redactar a partir de este resultado,
    no calcular directamente sobre los datos.
    """

    filtrado = filtrar_observaciones(
        df=df,
        docente=docente,
        servicio=servicio,
        materia=materia,
        corte=corte,
        tipo=tipo,
        clasificacion=clasificacion,
    )

    resumen = obtener_resumen_consulta(
        filtrado
    )

    return {
        "filtros": {
            "docente": docente,
            "servicio": servicio,
            "materia": materia,
            "corte": corte,
            "tipo": tipo,
            "clasificacion": clasificacion,
        },
        "resumen": resumen,
        "materias": listar_materias(
            filtrado
        )[:20],
        "servicios": listar_servicios(
            filtrado
        )[:20],
    }