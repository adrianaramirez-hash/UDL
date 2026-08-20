import re
import unicodedata

import pandas as pd

from backend.SERVICES.observation.observation_query import (
    buscar_docentes,
    filtrar_observaciones,
    listar_materias,
    listar_servicios,
    obtener_resumen_consulta,
)
from backend.SERVICES.observation.observation_teacher_detail import (
    obtener_detalle_docente,
)


def _normalizar(valor) -> str:
    if valor is None:
        return ""

    texto = str(valor).strip().lower()
    texto = " ".join(texto.split())
    texto = "".join(
        caracter
        for caracter in unicodedata.normalize("NFD", texto)
        if unicodedata.category(caracter) != "Mn"
    )
    return texto


def _normalizar_para_busqueda(valor) -> str:
    texto = _normalizar(valor)
    texto = re.sub(r"[^a-z0-9]+", " ", texto)
    return " ".join(texto.split())


def _buscar_columna(
    df: pd.DataFrame,
    opciones: list[str],
) -> str | None:
    columnas = {
        _normalizar(columna): columna
        for columna in df.columns
    }

    for opcion in opciones:
        clave = _normalizar(opcion)
        if clave in columnas:
            return columnas[clave]

    return None


def _detectar_valor_columna(
    df: pd.DataFrame,
    pregunta: str,
    opciones_columna: list[str],
) -> str | None:
    columna = _buscar_columna(df, opciones_columna)

    if columna is None:
        return None

    pregunta_normalizada = _normalizar_para_busqueda(pregunta)

    valores = (
        df[columna]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda serie: serie != ""]
        .drop_duplicates()
        .tolist()
    )

    valores = sorted(
        valores,
        key=lambda valor: len(_normalizar_para_busqueda(valor)),
        reverse=True,
    )

    for valor in valores:
        valor_normalizado = _normalizar_para_busqueda(valor)
        if valor_normalizado and valor_normalizado in pregunta_normalizada:
            return str(valor).strip()

    return None


def _detectar_clasificacion(
    pregunta: str,
) -> str | None:
    texto = _normalizar_para_busqueda(pregunta)

    if "no consolidado" in texto:
        return "No consolidado"
    if "en proceso" in texto:
        return "En proceso"
    if "consolidado" in texto:
        return "Consolidado"

    return None


def _extraer_busqueda_docente(
    pregunta: str,
) -> str | None:
    texto = _normalizar(pregunta)

    patrones = [
        r"busca(?:me)?\s+a\s+(.+)",
        r"buscar\s+a\s+(.+)",
        r"reporte\s+de\s+(.+)",
        r"resumen\s+de\s+(.+)",
        r"situacion\s+de\s+(.+)",
        r"informacion\s+de\s+(.+)",
        r"detalle\s+de\s+(.+)",
        r"docente\s+(.+)",
        r"profesor(?:a)?\s+(.+)",
        r"maestro(?:a)?\s+(.+)",
    ]

    for patron in patrones:
        coincidencia = re.search(patron, texto)

        if coincidencia:
            resultado = coincidencia.group(1)
            resultado = re.split(
                r"\s+(?:en|del|durante|para|con)\s+",
                resultado,
                maxsplit=1,
            )[0]
            return resultado.strip()

    return None


def _detectar_docente(
    df: pd.DataFrame,
    pregunta: str,
) -> tuple[str | None, list[dict]]:
    docente_directo = _detectar_valor_columna(
        df,
        pregunta,
        ["Nombre del docente"],
    )

    if docente_directo:
        return (
            docente_directo,
            buscar_docentes(df, docente_directo),
        )

    busqueda = _extraer_busqueda_docente(pregunta)

    if not busqueda:
        return None, []

    coincidencias = buscar_docentes(df, busqueda)

    if len(coincidencias) == 1:
        return (
            coincidencias[0]["docente"],
            coincidencias,
        )

    return None, coincidencias


def _es_pregunta_de_seguimiento_docente(
    pregunta: str,
) -> bool:
    texto = _normalizar_para_busqueda(pregunta)

    expresiones = [
        "que fortalezas",
        "cuales fortalezas",
        "fortalezas tiene",
        "sus fortalezas",
        "areas de oportunidad",
        "area de oportunidad",
        "que recomendacion",
        "que recomendaciones",
        "recomendacion recibio",
        "recomendaciones recibio",
        "sus recomendaciones",
        "en que materia",
        "que materia",
        "en cual materia",
        "en que asignatura",
        "que asignatura",
        "en que corte",
        "que corte",
        "en que grupo",
        "que grupo",
        "que clasificacion",
        "cual clasificacion",
        "cual fue su clasificacion",
        "que puntaje",
        "cual fue su puntaje",
        "cuanto obtuvo",
        "cual fue su promedio",
        "que promedio",
        "cuantas observaciones tiene",
        "cuantas observaciones",
        "ultima observacion",
        "observacion mas reciente",
        "cuando fue observado",
        "cuando lo observaron",
        "cuando la observaron",
        "que tipo de observacion",
        "cual fue el tipo",
    ]

    return any(expresion in texto for expresion in expresiones)


def _actualizar_contexto(
    contexto: dict | None,
    docente: str | None = None,
    servicio: str | None = None,
    materia: str | None = None,
    corte: str | None = None,
    tipo: str | None = None,
    clasificacion: str | None = None,
) -> dict:
    contexto_actualizado = dict(contexto or {})

    if docente:
        contexto_actualizado["ultimo_docente"] = docente
    if servicio:
        contexto_actualizado["ultimo_servicio"] = servicio
    if materia:
        contexto_actualizado["ultima_materia"] = materia
    if corte:
        contexto_actualizado["ultimo_corte"] = corte
    if tipo:
        contexto_actualizado["ultimo_tipo"] = tipo
    if clasificacion:
        contexto_actualizado["ultima_clasificacion"] = clasificacion

    return contexto_actualizado


def _porcentaje(
    cantidad: int,
    total: int,
) -> float:
    if total == 0:
        return 0.0
    return round(cantidad * 100 / total, 1)


def _crear_respuesta_resumen_ejecutiva(
    resumen: dict,
    contexto_texto: str | None = None,
) -> str:
    observaciones = int(resumen.get("observaciones", 0))

    if observaciones == 0:
        return (
            "No encontré observaciones que coincidan "
            "con los criterios solicitados."
        )

    promedio = resumen.get("promedio")
    consolidado = int(resumen.get("consolidado", 0))
    en_proceso = int(resumen.get("en_proceso", 0))
    no_consolidado = int(resumen.get("no_consolidado", 0))
    pct_consolidado = _porcentaje(consolidado, observaciones)

    inicio = (
        f"Durante el periodo analizado se registran "
        f"{observaciones} observaciones"
    )

    if contexto_texto:
        inicio += f" {contexto_texto}"

    inicio += "."

    partes = [inicio]

    if promedio is not None:
        partes.append(
            f"El promedio es de {float(promedio):.1f} puntos."
        )

    if pct_consolidado >= 85:
        partes.append(
            f"El desempeño general es favorable: "
            f"{pct_consolidado:.1f}% de las observaciones "
            f"se encuentra en nivel Consolidado."
        )
    elif pct_consolidado >= 70:
        partes.append(
            f"El desempeño general es aceptable, con "
            f"{pct_consolidado:.1f}% de las observaciones "
            f"en nivel Consolidado."
        )
    else:
        partes.append(
            f"El desempeño requiere atención: únicamente "
            f"{pct_consolidado:.1f}% de las observaciones "
            f"se encuentra en nivel Consolidado."
        )

    if no_consolidado > 0:
        termino = "caso" if no_consolidado == 1 else "casos"
        partes.append(
            f"Se identifican {no_consolidado} {termino} "
            f"No consolidado que requieren atención prioritaria."
        )

    if en_proceso > 0:
        partes.append(
            f"Además, permanecen {en_proceso} observaciones "
            f"En proceso que requieren seguimiento."
        )

    if no_consolidado == 0 and en_proceso == 0:
        partes.append(
            "No se identifican casos que requieran "
            "seguimiento prioritario."
        )

    return " ".join(partes)


def _respuesta_docente(
    df: pd.DataFrame,
    docente: str,
    contexto: dict | None = None,
) -> dict:
    detalle = obtener_detalle_docente(df, docente)
    contexto_actualizado = _actualizar_contexto(
        contexto,
        docente=docente,
    )

    if detalle["observaciones"] == 0:
        return {
            "respuesta": (
                f"No encontré observaciones registradas "
                f"para {docente}."
            ),
            "tipo_respuesta": "docente",
            "filtros": {"docente": docente},
            "datos": detalle,
            "contexto": contexto_actualizado,
        }

    observaciones = detalle["observaciones"]
    termino = "observación" if observaciones == 1 else "observaciones"

    respuesta = (
        f"{detalle['docente']} registra "
        f"{observaciones} {termino}, con un promedio de "
        f"{detalle['promedio']:.1f} puntos y clasificación "
        f"{detalle['clasificacion']}."
    )

    if detalle["historial"]:
        ultima = detalle["historial"][-1]
        respuesta += (
            f" La observación más reciente corresponde a "
            f"{ultima['corte']}, en {ultima['servicio']}, "
            f"para la asignatura {ultima['asignatura']}."
        )

    if detalle["areas_oportunidad"]:
        respuesta += (
            " Como principal área de oportunidad se registró: "
            f"{detalle['areas_oportunidad'][0]}."
        )

    if detalle["recomendaciones"]:
        respuesta += (
            " La recomendación registrada es: "
            f"{detalle['recomendaciones'][0]}."
        )

    return {
        "respuesta": respuesta,
        "tipo_respuesta": "docente",
        "filtros": {"docente": docente},
        "datos": detalle,
        "contexto": contexto_actualizado,
    }


def _responder_seguimiento_docente(
    df: pd.DataFrame,
    pregunta: str,
    docente: str,
    contexto: dict | None = None,
) -> dict | None:
    texto = _normalizar_para_busqueda(pregunta)
    detalle = obtener_detalle_docente(df, docente)

    contexto_actualizado = _actualizar_contexto(
        contexto,
        docente=docente,
    )

    if detalle["observaciones"] == 0:
        return {
            "respuesta": (
                f"No encontré observaciones registradas "
                f"para {docente}."
            ),
            "tipo_respuesta": "docente",
            "filtros": {"docente": docente},
            "datos": detalle,
            "contexto": contexto_actualizado,
        }

    historial = detalle.get("historial", [])
    ultima = historial[-1] if historial else None

    if "fortaleza" in texto or "fortalezas" in texto:
        fortalezas = detalle.get("fortalezas", [])

        if fortalezas:
            respuesta = (
                f"Las fortalezas registradas para "
                f"{detalle['docente']} son: "
                + "; ".join(fortalezas)
                + "."
            )
        else:
            respuesta = (
                f"No hay fortalezas registradas para "
                f"{detalle['docente']}."
            )

        return {
            "respuesta": respuesta,
            "tipo_respuesta": "fortalezas_docente",
            "filtros": {"docente": docente},
            "datos": {"fortalezas": fortalezas},
            "contexto": contexto_actualizado,
        }

    if (
        "area de oportunidad" in texto
        or "areas de oportunidad" in texto
        or "oportunidad" in texto
    ):
        areas = detalle.get("areas_oportunidad", [])

        if areas:
            respuesta = (
                f"Las áreas de oportunidad registradas para "
                f"{detalle['docente']} son: "
                + "; ".join(areas)
                + "."
            )
        else:
            respuesta = (
                f"No hay áreas de oportunidad registradas para "
                f"{detalle['docente']}."
            )

        return {
            "respuesta": respuesta,
            "tipo_respuesta": "areas_oportunidad_docente",
            "filtros": {"docente": docente},
            "datos": {"areas_oportunidad": areas},
            "contexto": contexto_actualizado,
        }

    if "recomendacion" in texto or "recomendaciones" in texto:
        recomendaciones = detalle.get("recomendaciones", [])

        if recomendaciones:
            respuesta = (
                f"Las recomendaciones registradas para "
                f"{detalle['docente']} son: "
                + "; ".join(recomendaciones)
                + "."
            )
        else:
            respuesta = (
                f"No hay recomendaciones registradas para "
                f"{detalle['docente']}."
            )

        return {
            "respuesta": respuesta,
            "tipo_respuesta": "recomendaciones_docente",
            "filtros": {"docente": docente},
            "datos": {"recomendaciones": recomendaciones},
            "contexto": contexto_actualizado,
        }

    if (
        "ultima observacion" in texto
        or "observacion mas reciente" in texto
    ):
        if ultima:
            respuesta = (
                f"La observación más reciente de "
                f"{detalle['docente']} corresponde a "
                f"{ultima['corte']}, en la asignatura "
                f"{ultima['asignatura']}, con "
                f"{ultima['puntaje']:.1f} puntos y clasificación "
                f"{ultima['clasificacion']}."
            )
        else:
            respuesta = (
                f"No hay historial de observaciones disponible "
                f"para {detalle['docente']}."
            )

        return {
            "respuesta": respuesta,
            "tipo_respuesta": "ultima_observacion_docente",
            "filtros": {"docente": docente},
            "datos": {"ultima_observacion": ultima},
            "contexto": contexto_actualizado,
        }

    if "materia" in texto or "asignatura" in texto:
        if ultima:
            respuesta = (
                f"{detalle['docente']} fue observado en la asignatura "
                f"{ultima['asignatura']}."
            )
        else:
            respuesta = (
                f"No hay una asignatura registrada para "
                f"{detalle['docente']}."
            )

        return {
            "respuesta": respuesta,
            "tipo_respuesta": "materia_docente",
            "filtros": {"docente": docente},
            "datos": {
                "asignatura": ultima.get("asignatura") if ultima else None
            },
            "contexto": contexto_actualizado,
        }

    if "corte" in texto:
        if ultima:
            respuesta = (
                f"La observación más reciente de "
                f"{detalle['docente']} corresponde al corte "
                f"{ultima['corte']}."
            )
        else:
            respuesta = (
                f"No hay un corte registrado para "
                f"{detalle['docente']}."
            )

        return {
            "respuesta": respuesta,
            "tipo_respuesta": "corte_docente",
            "filtros": {"docente": docente},
            "datos": {"corte": ultima.get("corte") if ultima else None},
            "contexto": contexto_actualizado,
        }

    if "grupo" in texto:
        if ultima:
            respuesta = (
                f"El grupo registrado en la observación más reciente de "
                f"{detalle['docente']} es {ultima['grupo']}."
            )
        else:
            respuesta = (
                f"No hay un grupo registrado para "
                f"{detalle['docente']}."
            )

        return {
            "respuesta": respuesta,
            "tipo_respuesta": "grupo_docente",
            "filtros": {"docente": docente},
            "datos": {"grupo": ultima.get("grupo") if ultima else None},
            "contexto": contexto_actualizado,
        }

    if "clasificacion" in texto:
        return {
            "respuesta": (
                f"La clasificación actual de "
                f"{detalle['docente']} es "
                f"{detalle['clasificacion']}."
            ),
            "tipo_respuesta": "clasificacion_docente",
            "filtros": {"docente": docente},
            "datos": {"clasificacion": detalle["clasificacion"]},
            "contexto": contexto_actualizado,
        }

    if (
        "puntaje" in texto
        or "cuanto obtuvo" in texto
        or "puntos obtuvo" in texto
    ):
        if ultima:
            respuesta = (
                f"En su observación más reciente, "
                f"{detalle['docente']} obtuvo "
                f"{ultima['puntaje']:.1f} puntos."
            )
        else:
            respuesta = (
                f"No hay un puntaje disponible para "
                f"{detalle['docente']}."
            )

        return {
            "respuesta": respuesta,
            "tipo_respuesta": "puntaje_docente",
            "filtros": {"docente": docente},
            "datos": {"puntaje": ultima.get("puntaje") if ultima else None},
            "contexto": contexto_actualizado,
        }

    if "promedio" in texto:
        return {
            "respuesta": (
                f"El promedio de {detalle['docente']} es de "
                f"{detalle['promedio']:.1f} puntos, considerando "
                f"{detalle['observaciones']} observaciones."
            ),
            "tipo_respuesta": "promedio_docente",
            "filtros": {"docente": docente},
            "datos": {
                "promedio": detalle["promedio"],
                "observaciones": detalle["observaciones"],
            },
            "contexto": contexto_actualizado,
        }

    if (
        "cuantas observaciones" in texto
        or "numero de observaciones" in texto
    ):
        return {
            "respuesta": (
                f"{detalle['docente']} registra "
                f"{detalle['observaciones']} observación"
                f"{'' if detalle['observaciones'] == 1 else 'es'}."
            ),
            "tipo_respuesta": "observaciones_docente",
            "filtros": {"docente": docente},
            "datos": {"observaciones": detalle["observaciones"]},
            "contexto": contexto_actualizado,
        }

    if (
        "tipo de observacion" in texto
        or "cual fue el tipo" in texto
    ):
        if ultima:
            respuesta = (
                f"El tipo de la observación más reciente de "
                f"{detalle['docente']} fue "
                f"{ultima['tipo']}."
            )
        else:
            respuesta = (
                f"No hay un tipo de observación disponible para "
                f"{detalle['docente']}."
            )

        return {
            "respuesta": respuesta,
            "tipo_respuesta": "tipo_observacion_docente",
            "filtros": {"docente": docente},
            "datos": {"tipo": ultima.get("tipo") if ultima else None},
            "contexto": contexto_actualizado,
        }

    if (
        "cuando fue observado" in texto
        or "cuando lo observaron" in texto
        or "cuando la observaron" in texto
    ):
        if ultima:
            respuesta = (
                f"La observación más reciente de "
                f"{detalle['docente']} corresponde al corte "
                f"{ultima['corte']}."
            )
        else:
            respuesta = (
                f"No hay información temporal disponible para "
                f"{detalle['docente']}."
            )

        return {
            "respuesta": respuesta,
            "tipo_respuesta": "fecha_docente",
            "filtros": {"docente": docente},
            "datos": {"corte": ultima.get("corte") if ultima else None},
            "contexto": contexto_actualizado,
        }

    return None


def responder_pregunta_observacion(
    df: pd.DataFrame,
    pregunta: str,
    contexto: dict | None = None,
) -> dict:
    pregunta = str(pregunta or "").strip()
    contexto = dict(contexto or {})

    if not pregunta:
        return {
            "respuesta": "Escribe una pregunta sobre Observación de Clases.",
            "tipo_respuesta": "error",
            "filtros": {},
            "datos": {},
            "contexto": contexto,
        }

    texto = _normalizar_para_busqueda(pregunta)

    servicio = _detectar_valor_columna(
        df,
        pregunta,
        ["Indica el servicio"],
    )

    materia = _detectar_valor_columna(
        df,
        pregunta,
        ["Asignatura", "Materia"],
    )

    corte = _detectar_valor_columna(
        df,
        pregunta,
        ["Corte"],
    )

    tipo = _detectar_valor_columna(
        df,
        pregunta,
        ["Tipo de observación"],
    )

    clasificacion = _detectar_clasificacion(
        pregunta
    )

    docente, coincidencias_docente = _detectar_docente(
        df,
        pregunta,
    )

    if (
        docente is None
        and _es_pregunta_de_seguimiento_docente(pregunta)
    ):
        docente_contexto = contexto.get("ultimo_docente")
        if docente_contexto:
            docente = str(docente_contexto).strip()

    if (
        docente is None
        and len(coincidencias_docente) > 1
    ):
        nombres = [
            item["docente"]
            for item in coincidencias_docente[:5]
        ]

        return {
            "respuesta": (
                "Encontré más de un docente que coincide "
                "con la búsqueda: "
                + ", ".join(nombres)
                + ". Indícame cuál deseas consultar."
            ),
            "tipo_respuesta": "docentes_coincidentes",
            "filtros": {},
            "datos": {"coincidencias": coincidencias_docente},
            "contexto": contexto,
        }

    if (
        docente
        and _es_pregunta_de_seguimiento_docente(pregunta)
    ):
        respuesta_seguimiento = _responder_seguimiento_docente(
            df=df,
            pregunta=pregunta,
            docente=docente,
            contexto=contexto,
        )

        if respuesta_seguimiento is not None:
            return respuesta_seguimiento

    palabras_docente = [
        "busca",
        "buscar",
        "reporte",
        "resumen",
        "situacion",
        "informacion",
        "detalle",
        "docente",
        "profesor",
        "profesora",
        "maestro",
        "maestra",
    ]

    if (
        docente
        and any(
            palabra in texto
            for palabra in palabras_docente
        )
    ):
        df_docente = filtrar_observaciones(
            df=df,
            docente=docente,
            servicio=servicio,
            materia=materia,
            corte=corte,
            tipo=tipo,
            clasificacion=clasificacion,
        )

        contexto_actualizado = _actualizar_contexto(
            contexto,
            docente=docente,
            servicio=servicio,
            materia=materia,
            corte=corte,
            tipo=tipo,
            clasificacion=clasificacion,
        )

        return _respuesta_docente(
            df=df_docente,
            docente=docente,
            contexto=contexto_actualizado,
        )

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

    filtros = {
        "docente": docente,
        "servicio": servicio,
        "materia": materia,
        "corte": corte,
        "tipo": tipo,
        "clasificacion": clasificacion,
    }

    contexto_actualizado = _actualizar_contexto(
        contexto,
        docente=docente,
        servicio=servicio,
        materia=materia,
        corte=corte,
        tipo=tipo,
        clasificacion=clasificacion,
    )

    if (
        "materias" in texto
        or "asignaturas" in texto
        or "que materia" in texto
        or "cuales materia" in texto
    ):
        materias = listar_materias(
            filtrado
        )

        if not materias:
            return {
                "respuesta": (
                    "No encontré materias asociadas "
                    "a los criterios solicitados."
                ),
                "tipo_respuesta": "materias",
                "filtros": filtros,
                "datos": {"materias": []},
                "contexto": contexto_actualizado,
            }

        nombres = [
            f"{item['materia']} ({item['observaciones']})"
            for item in materias[:15]
        ]

        return {
            "respuesta": (
                f"Se identificaron {len(materias)} materias "
                f"en el conjunto analizado. "
                "Entre las registradas se encuentran: "
                + ", ".join(nombres)
                + "."
            ),
            "tipo_respuesta": "materias",
            "filtros": filtros,
            "datos": {"materias": materias},
            "contexto": contexto_actualizado,
        }

    if (
        "servicios" in texto
        or "carreras" in texto
        or "que carrera" in texto
        or "cuales carrera" in texto
    ):
        servicios = listar_servicios(
            filtrado
        )

        if not servicios:
            return {
                "respuesta": (
                    "No encontré servicios o carreras "
                    "para los criterios solicitados."
                ),
                "tipo_respuesta": "servicios",
                "filtros": filtros,
                "datos": {"servicios": []},
                "contexto": contexto_actualizado,
            }

        nombres = [
            f"{item['servicio']} ({item['observaciones']})"
            for item in servicios[:15]
        ]

        return {
            "respuesta": (
                "Los servicios identificados son: "
                + ", ".join(nombres)
                + "."
            ),
            "tipo_respuesta": "servicios",
            "filtros": filtros,
            "datos": {"servicios": servicios},
            "contexto": contexto_actualizado,
        }

    if "promedio" in texto:
        if resumen["observaciones"] == 0:
            respuesta = (
                "No encontré observaciones para calcular "
                "el promedio solicitado."
            )
        elif resumen["promedio"] is None:
            respuesta = (
                "Encontré observaciones, pero no hay "
                "un promedio disponible."
            )
        else:
            respuesta = (
                f"El promedio del conjunto analizado es de "
                f"{resumen['promedio']:.1f} puntos, considerando "
                f"{resumen['observaciones']} observaciones."
            )

        return {
            "respuesta": respuesta,
            "tipo_respuesta": "promedio",
            "filtros": filtros,
            "datos": resumen,
            "contexto": contexto_actualizado,
        }

    if clasificacion:
        cantidad = 0

        if clasificacion == "Consolidado":
            cantidad = resumen["consolidado"]
        elif clasificacion == "En proceso":
            cantidad = resumen["en_proceso"]
        elif clasificacion == "No consolidado":
            cantidad = resumen["no_consolidado"]

        if (
            "cuantas" in texto
            or "cuantos" in texto
            or "cantidad" in texto
        ):
            return {
                "respuesta": (
                    f"Se registran {cantidad} observaciones "
                    f"clasificadas como {clasificacion} "
                    f"para los criterios solicitados."
                ),
                "tipo_respuesta": "cantidad_clasificacion",
                "filtros": filtros,
                "datos": {
                    "clasificacion": clasificacion,
                    "cantidad": cantidad,
                },
                "contexto": contexto_actualizado,
            }

    contexto_texto = None

    if materia:
        contexto_texto = f"de la materia {materia}"
    elif servicio:
        contexto_texto = f"en {servicio}"
    elif corte:
        contexto_texto = f"en el corte {corte}"
    elif tipo:
        contexto_texto = f"del tipo {tipo}"
    elif clasificacion:
        contexto_texto = f"clasificadas como {clasificacion}"

    respuesta = _crear_respuesta_resumen_ejecutiva(
        resumen,
        contexto_texto,
    )

    return {
        "respuesta": respuesta,
        "tipo_respuesta": "resumen",
        "filtros": filtros,
        "datos": resumen,
        "contexto": contexto_actualizado,
    }
