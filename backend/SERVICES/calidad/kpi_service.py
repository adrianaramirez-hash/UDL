from collections import defaultdict
from statistics import mean
import unicodedata
from backend.SERVICES.calidad.normalizer import cargar_personas_evaluadas


def _es_si(valor) -> bool:
    return str(valor or "").strip().upper() in {
        "SÍ",
        "SI",
    }


def _numero(valor):
    if valor in ("", None):
        return None

    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


def _promedio(valores):
    valores_validos = [
        valor
        for valor in valores
        if valor is not None
    ]

    if not valores_validos:
        return None

    return round(
        mean(valores_validos),
        2,
    )


def _distribucion_perfil(
    encuestas: list[dict],
    campo: str,
) -> list[dict]:
    conteos = defaultdict(
        int
    )

    for encuesta in encuestas:
        valor = " ".join(
            str(
                encuesta.get(
                    campo,
                    "",
                )
                or ""
            ).strip().split()
        )

        if valor:
            conteos[valor] += 1

    total_encuestas = len(
        encuestas
    )

    con_dato = sum(
        conteos.values()
    )

    sin_dato = (
        total_encuestas
        - con_dato
    )

    distribucion = [
        {
            "valor": valor,
            "cantidad": cantidad,
            "porcentaje": round(
                cantidad
                / con_dato
                * 100,
                2,
            ),
        }
        for valor, cantidad
        in conteos.items()
    ] if con_dato else []

    distribucion = sorted(
        distribucion,
        key=lambda item: (
            -item["cantidad"],
            item["valor"].casefold(),
        ),
    )

    return {
        "total_encuestas": total_encuestas,
        "con_dato": con_dato,
        "sin_dato": sin_dato,
        "cobertura_pct": (
            round(
                con_dato
                / total_encuestas
                * 100,
                2,
            )
            if total_encuestas
            else 0.0
        ),
        "distribucion": distribucion,
    }


def _normalizar_texto_cualitativo(valor) -> str:
    import unicodedata

    texto = " ".join(
        str(valor or "").strip().casefold().split()
    )

    return "".join(
        caracter
        for caracter in unicodedata.normalize(
            "NFD",
            texto,
        )
        if unicodedata.category(caracter) != "Mn"
    )


def _es_texto_cualitativo_informativo(valor) -> bool:
    texto = _normalizar_texto_cualitativo(
        valor
    )

    if not texto:
        return False

    respuestas_no_informativas = {
        "-",
        ".",
        "..",
        "...",
        "no",
        "nop",
        "nope",
        "na",
        "n/a",
        "ninguno",
        "ninguna",
        "ningun",
        "nada",
        "sin comentarios",
        "sin comentario",
        "sin sugerencias",
        "sin sugerencia",
        "no tengo",
        "no aplica",
        "no se",
        "nose",
        "mmm",
        "mm",
        "x",
    }

    if texto in respuestas_no_informativas:
        return False

    # Evita ruido de una sola letra o s?mbolos.
    caracteres_utiles = [
        caracter
        for caracter in texto
        if caracter.isalnum()
    ]

    return len(caracteres_utiles) >= 3


def _clasificar_campo_cualitativo(texto_corto: str) -> dict:
    texto = _normalizar_texto_cualitativo(
        texto_corto
    )

    if "atencion destacada" in texto:
        return {
            "categoria": "reconocimiento",
            "incluir_resumen": False,
            "polaridad_base": None,
        }

    if "aspectos positivos" in texto:
        return {
            "categoria": "comentario",
            "incluir_resumen": True,
            "polaridad_base": "positivo",
        }

    if "aspectos por mejorar" in texto:
        return {
            "categoria": "comentario",
            "incluir_resumen": True,
            "polaridad_base": "oportunidad",
        }

    if "incidencia" in texto:
        return {
            "categoria": "incidencia",
            "incluir_resumen": True,
            "polaridad_base": "negativo",
        }

    if "motivo de recomendacion" in texto:
        return {
            "categoria": "recomendacion",
            "incluir_resumen": True,
            "polaridad_base": None,
        }

    if (
        "comentario" in texto
        or "sugerencia" in texto
    ):
        return {
            "categoria": "comentario",
            "incluir_resumen": True,
            "polaridad_base": None,
        }

    return {
        "categoria": "otro_texto",
        "incluir_resumen": False,
        "polaridad_base": None,
    }


def calcular_kpis_preview(
    registros: list[dict],
    solo_comparables_historico: bool = False,
) -> dict:
    """
    Calcula una vista previa de KPIs.

    No escribe datos en el MASTER.
    Respeta:
    - promedio por pregunta;
    - igual peso de preguntas dentro de sección;
    - igual peso de secciones en el índice general.
    """

    filas_origen = {
        registro["FILA_ORIGEN"]
        for registro in registros
    }

    encuestas_por_fila = {}

    for registro in registros:
        fila_origen = registro.get(
            "FILA_ORIGEN"
        )

        if fila_origen not in encuestas_por_fila:
            encuestas_por_fila[
                fila_origen
            ] = registro

    encuestas_unicas = list(
        encuestas_por_fila.values()
    )

    perfil_respuestas = {
        "total_encuestas": len(
            encuestas_unicas
        ),
        "servicio_programa": _distribucion_perfil(
            encuestas_unicas,
            "SERVICIO_PROGRAMA",
        ),
        "modalidad_detalle": _distribucion_perfil(
            encuestas_unicas,
            "MODALIDAD_DETALLE",
        ),
        "grado": _distribucion_perfil(
            encuestas_unicas,
            "GRADO",
        ),
        "ciclo_escolar": _distribucion_perfil(
            encuestas_unicas,
            "CICLO_ESCOLAR",
        ),
        "turno": _distribucion_perfil(
            encuestas_unicas,
            "TURNO",
        ),
    }

    validos_kpi = [
        registro
        for registro in registros
        if _es_si(
            registro.get(
                "INCLUIR_EN_KPI",
                "",
            )
        )
        and _numero(
            registro.get(
                "INDICE_100"
            )
        )
        is not None
        and (
            not solo_comparables_historico
            or _es_si(
                registro.get(
                    "COMPARABLE_HISTORICO",
                    "",
                )
            )
        )
    ]

    por_pregunta = defaultdict(
            list
        )

    for registro in validos_kpi:
            por_pregunta[
                registro["PREGUNTA_ID"]
            ].append(
                registro
            )

    preguntas = []

    for pregunta_id, grupo in por_pregunta.items():
            indices = [
                _numero(
                    registro.get(
                        "INDICE_100"
                    )
                )
                for registro in grupo
            ]

            valores_base = [
                _numero(
                    registro.get(
                        "VALOR_BASE"
                    )
                )
                for registro in grupo
            ]

            valores_base = [
                valor
                for valor in valores_base
                if valor is not None
            ]

            n = len(indices)

            ejemplo = grupo[0]

            tipo_dato = str(
                ejemplo.get(
                    "TIPO_DATO",
                    "",
                )
            ).strip().upper()

            if tipo_dato == "BINARIA":
                favorables = sum(
                    1
                    for valor in indices
                    if valor == 100
                )

                neutrales = 0

                desfavorables = sum(
                    1
                    for valor in indices
                    if valor == 0
                )

            else:
                favorables = sum(
                    1
                    for valor in valores_base
                    if valor >= 4
                )

                neutrales = sum(
                    1
                    for valor in valores_base
                    if valor == 3
                )

                desfavorables = sum(
                    1
                    for valor in valores_base
                    if valor <= 2
                )

            preguntas.append(
                {
                    "pregunta_id": pregunta_id,
                    "seccion_id": ejemplo.get(
                        "SECCION_ID",
                        "",
                    ),
                    "seccion_nombre": ejemplo.get(
                        "SECCION_NOMBRE",
                        "",
                    ),
                    "texto_corto": ejemplo.get(
                        "TEXTO_CORTO",
                        "",
                    ),
                    "tipo_dato": ejemplo.get(
                        "TIPO_DATO",
                        "",
                    ),
                    "persona_evaluada": ejemplo.get(
                        "PERSONA_EVALUADA",
                        "",
                    ),
                    "rol_evaluado": ejemplo.get(
                        "ROL_EVALUADO",
                        "",
                    ),
                    "indice": _promedio(
                        indices
                    ),
                    "n": n,
                    "favorable_pct": round(
                        favorables / n * 100,
                        2,
                    )
                    if n
                    else None,
                    "neutral_pct": round(
                        neutrales / n * 100,
                        2,
                    )
                    if n
                    else None,
                    "desfavorable_pct": round(
                        desfavorables / n * 100,
                        2,
                    )
                    if n
                    else None,
                    "entra_seccion": _es_si(
                        ejemplo.get(
                            "ENTRA_PROMEDIO_SECCION",
                            "",
                        )
                    ),
                    "entra_general": _es_si(
                        ejemplo.get(
                            "ENTRA_PROMEDIO_GENERAL",
                            "",
                        )
                    ),
                }
            )

    preguntas_por_persona = defaultdict(
            list
        )

    filas_por_persona = defaultdict(
            set
        )

    for pregunta in preguntas:
            persona = str(
                pregunta.get(
                    "persona_evaluada",
                    "",
                )
                or ""
            ).strip()

            rol = str(
                pregunta.get(
                    "rol_evaluado",
                    "",
                )
                or ""
            ).strip()

            if (
                persona
                and pregunta.get("entra_seccion")
                and pregunta.get("indice") is not None
            ):
                clave = (
                    pregunta.get(
                        "seccion_id",
                        "",
                    ),
                    persona,
                    rol,
                )

                preguntas_por_persona[
                    clave
                ].append(
                    pregunta
                )

    for registro in validos_kpi:
            persona = str(
                registro.get(
                    "PERSONA_EVALUADA",
                    "",
                )
                or ""
            ).strip()

            rol = str(
                registro.get(
                    "ROL_EVALUADO",
                    "",
                )
                or ""
            ).strip()

            if persona:
                clave = (
                    registro.get(
                        "SECCION_ID",
                        "",
                    ),
                    persona,
                    rol,
                )

                filas_por_persona[
                    clave
                ].add(
                    registro.get(
                        "FILA_ORIGEN"
                    )
                )

    personas = []

    for clave, grupo in preguntas_por_persona.items():
            (
                seccion_id,
                persona,
                rol,
            ) = clave

            personas.append(
                {
                    "seccion_id": seccion_id,
                    "seccion_nombre": grupo[0].get(
                        "seccion_nombre",
                        "",
                    ),
                    "persona_evaluada": persona,
                    "rol_evaluado": rol,
                    "indice": _promedio(
                        [
                            pregunta["indice"]
                            for pregunta in grupo
                        ]
                    ),
                    "preguntas_con_datos": len(
                        grupo
                    ),
                    "evaluaciones": len(
                        filas_por_persona.get(
                            clave,
                            set(),
                        )
                    ),
                }
            )

        # -------------------------------------------------
        # Comparativos de personas evaluadas
        # Carrera -> persona -> rol
        # Conserva igual peso por pregunta.
        # -------------------------------------------------

        # -------------------------------------------------
        # Catalogo oficial de personas evaluadas
        # MASTER: 04_PERSONAS_EVALUADAS
        # -------------------------------------------------

    fuentes_personal = {
            str(
                registro.get(
                    "FUENTE_ID",
                    "",
                )
                or ""
            ).strip()
            for registro in registros
            if str(
                registro.get(
                    "FUENTE_ID",
                    "",
                )
                or ""
            ).strip()
        }

    anios_personal = {
            int(
                registro.get(
                    "ANIO"
                )
            )
            for registro in registros
            if registro.get(
                "ANIO"
            ) not in (
                "",
                None,
            )
        }

    catalogo_personal = []

    if (
            len(fuentes_personal) == 1
            and len(anios_personal) == 1
        ):
            catalogo_personal = (
                cargar_personas_evaluadas(
                    next(
                        iter(
                            fuentes_personal
                        )
                    ),
                    next(
                        iter(
                            anios_personal
                        )
                    ),
                )
            )

    def _clave_texto_personal(
        valor,
    ) -> str:
        return str(
            valor or ""
        ).strip().casefold()

    def _clave_personal_ascii(valor) -> str:
        texto = unicodedata.normalize(
            "NFD",
            str(valor or ""),
        )
        texto = "".join(
            caracter
            for caracter in texto
            if unicodedata.category(caracter) != "Mn"
        )
        return " ".join(
            texto.casefold().strip().split()
        )

    def _asignacion_personal_confirmada(
        carrera,
        modalidad,
        persona,
    ) -> bool:
        """
        Ajustes operativos confirmados para asistentes y responsables.

        Solo restringe a las personas listadas; el resto del catalogo
        institucional continua funcionando sin cambios.
        """
        nombre = _clave_personal_ascii(persona)
        carrera_key = _clave_personal_ascii(carrera)
        modalidad_key = _clave_personal_ascii(modalidad)

        reglas = {
            "balbina garcia arroyo": ["actuacion"],
            "militza fermoso": ["odontologia legal"],
            "grecia ceballos ibarra": [
                "administracion de empresas",
                "contaduria",
                "finanzas",
                "mercadotecnia",
            ],
            "iris salazar de lucio": [
                "diseno grafico",
                "comunicacion multimedia",
                "cine y tv digital",
                "cine y television digital",
            ],
            "sonia olvera": ["derecho"],
            "lili guerra lopez": ["diseno de modas"],
            "alejandro rodriguez": ["gastronomia"],
            "isabel galicia": ["tecnologias de la informacion"],
            "karla sanchez cruz": ["psicologia"],
            "oscar corona amezcua": ["desarrollo del potencial humano"],
            "lourdes quiroga": ["psicoterapia"],
        }

        if nombre == "claudia diaz":
            return (
                "posgrado" in modalidad_key
                or "maestr" in carrera_key
            )

        if nombre in {
            "alejandra romero pena",
            "brenda ponce perez",
        }:
            if "ejecut" not in modalidad_key:
                return False

            carreras_ejecutivas = [
                "administracion",
                "contaduria",
                "derecho",
                "informatica",
                "mercadotecnia",
                "pedagogia",
            ]
            return any(
                item in carrera_key
                for item in carreras_ejecutivas
            )

        permitidas = reglas.get(nombre)
        if not permitidas:
            return True

        if nombre in {
            "grecia ceballos ibarra",
            "sonia olvera",
        } and "ejecut" in modalidad_key:
            return False

        return any(
            item in carrera_key
            for item in permitidas
        )

    def _es_persona_evaluada_valida(
        valor,
    ) -> bool:
        texto = " ".join(
            str(valor or "").strip().split()
        )

        if not texto:
            return False

        clave = _clave_texto_personal(
            texto
        )

        # Evita que nombres de fuente, hoja o archivo
        # terminen interpretandose como una persona.
        if clave.startswith("@encuesta"):
            return False

        if (
            "encuesta de calidad" in clave
            and "respuestas" in clave
        ):
            return False

        return True

    def _rol_canonico_personal(
        rol,
    ) -> str:
        texto = _clave_texto_personal(
            rol
        )

        if (
            "equipo de licenciaturas ejecutivas"
            in texto
            and "coordinadora" in texto
        ):
            return "coordinadora"

        return texto

    def _clave_asignacion_personal(
        carrera,
        modalidad,
        persona,
        rol,
    ):
        return (
            _clave_texto_personal(
                carrera
            ),
            _clave_texto_personal(
                modalidad
            ),
            _clave_texto_personal(
                persona
            ),
            _rol_canonico_personal(
                rol
            ),
        )

    # Si el endpoint ya viene filtrado a un solo
    # servicio, reducimos tambien el catalogo a la
    # poblacion consultada.
    servicios_actuales = {
        _clave_texto_personal(
            registro.get(
                "SERVICIO_PROGRAMA",
                "",
            )
        )
        for registro in registros
        if _clave_texto_personal(
            registro.get(
                "SERVICIO_PROGRAMA",
                "",
            )
        )
    }

    poblaciones_actuales = {
        (
            _clave_texto_personal(
                registro.get(
                    "CARRERA_BASE",
                    "",
                )
                or registro.get(
                    "SERVICIO_PROGRAMA",
                    "",
                )
            ),
            _clave_texto_personal(
                registro.get(
                    "MODALIDAD_ACADEMICA",
                    "",
                )
                or registro.get(
                    "MODALIDAD_DETALLE",
                    "",
                )
            ),
        )
        for registro in registros
        if (
            registro.get(
                "CARRERA_BASE"
            )
            or registro.get(
                "SERVICIO_PROGRAMA"
            )
        )
    }

    if (
        len(servicios_actuales) == 1
        and poblaciones_actuales
    ):
        catalogo_personal = [
            fila
            for fila in catalogo_personal
            if (
                _clave_texto_personal(
                    fila.get(
                        "CARRERA_BASE",
                        "",
                    )
                ),
                _clave_texto_personal(
                    fila.get(
                        "MODALIDAD_ACADEMICA",
                        "",
                    )
                ),
            )
            in poblaciones_actuales
        ]

    # Conservamos un universo independiente de poblaciones antes de
    # corregir las asignaciones persona -> carrera.  Este universo se
    # usa para que TODAS las carreras/modalidades sigan apareciendo en
    # la vista de Asistentes, incluso cuando no tengan una persona
    # confirmada o no registren respuestas en el periodo.
    catalogo_poblaciones_personal = list(catalogo_personal)

    # Aplicar las asignaciones confirmadas sin alterar otros roles.
    roles_por_persona = {}
    for fila in catalogo_personal:
        nombre_key = _clave_personal_ascii(
            fila.get("PERSONA_EVALUADA", "")
        )
        rol_actual = str(
            fila.get("ROL_EVALUADO", "") or ""
        ).strip()
        if nombre_key and rol_actual and nombre_key not in roles_por_persona:
            roles_por_persona[nombre_key] = rol_actual

    catalogo_personal = [
        fila
        for fila in catalogo_personal
        if _asignacion_personal_confirmada(
            fila.get("CARRERA_BASE", ""),
            fila.get("MODALIDAD_ACADEMICA", ""),
            fila.get("PERSONA_EVALUADA", ""),
        )
    ]

    asignaciones_confirmadas = {
        "Balbina García Arroyo": (
            "Presencial",
            ["Actuación"],
            "Asistente",
        ),
        "Militza Fermoso": (
            "Posgrado",
            ["Maestría en Odontología Legal y Forense"],
            "Asistente",
        ),
        "Grecia Ceballos Ibarra": (
            "Presencial",
            [
                "Administración de Empresas",
                "Contaduría",
                "Finanzas",
                "Mercadotecnia",
            ],
            "Asistente",
        ),
        "Iris Salazar de Lucio": (
            "Presencial",
            [
                "Diseño Gráfico",
                "Comunicación Multimedia",
                "Cine y TV Digital",
            ],
            "Asistente",
        ),
        "Sonia Olvera": (
            "Presencial",
            ["Derecho"],
            "Asistente",
        ),
        "Lili Guerra López": (
            "Presencial",
            ["Diseño de Modas"],
            "Asistente",
        ),
        "Alejandro Rodríguez": (
            "Presencial",
            ["Gastronomía"],
            "Asistente",
        ),
        "Isabel Galicia": (
            "Presencial",
            ["Tecnologías de la Información"],
            "Asistente",
        ),
        "Karla Sánchez Cruz": (
            "Presencial",
            ["Psicología"],
            "Asistente",
        ),
        "Oscar Corona Amezcua": (
            "Posgrado",
            ["Maestría en Desarrollo del Potencial Humano y Organizacional"],
            "Asistente",
        ),
        "Lourdes Quiroga": (
            "Posgrado",
            ["Maestría en Psicoterapia"],
            "Asistente",
        ),
        "Alejandra Romero Peña": (
            "Ejecutiva",
            [
                "Administración de Empresas",
                "Contaduría",
                "Derecho",
                "Informática",
                "Mercadotecnia",
                "Pedagogía",
            ],
            "Responsable de titulación",
        ),
        "Brenda Ponce Pérez": (
            "Ejecutiva",
            [
                "Administración de Empresas",
                "Contaduría",
                "Derecho",
                "Informática",
                "Mercadotecnia",
                "Pedagogía",
            ],
            "Responsable de evaluación",
        ),
    }

    claves_existentes = {
        (
            _clave_personal_ascii(fila.get("CARRERA_BASE", "")),
            _clave_personal_ascii(fila.get("MODALIDAD_ACADEMICA", "")),
            _clave_personal_ascii(fila.get("PERSONA_EVALUADA", "")),
        )
        for fila in catalogo_personal
    }

    for persona_confirmada, (
        modalidad_confirmada,
        carreras_confirmadas,
        rol_fallback,
    ) in asignaciones_confirmadas.items():
        nombre_key = _clave_personal_ascii(persona_confirmada)
        rol_confirmado = roles_por_persona.get(
            nombre_key,
            rol_fallback,
        )

        for carrera_confirmada in carreras_confirmadas:
            clave = (
                _clave_personal_ascii(carrera_confirmada),
                _clave_personal_ascii(modalidad_confirmada),
                nombre_key,
            )
            if clave in claves_existentes:
                continue

            catalogo_personal.append(
                {
                    "CARRERA_BASE": carrera_confirmada,
                    "MODALIDAD_ACADEMICA": modalidad_confirmada,
                    "PERSONA_EVALUADA": persona_confirmada,
                    "ROL_EVALUADO": rol_confirmado,
                }
            )
            claves_existentes.add(clave)

    claves_catalogo_personal = {
        _clave_asignacion_personal(
            fila.get(
                "CARRERA_BASE",
                "",
            ),
            fila.get(
                "MODALIDAD_ACADEMICA",
                "",
            ),
            fila.get(
                "PERSONA_EVALUADA",
                "",
            ),
            fila.get(
                "ROL_EVALUADO",
                "",
            ),
        )
        for fila in catalogo_personal
        if fila.get(
            "PERSONA_EVALUADA"
        )
    }

    candidatos_personal = [
        registro
        for registro in validos_kpi
        if _es_persona_evaluada_valida(
            registro.get(
                "PERSONA_EVALUADA",
                "",
            )
        )
        and _asignacion_personal_confirmada(
            registro.get("CARRERA_BASE", "")
            or registro.get("SERVICIO_PROGRAMA", ""),
            registro.get("MODALIDAD_ACADEMICA", "")
            or registro.get("MODALIDAD_DETALLE", ""),
            registro.get("PERSONA_EVALUADA", ""),
        )
    ]

    nombres_confirmados = {
        _clave_personal_ascii(nombre)
        for nombre in asignaciones_confirmadas
    }
    nombres_confirmados.add("claudia diaz")

    if claves_catalogo_personal:
        registros_personal = [
            registro
            for registro in candidatos_personal
            if (
                _clave_personal_ascii(
                    registro.get("PERSONA_EVALUADA", "")
                )
                in nombres_confirmados
                or _clave_asignacion_personal(
                    registro.get(
                        "CARRERA_BASE",
                        "",
                    )
                    or registro.get(
                        "SERVICIO_PROGRAMA",
                        "",
                    ),
                    registro.get(
                        "MODALIDAD_ACADEMICA",
                        "",
                    )
                    or registro.get(
                        "MODALIDAD_DETALLE",
                        "",
                    ),
                    registro.get(
                        "PERSONA_EVALUADA",
                        "",
                    ),
                    registro.get(
                        "ROL_EVALUADO",
                        "",
                    ),
                )
                in claves_catalogo_personal
            )
        ]
    else:
        # Compatibilidad con instrumentos que aun
        # no tengan catalogo de personal en MASTER.
        registros_personal = (
            candidatos_personal
        )

    def _indice_por_grupo(
        registros_grupo,
    ):
        por_pregunta_grupo = defaultdict(
            list
        )

        for registro in registros_grupo:
            indice = _numero(
                registro.get(
                    "INDICE_100"
                )
            )

            if indice is None:
                continue

            pregunta_id = str(
                registro.get(
                    "PREGUNTA_ID",
                    "",
                )
                or ""
            ).strip()

            if not pregunta_id:
                continue

            por_pregunta_grupo[
                pregunta_id
            ].append(
                indice
            )

        indices_pregunta = [
            _promedio(valores)
            for valores
            in por_pregunta_grupo.values()
        ]

        return _promedio(
            indices_pregunta
        )

    # -------------------------
    # Comparativo por carrera
    # -------------------------

    personal_por_carrera = defaultdict(
        list
    )

    for registro in registros_personal:
        servicio = str(
            registro.get(
                "SERVICIO_PROGRAMA",
                "",
            )
            or ""
        ).strip()

        seccion_id = str(
            registro.get(
                "SECCION_ID",
                "",
            )
            or ""
        ).strip()

        if servicio and seccion_id:
            personal_por_carrera[
                (
                    seccion_id,
                    servicio,
                )
            ].append(
                registro
            )

    comparativo_carreras_personal = []

    for (
        seccion_id,
        servicio,
    ), grupo in personal_por_carrera.items():

        personas_carrera = {
            str(
                registro.get(
                    "PERSONA_EVALUADA",
                    "",
                )
                or ""
            ).strip()
            for registro in grupo
            if str(
                registro.get(
                    "PERSONA_EVALUADA",
                    "",
                )
                or ""
            ).strip()
        }

        filas_carrera = {
            registro.get(
                "FILA_ORIGEN"
            )
            for registro in grupo
        }

        comparativo_carreras_personal.append(
            {
                "seccion_id": seccion_id,
                "servicio_programa": servicio,
                "indice": _indice_por_grupo(
                    grupo
                ),
                "personas_evaluadas": len(
                    personas_carrera
                ),
                "evaluaciones": len(
                    filas_carrera
                ),
            }
        )

    # ---------------------------------
    # Persona segmentada por carrera
    # ---------------------------------

    personal_por_persona_carrera = defaultdict(
        list
    )

    for registro in registros_personal:
        servicio = str(
            registro.get(
                "SERVICIO_PROGRAMA",
                "",
            )
            or ""
        ).strip()

        persona = str(
            registro.get(
                "PERSONA_EVALUADA",
                "",
            )
            or ""
        ).strip()

        rol = str(
            registro.get(
                "ROL_EVALUADO",
                "",
            )
            or ""
        ).strip()

        seccion_id = str(
            registro.get(
                "SECCION_ID",
                "",
            )
            or ""
        ).strip()

        if (
            servicio
            and persona
            and seccion_id
        ):
            personal_por_persona_carrera[
                (
                    seccion_id,
                    servicio,
                    persona,
                    rol,
                )
            ].append(
                registro
            )

    personas_por_carrera = []

    for (
        seccion_id,
        servicio,
        persona,
        rol,
    ), grupo in personal_por_persona_carrera.items():

        filas_persona = {
            registro.get(
                "FILA_ORIGEN"
            )
            for registro in grupo
        }

        preguntas_persona = {
            str(
                registro.get(
                    "PREGUNTA_ID",
                    "",
                )
                or ""
            ).strip()
            for registro in grupo
            if str(
                registro.get(
                    "PREGUNTA_ID",
                    "",
                )
                or ""
            ).strip()
        }

        personas_por_carrera.append(
            {
                "seccion_id": seccion_id,
                "servicio_programa": servicio,
                "persona_evaluada": persona,
                "rol_evaluado": rol,
                "indice": _indice_por_grupo(
                    grupo
                ),
                "preguntas_con_datos": len(
                    preguntas_persona
                ),
                "evaluaciones": len(
                    filas_persona
                ),
            }
        )

    # ---------------------------------
    # Reactivos por persona y carrera
    # ---------------------------------

    registros_por_pregunta_persona_carrera = defaultdict(
        list
    )

    for registro in registros_personal:
        servicio = str(
            registro.get(
                "SERVICIO_PROGRAMA",
                "",
            )
            or ""
        ).strip()

        persona = str(
            registro.get(
                "PERSONA_EVALUADA",
                "",
            )
            or ""
        ).strip()

        rol = str(
            registro.get(
                "ROL_EVALUADO",
                "",
            )
            or ""
        ).strip()

        seccion_id = str(
            registro.get(
                "SECCION_ID",
                "",
            )
            or ""
        ).strip()

        pregunta_id = str(
            registro.get(
                "PREGUNTA_ID",
                "",
            )
            or ""
        ).strip()

        if (
            servicio
            and persona
            and seccion_id
            and pregunta_id
        ):
            registros_por_pregunta_persona_carrera[
                (
                    seccion_id,
                    servicio,
                    persona,
                    rol,
                    pregunta_id,
                )
            ].append(
                registro
            )

    preguntas_persona_por_carrera = []

    for (
        seccion_id,
        servicio,
        persona,
        rol,
        pregunta_id,
    ), grupo in registros_por_pregunta_persona_carrera.items():

        indices = [
            _numero(
                registro.get(
                    "INDICE_100"
                )
            )
            for registro in grupo
        ]

        indices = [
            indice
            for indice in indices
            if indice is not None
        ]

        n = len(indices)

        favorables = sum(
            1
            for indice in indices
            if indice >= 75
        )

        neutrales = sum(
            1
            for indice in indices
            if 50 <= indice < 75
        )

        desfavorables = sum(
            1
            for indice in indices
            if indice < 50
        )

        ejemplo = grupo[0]

        preguntas_persona_por_carrera.append(
            {
                "seccion_id": seccion_id,
                "servicio_programa": servicio,
                "persona_evaluada": persona,
                "rol_evaluado": rol,
                "pregunta_id": pregunta_id,
                "texto_corto": str(
                    ejemplo.get(
                        "TEXTO_CORTO",
                        "",
                    )
                    or ""
                ).strip(),
                "tipo_dato": str(
                    ejemplo.get(
                        "TIPO_DATO",
                        "",
                    )
                    or ""
                ).strip(),
                "indice": _promedio(
                    indices
                ),
                "n": n,
                "favorable_pct": (
                    round(
                        favorables / n * 100,
                        2,
                    )
                    if n
                    else None
                ),
                "neutral_pct": (
                    round(
                        neutrales / n * 100,
                        2,
                    )
                    if n
                    else None
                ),
                "desfavorable_pct": (
                    round(
                        desfavorables / n * 100,
                        2,
                    )
                    if n
                    else None
                ),
            }
        )

    # ---------------------------------
    # Comparativos canonicos
    # Carrera base -> modalidad academica
    # ---------------------------------

    personal_por_carrera_base = defaultdict(
        list
    )

    personal_por_carrera_modalidad = defaultdict(
        list
    )

    personal_persona_carrera_modalidad = defaultdict(
        list
    )

    preguntas_persona_carrera_modalidad = defaultdict(
        list
    )

    for registro in registros_personal:
        seccion_id = str(
            registro.get(
                "SECCION_ID",
                "",
            )
            or ""
        ).strip()

        carrera_base = str(
            registro.get(
                "CARRERA_BASE",
                "",
            )
            or registro.get(
                "SERVICIO_PROGRAMA",
                "",
            )
            or ""
        ).strip()

        modalidad_academica = str(
            registro.get(
                "MODALIDAD_ACADEMICA",
                "",
            )
            or registro.get(
                "MODALIDAD_DETALLE",
                "",
            )
            or ""
        ).strip()

        persona = str(
            registro.get(
                "PERSONA_EVALUADA",
                "",
            )
            or ""
        ).strip()

        rol = str(
            registro.get(
                "ROL_EVALUADO",
                "",
            )
            or ""
        ).strip()

        pregunta_id = str(
            registro.get(
                "PREGUNTA_ID",
                "",
            )
            or ""
        ).strip()

        if not (
            seccion_id
            and carrera_base
        ):
            continue

        personal_por_carrera_base[
            (
                seccion_id,
                carrera_base,
            )
        ].append(
            registro
        )

        personal_por_carrera_modalidad[
            (
                seccion_id,
                carrera_base,
                modalidad_academica,
            )
        ].append(
            registro
        )

        if persona:
            personal_persona_carrera_modalidad[
                (
                    seccion_id,
                    carrera_base,
                    modalidad_academica,
                    persona,
                    rol,
                )
            ].append(
                registro
            )

        if persona and pregunta_id:
            preguntas_persona_carrera_modalidad[
                (
                    seccion_id,
                    carrera_base,
                    modalidad_academica,
                    persona,
                    rol,
                    pregunta_id,
                )
            ].append(
                registro
            )

    comparativo_carrera_base = []

    for (
        seccion_id,
        carrera_base,
    ), grupo in personal_por_carrera_base.items():

        personas_base = {
            str(
                registro.get(
                    "PERSONA_EVALUADA",
                    "",
                )
                or ""
            ).strip()
            for registro in grupo
            if str(
                registro.get(
                    "PERSONA_EVALUADA",
                    "",
                )
                or ""
            ).strip()
        }

        filas_base = {
            registro.get(
                "FILA_ORIGEN"
            )
            for registro in grupo
        }

        modalidades_base = sorted(
            {
                str(
                    registro.get(
                        "MODALIDAD_ACADEMICA",
                        "",
                    )
                    or registro.get(
                        "MODALIDAD_DETALLE",
                        "",
                    )
                    or ""
                ).strip()
                for registro in grupo
                if str(
                    registro.get(
                        "MODALIDAD_ACADEMICA",
                        "",
                    )
                    or registro.get(
                        "MODALIDAD_DETALLE",
                        "",
                    )
                    or ""
                ).strip()
            }
        )

        comparativo_carrera_base.append(
            {
                "seccion_id": seccion_id,
                "carrera_base": carrera_base,
                "indice": _indice_por_grupo(
                    grupo
                ),
                "personas_evaluadas": len(
                    personas_base
                ),
                "evaluaciones": len(
                    filas_base
                ),
                "modalidades": modalidades_base,
                "requiere_modalidad": (
                    len(modalidades_base) > 1
                ),
                "indice_mostrable": (
                    _indice_por_grupo(grupo)
                    if len(modalidades_base) <= 1
                    else None
                ),
            }
        )

    comparativo_carrera_modalidad = []

    for (
        seccion_id,
        carrera_base,
        modalidad_academica,
    ), grupo in personal_por_carrera_modalidad.items():

        personas_modalidad = {
            str(
                registro.get(
                    "PERSONA_EVALUADA",
                    "",
                )
                or ""
            ).strip()
            for registro in grupo
            if str(
                registro.get(
                    "PERSONA_EVALUADA",
                    "",
                )
                or ""
            ).strip()
        }

        filas_modalidad = {
            registro.get(
                "FILA_ORIGEN"
            )
            for registro in grupo
            if registro.get(
                "FILA_ORIGEN"
            ) not in ("", None)
        }

        criterios_modalidad = {
            _normalizar_texto_cualitativo(
                registro.get(
                    "TEXTO_CORTO",
                    "",
                )
            )
            for registro in grupo
            if _normalizar_texto_cualitativo(
                registro.get(
                    "TEXTO_CORTO",
                    "",
                )
            )
        }

        comparativo_carrera_modalidad.append(
            {
                "seccion_id": seccion_id,
                "carrera_base": carrera_base,
                "modalidad_academica": modalidad_academica,
                "indice": _indice_por_grupo(
                    grupo
                ),
                "personas_evaluadas": len(
                    personas_modalidad
                ),
                "evaluaciones": len(
                    filas_modalidad
                ),
                "criterios_evaluados": len(
                    criterios_modalidad
                ),
            }
        )

    # -------------------------------------------------
    # Completar carrera + modalidad desde el MASTER
    # aunque todavia no existan respuestas.
    # -------------------------------------------------

    catalogo_por_poblacion = defaultdict(
        set
    )

    # IMPORTANTE: la lista de poblaciones se arma con el catalogo
    # original (antes de corregir persona -> carrera). Asi una carrera
    # no desaparece solo porque su asignacion nominal fue corregida.
    for fila in catalogo_poblaciones_personal:
        carrera_catalogo = str(
            fila.get(
                "CARRERA_BASE",
                "",
            )
            or ""
        ).strip()

        modalidad_catalogo = str(
            fila.get(
                "MODALIDAD_ACADEMICA",
                "",
            )
            or ""
        ).strip()

        persona_catalogo = str(
            fila.get(
                "PERSONA_EVALUADA",
                "",
            )
            or ""
        ).strip()

        rol_catalogo = str(
            fila.get(
                "ROL_EVALUADO",
                "",
            )
            or ""
        ).strip()

        if not (
            carrera_catalogo
            and modalidad_catalogo
        ):
            continue

        catalogo_por_poblacion[
            (
                carrera_catalogo,
                modalidad_catalogo,
            )
        ].add(
            (
                persona_catalogo,
                rol_catalogo,
            )
        )

    comparativo_por_clave = {
        (
            _clave_texto_personal(
                item.get(
                    "carrera_base",
                    "",
                )
            ),
            _clave_texto_personal(
                item.get(
                    "modalidad_academica",
                    "",
                )
            ),
        ): item
        for item in comparativo_carrera_modalidad
    }

    for (
        carrera_catalogo,
        modalidad_catalogo,
    ), personas_catalogo in catalogo_por_poblacion.items():

        clave = (
            _clave_texto_personal(
                carrera_catalogo
            ),
            _clave_texto_personal(
                modalidad_catalogo
            ),
        )

        personas_asignadas = len(
            {
                persona
                for persona, _ in personas_catalogo
                if persona
            }
        )

        existente = comparativo_por_clave.get(
            clave
        )

        if existente is not None:
            existente[
                "personas_asignadas"
            ] = personas_asignadas

            existente[
                "tiene_datos"
            ] = (
                existente.get(
                    "evaluaciones",
                    0,
                ) > 0
            )

            continue

        nuevo_item = {
            "seccion_id": "ESC_01_PERSONAL",
            "carrera_base": carrera_catalogo,
            "modalidad_academica": modalidad_catalogo,
            "indice": None,
            "personas_evaluadas": 0,
            "personas_asignadas": personas_asignadas,
            "evaluaciones": 0,
            "criterios_evaluados": 0,
            "tiene_datos": False,
        }

        comparativo_carrera_modalidad.append(
            nuevo_item
        )

        comparativo_por_clave[
            clave
        ] = nuevo_item

    # Los registros que si tienen respuestas deben
    # exponer tambien el indicador tiene_datos.
    for item in comparativo_carrera_modalidad:
        item.setdefault(
            "tiene_datos",
            item.get(
                "evaluaciones",
                0,
            ) > 0,
        )

        item.setdefault(
            "personas_asignadas",
            item.get(
                "personas_evaluadas",
                0,
            ),
        )

        item.setdefault(
            "criterios_evaluados",
            0,
        )

    personas_carrera_modalidad = []

    for (
        seccion_id,
        carrera_base,
        modalidad_academica,
        persona,
        rol,
    ), grupo in personal_persona_carrera_modalidad.items():

        filas_persona = {
            registro.get(
                "FILA_ORIGEN"
            )
            for registro in grupo
        }

        preguntas_persona = {
            str(
                registro.get(
                    "PREGUNTA_ID",
                    "",
                )
                or ""
            ).strip()
            for registro in grupo
            if str(
                registro.get(
                    "PREGUNTA_ID",
                    "",
                )
                or ""
            ).strip()
        }

        personas_carrera_modalidad.append(
            {
                "seccion_id": seccion_id,
                "carrera_base": carrera_base,
                "modalidad_academica": modalidad_academica,
                "persona_evaluada": persona,
                "rol_evaluado": rol,
                "indice": _indice_por_grupo(
                    grupo
                ),
                "preguntas_con_datos": len(
                    preguntas_persona
                ),
                "evaluaciones": len(
                    filas_persona
                ),
            }
        )

    preguntas_carrera_modalidad = []

    for (
        seccion_id,
        carrera_base,
        modalidad_academica,
        persona,
        rol,
        pregunta_id,
    ), grupo in preguntas_persona_carrera_modalidad.items():

        indices = [
            _numero(
                registro.get(
                    "INDICE_100"
                )
            )
            for registro in grupo
        ]

        indices = [
            indice
            for indice in indices
            if indice is not None
        ]

        n = len(indices)

        favorables = sum(
            1
            for indice in indices
            if indice >= 75
        )

        neutrales = sum(
            1
            for indice in indices
            if 50 <= indice < 75
        )

        desfavorables = sum(
            1
            for indice in indices
            if indice < 50
        )

        ejemplo = grupo[0]

        preguntas_carrera_modalidad.append(
            {
                "seccion_id": seccion_id,
                "carrera_base": carrera_base,
                "modalidad_academica": modalidad_academica,
                "persona_evaluada": persona,
                "rol_evaluado": rol,
                "pregunta_id": pregunta_id,
                "texto_corto": str(
                    ejemplo.get(
                        "TEXTO_CORTO",
                        "",
                    )
                    or ""
                ).strip(),
                "tipo_dato": str(
                    ejemplo.get(
                        "TIPO_DATO",
                        "",
                    )
                    or ""
                ).strip(),
                "indice": _promedio(
                    indices
                ),
                "n": n,
                "favorable_pct": (
                    round(
                        favorables / n * 100,
                        2,
                    )
                    if n
                    else None
                ),
                "neutral_pct": (
                    round(
                        neutrales / n * 100,
                        2,
                    )
                    if n
                    else None
                ),
                "desfavorable_pct": (
                    round(
                        desfavorables / n * 100,
                        2,
                    )
                    if n
                    else None
                ),
            }
        )

    # -------------------------
    # Comparativo por rol
    # -------------------------

    personal_por_rol = defaultdict(
        list
    )

    for registro in registros_personal:
        rol = str(
            registro.get(
                "ROL_EVALUADO",
                "",
            )
            or ""
        ).strip()

        seccion_id = str(
            registro.get(
                "SECCION_ID",
                "",
            )
            or ""
        ).strip()

        if rol and seccion_id:
            personal_por_rol[
                (
                    seccion_id,
                    rol,
                )
            ].append(
                registro
            )

    comparativo_roles_personal = []

    for (
        seccion_id,
        rol,
    ), grupo in personal_por_rol.items():

        personas_rol = {
            str(
                registro.get(
                    "PERSONA_EVALUADA",
                    "",
                )
                or ""
            ).strip()
            for registro in grupo
            if str(
                registro.get(
                    "PERSONA_EVALUADA",
                    "",
                )
                or ""
            ).strip()
        }

        filas_rol = {
            registro.get(
                "FILA_ORIGEN"
            )
            for registro in grupo
        }

        comparativo_roles_personal.append(
            {
                "seccion_id": seccion_id,
                "rol_evaluado": rol,
                "indice": _indice_por_grupo(
                    grupo
                ),
                "personas_evaluadas": len(
                    personas_rol
                ),
                "evaluaciones": len(
                    filas_rol
                ),
            }
        )

    registros_personal_validos = [
        registro
        for registro in registros_personal
        if registro.get(
            "FILA_ORIGEN"
        ) not in ("", None)
    ]

    filas_personal_unicas = {
        registro.get(
            "FILA_ORIGEN"
        )
        for registro in registros_personal_validos
    }

    criterios_personal_unicos = {
        _normalizar_texto_cualitativo(
            registro.get(
                "TEXTO_CORTO",
                "",
            )
        )
        for registro in registros_personal_validos
        if _normalizar_texto_cualitativo(
            registro.get(
                "TEXTO_CORTO",
                "",
            )
        )
    }

    personas_personal_unicas = {
        _clave_texto_personal(
            registro.get(
                "PERSONA_EVALUADA",
                "",
            )
        )
        for registro in registros_personal_validos
        if _es_persona_evaluada_valida(
            registro.get(
                "PERSONA_EVALUADA",
                "",
            )
        )
    }

    resumen_personal = {
        "evaluaciones": len(
            filas_personal_unicas
        ),
        "criterios_evaluados": len(
            criterios_personal_unicos
        ),
        "personas_evaluadas": len(
            personas_personal_unicas
        ),
    }

    comparativos_personal = {
        # Assignment membership comes from the active catalog and confirmed
        # mappings. Answer groups only supply the optional observed metrics.
        "asignaciones": [
            next((dict(observada) for observada in personas_carrera_modalidad
                  if _clave_asignacion_personal(
                      observada["carrera_base"], observada["modalidad_academica"],
                      observada["persona_evaluada"], observada["rol_evaluado"])
                  == _clave_asignacion_personal(
                      fila.get("CARRERA_BASE"), fila.get("MODALIDAD_ACADEMICA"),
                      fila.get("PERSONA_EVALUADA"), fila.get("ROL_EVALUADO"))), {
                "seccion_id": "ESC_01_PERSONAL",
                "carrera_base": fila.get("CARRERA_BASE", ""),
                "modalidad_academica": fila.get("MODALIDAD_ACADEMICA", ""),
                "persona_evaluada": fila.get("PERSONA_EVALUADA", ""),
                "rol_evaluado": fila.get("ROL_EVALUADO", ""),
                "indice": None, "preguntas_con_datos": 0, "evaluaciones": 0,
            })
            for fila in catalogo_personal
            if fila.get("CARRERA_BASE") and fila.get("PERSONA_EVALUADA")
            and (len(servicios_actuales) != 1 or (
                _clave_texto_personal(fila.get("CARRERA_BASE")),
                _clave_texto_personal(fila.get("MODALIDAD_ACADEMICA")),
            ) in poblaciones_actuales)
        ],
        "resumen": resumen_personal,
        "por_carrera_base": sorted(
            comparativo_carrera_base,
            key=lambda item: (
                item["seccion_id"],
                -(
                    item["indice"]
                    if item["indice"] is not None
                    else -1
                ),
                item["carrera_base"],
            ),
        ),
        "por_carrera_modalidad": sorted(
            comparativo_carrera_modalidad,
            key=lambda item: (
                item["seccion_id"],
                item["carrera_base"],
                item["modalidad_academica"],
            ),
        ),
        "personas_por_carrera_modalidad": sorted(
            personas_carrera_modalidad,
            key=lambda item: (
                item["seccion_id"],
                item["carrera_base"],
                item["modalidad_academica"],
                -(
                    item["indice"]
                    if item["indice"] is not None
                    else -1
                ),
                item["persona_evaluada"],
            ),
        ),
        "preguntas_persona_por_carrera_modalidad": sorted(
            preguntas_carrera_modalidad,
            key=lambda item: (
                item["seccion_id"],
                item["carrera_base"],
                item["modalidad_academica"],
                item["persona_evaluada"],
                item["pregunta_id"],
            ),
        ),
        "por_carrera": sorted(
            comparativo_carreras_personal,
            key=lambda item: (
                item["seccion_id"],
                -(
                    item["indice"]
                    if item["indice"]
                    is not None
                    else -1
                ),
                item["servicio_programa"],
            ),
        ),
        "preguntas_persona_por_carrera": sorted(
            preguntas_persona_por_carrera,
            key=lambda item: (
                item["seccion_id"],
                item["servicio_programa"],
                item["persona_evaluada"],
                item["pregunta_id"],
            ),
        ),
        "personas_por_carrera": sorted(
            personas_por_carrera,
            key=lambda item: (
                item["seccion_id"],
                item["servicio_programa"],
                -(
                    item["indice"]
                    if item["indice"]
                    is not None
                    else -1
                ),
                item["persona_evaluada"],
            ),
        ),
        "por_rol": sorted(
            comparativo_roles_personal,
            key=lambda item: (
                item["seccion_id"],
                -(
                    item["indice"]
                    if item["indice"]
                    is not None
                    else -1
                ),
                item["rol_evaluado"],
            ),
        ),
    }

    todas_preguntas_seccion = defaultdict(
        list
    )

    preguntas_seccion = defaultdict(
        list
    )

    for pregunta in preguntas:
        todas_preguntas_seccion[
            pregunta["seccion_id"]
        ].append(
            pregunta
        )

        if (
            pregunta["entra_seccion"]
            and pregunta["indice"]
            is not None
        ):
            preguntas_seccion[
                pregunta["seccion_id"]
            ].append(
                pregunta
            )

    secciones = []

    for seccion_id, todas in todas_preguntas_seccion.items():
        grupo = preguntas_seccion.get(
            seccion_id,
            [],
        )

        secciones.append(
            {
                "seccion_id": seccion_id,
                "seccion_nombre": todas[
                    0
                ][
                    "seccion_nombre"
                ],
                "indice": _promedio(
                    [
                        pregunta[
                            "indice"
                        ]
                        for pregunta in grupo
                    ]
                ),
                "preguntas_con_datos": len(
                    grupo
                ),
            }
        )

    preguntas_general = defaultdict(
        list
    )

    for pregunta in preguntas:
        if (
            pregunta["entra_general"]
            and pregunta["indice"]
            is not None
        ):
            preguntas_general[
                pregunta["seccion_id"]
            ].append(
                pregunta["indice"]
            )

    secciones_general = [
        _promedio(indices)
        for indices in preguntas_general.values()
        if indices
    ]

    indice_general = _promedio(
        secciones_general
    )

    comentarios = []

    for registro in registros:
        tipo_dato = str(
            registro.get(
                "TIPO_DATO",
                "",
            )
            or ""
        ).strip().upper()

        if tipo_dato != "TEXTO":
            continue

        texto = str(
            registro.get(
                "RESPUESTA_ORIGINAL"
            )
            or registro.get(
                "RESPUESTA_NORMALIZADA"
            )
            or ""
        ).strip()

        if not texto:
            continue

        clasificacion_cualitativa = (
            _clasificar_campo_cualitativo(
                str(
                    registro.get(
                        "TEXTO_CORTO",
                        "",
                    )
                    or ""
                )
            )
        )

        texto_informativo = (
            _es_texto_cualitativo_informativo(
                texto
            )
        )

        comentarios.append(
            {
                "texto_informativo": texto_informativo,
                "categoria_cualitativa": (
                    clasificacion_cualitativa[
                        "categoria"
                    ]
                ),
                "incluir_resumen": (
                    clasificacion_cualitativa[
                        "incluir_resumen"
                    ]
                ),
                "polaridad_base": (
                    clasificacion_cualitativa[
                        "polaridad_base"
                    ]
                ),
                "fila_origen": registro.get(
                    "FILA_ORIGEN"
                ),
                "servicio_programa": str(
                    registro.get(
                        "SERVICIO_PROGRAMA",
                        "",
                    )
                    or ""
                ).strip(),
                "seccion_id": str(
                    registro.get(
                        "SECCION_ID",
                        "",
                    )
                    or ""
                ).strip(),
                "seccion_nombre": str(
                    registro.get(
                        "SECCION_NOMBRE",
                        "",
                    )
                    or ""
                ).strip(),
                "pregunta_id": str(
                    registro.get(
                        "PREGUNTA_ID",
                        "",
                    )
                    or ""
                ).strip(),
                "texto_corto": str(
                    registro.get(
                        "TEXTO_CORTO",
                        "",
                    )
                    or ""
                ).strip(),
                "persona_evaluada": str(
                    registro.get(
                        "PERSONA_EVALUADA",
                        "",
                    )
                    or ""
                ).strip(),
                "rol_evaluado": str(
                    registro.get(
                        "ROL_EVALUADO",
                        "",
                    )
                    or ""
                ).strip(),
                "texto": texto,
            }
        )

    resumen_cualitativo = {}

    for comentario in comentarios:
        if not comentario.get(
            "incluir_resumen"
        ):
            continue

        if not comentario.get(
            "texto_informativo"
        ):
            continue

        seccion_id = comentario.get(
            "seccion_id",
            "",
        )

        if not seccion_id:
            continue

        if seccion_id not in resumen_cualitativo:
            resumen_cualitativo[
                seccion_id
            ] = {
                "seccion_id": seccion_id,
                "seccion_nombre": comentario.get(
                    "seccion_nombre",
                    "",
                ),
                "total": 0,
                "positivos": 0,
                "oportunidades": 0,
                "incidencias": 0,
                "sin_clasificar": 0,
            }

        resumen = resumen_cualitativo[
            seccion_id
        ]

        resumen["total"] += 1

        polaridad = comentario.get(
            "polaridad_base"
        )

        if polaridad == "positivo":
            resumen["positivos"] += 1

        elif polaridad == "oportunidad":
            resumen["oportunidades"] += 1

        elif polaridad == "negativo":
            resumen["incidencias"] += 1

        else:
            resumen["sin_clasificar"] += 1

    resumen_cualitativo = list(
        resumen_cualitativo.values()
    )

    # =========================================================
    # Comparativo de Direccion de carrera
    #
    # Se muestran TODAS las poblaciones observadas en el
    # periodo y filtros actuales.
    #
    # Estados:
    # - con_datos: la ruta pasa por Direccion y existen KPIs;
    # - sin_datos: la ruta llego a Direccion, pero no existen
    #   respuestas KPI validas;
    # - no_aplica: la ruta del formulario no pasa por Direccion.
    # =========================================================

    secciones_direccion_validas = {
        "ESC_02_DIRECCION",
        "ESC25_02_DIRECCION",
    }

    preguntas_direccion_validas = {
        "ESC_DIR_01",
        "ESC_DIR_02",
        "ESC_DIR_03",
        "ESC_DIR_04",
        "ESC25_DIR_01",
        "ESC25_DIR_02",
        "ESC25_DIR_03",
        "ESC25_DIR_04",
    }

    def _es_seccion_direccion(valor) -> bool:
        return str(valor or "").strip() in secciones_direccion_validas

    def _clave_carrera_direccion(valor) -> str:
        texto = _normalizar_texto_cualitativo(valor)
        reemplazos = {
            "cine y television digital": "cine y tv digital",
            "cine television digital": "cine y tv digital",
            "tecnologias de informacion": "tecnologias de la informacion",
            "tecnologia de la informacion": "tecnologias de la informacion",
        }
        return reemplazos.get(texto, texto)

    # Catálogo institucional de responsables de Dirección.
    # Sirve también para mostrar carreras sin encuestas aplicadas.
    catalogo_direccion_institucional = [
        ("Actuación", "Presencial", "Luly Rede Martínez"),
        ("Administración de Empresas", "Presencial", "Andrés Vázquez López"),
        ("Contaduría", "Presencial", "Andrés Vázquez López"),
        ("Finanzas", "Presencial", "Andrés Vázquez López"),
        ("Mercadotecnia", "Presencial", "Andrés Vázquez López"),
        ("Diseño Gráfico", "Presencial", "Claudia Vázquez Barajas"),
        ("Comunicación Multimedia", "Presencial", "Claudia Vázquez Barajas"),
        ("Cine y TV Digital", "Presencial", "Claudia Vázquez Barajas"),
        ("Derecho", "Presencial", "Elizabeth Pérez Camacho"),
        ("Diseño de Modas", "Presencial", "Patricia Verdeja"),
        ("Gastronomía", "Presencial", "Josué Espinosa Castelán"),
        ("Creación y Gestión de Empresas Turísticas", "Presencial", "Miroslava Aguilar Cortés"),
        ("Tecnologías de la Información", "Presencial", "Daniel Becerril"),
        ("Informática", "Presencial", "Daniel Becerril"),
        ("Nutrición", "Presencial", "María del Carmen Villagómez Amezcua"),
        ("Pedagogía", "Presencial", "Vera Granados García"),
        ("Psicología", "Presencial", "Tania Gómez Rivas"),
        ("Administración", "Ejecutiva", "Jesús Heredia Aparicio"),
        ("Contaduría", "Ejecutiva", "Jesús Heredia Aparicio"),
        ("Derecho", "Ejecutiva", "Jesús Heredia Aparicio"),
        ("Informática", "Ejecutiva", "Jesús Heredia Aparicio"),
        ("Mercadotecnia", "Ejecutiva", "Jesús Heredia Aparicio"),
        ("Pedagogía", "Ejecutiva", "Jesús Heredia Aparicio"),
        ("Posgrados y maestrías", "Posgrado", "Marisol Rodríguez González"),
        ("Preparatoria y certificaciones", "Prepa", "Elvira Martínez Rosas"),
        ("Entrenamiento Deportivo", "Presencial", "Luis Díaz"),
        ("Maestría en Metodología del Entrenamiento", "Posgrado", "Luis Díaz"),
    ]

    modalidades_agrupadas_actuales = {
        str(registro.get("MODALIDAD_AGRUPADA", "") or "").strip().upper()
        for registro in registros
        if str(registro.get("MODALIDAD_AGRUPADA", "") or "").strip()
    }

    def _modalidad_catalogo_aplica(modalidad: str) -> bool:
        if not modalidades_agrupadas_actuales:
            return True
        if modalidad == "Presencial":
            return any("ESCOLAR" in valor for valor in modalidades_agrupadas_actuales)
        if modalidad == "Ejecutiva":
            return any("EJECUT" in valor for valor in modalidades_agrupadas_actuales)
        if modalidad == "Posgrado":
            return any("POSGR" in valor or "MAEST" in valor for valor in modalidades_agrupadas_actuales)
        if modalidad == "Prepa":
            return any("PREPA" in valor or "BACH" in valor for valor in modalidades_agrupadas_actuales)
        return True

    servicios_direccion_actuales = {
        _clave_carrera_direccion(
            registro.get("CARRERA_BASE", "")
            or registro.get("SERVICIO_PROGRAMA", "")
        )
        for registro in registros
        if _clave_carrera_direccion(
            registro.get("CARRERA_BASE", "")
            or registro.get("SERVICIO_PROGRAMA", "")
        )
    }

    catalogo_direccion_aplicable = [
        (carrera, modalidad, persona)
        for carrera, modalidad, persona in catalogo_direccion_institucional
        if _modalidad_catalogo_aplica(modalidad)
    ]
    # Active current catalog populations remain visible even with no answers.
    # Historical responses never participate in constructing this universe.
    catalogo_direccion_aplicable.extend(
        (fila["CARRERA_BASE"], fila.get("MODALIDAD_ACADEMICA", ""), "")
        for fila in catalogo_poblaciones_personal
        if str(fila.get("CARRERA_BASE") or "").strip()
    )

    # Cuando el endpoint está filtrado a un solo servicio, el catálogo
    # también se limita a esa población.
    if len(servicios_direccion_actuales) == 1:
        unica = next(iter(servicios_direccion_actuales))
        filtrado = [
            fila
            for fila in catalogo_direccion_aplicable
            if _clave_carrera_direccion(fila[0]) == unica
        ]
        if filtrado:
            catalogo_direccion_aplicable = filtrado

    def _poblacion_direccion(
        registro,
    ):
        carrera = str(
            registro.get(
                "CARRERA_BASE",
                "",
            )
            or registro.get(
                "SERVICIO_PROGRAMA",
                "",
            )
            or ""
        ).strip()

        modalidad = str(
            registro.get(
                "MODALIDAD_ACADEMICA",
                "",
            )
            or registro.get(
                "MODALIDAD_DETALLE",
                "",
            )
            or ""
        ).strip()

        return (
            carrera,
            modalidad,
        )

    # ---------------------------------------------------------
    # Todas las carreras/modalidades realmente presentes en
    # las encuestas seleccionadas.
    # ---------------------------------------------------------

    poblaciones_totales = defaultdict(
        set
    )

    for registro in registros:
        carrera, modalidad = (
            _poblacion_direccion(
                registro
            )
        )

        if not carrera:
            continue

        fila = registro.get(
            "FILA_ORIGEN"
        )

        if fila not in (
            "",
            None,
        ):
            poblaciones_totales[
                (
                    carrera,
                    modalidad,
                )
            ].add(
                fila
            )

    claves_catalogo_direccion = set()
    for carrera_catalogo, modalidad_catalogo, _ in catalogo_direccion_aplicable:
        clave_catalogo = (carrera_catalogo, modalidad_catalogo)
        claves_catalogo_direccion.add(
            (
                _clave_carrera_direccion(carrera_catalogo),
                _clave_texto_personal(modalidad_catalogo),
            )
        )

        # defaultdict(set): acceder a la clave basta para conservarla
        # aunque no exista ninguna fila de encuesta.
        poblaciones_totales[clave_catalogo]

    # ---------------------------------------------------------
    # Detectar si la ruta realmente llego a Direccion.
    #
    # No basta con que exista una columna normalizada:
    # debe existir una respuesta no vacia en alguno de los
    # cuatro reactivos cuantitativos.
    # ---------------------------------------------------------

    poblaciones_ruta_direccion = defaultdict(
        set
    )

    for registro in registros:
        if str(
            registro.get(
                "SECCION_ID",
                "",
            )
            or ""
        ).strip() not in secciones_direccion_validas:
            continue

        if str(
            registro.get(
                "PREGUNTA_ID",
                "",
            )
            or ""
        ).strip() not in preguntas_direccion_validas:
            continue

        respuesta = registro.get(
            "RESPUESTA_ORIGINAL"
        )

        if respuesta in (
            "",
            None,
        ):
            respuesta = registro.get(
                "RESPUESTA_NORMALIZADA"
            )

        if str(
            respuesta
            if respuesta is not None
            else ""
        ).strip() == "":
            continue

        carrera, modalidad = (
            _poblacion_direccion(
                registro
            )
        )

        fila = registro.get(
            "FILA_ORIGEN"
        )

        if (
            carrera
            and fila not in (
                "",
                None,
            )
        ):
            poblaciones_ruta_direccion[
                (
                    carrera,
                    modalidad,
                )
            ].add(
                fila
            )

    # ---------------------------------------------------------
    # Registros KPI validos de Direccion.
    # ---------------------------------------------------------

    registros_direccion = [
        registro
        for registro in validos_kpi
        if str(
            registro.get(
                "SECCION_ID",
                "",
            )
            or ""
        ).strip() in secciones_direccion_validas
        and str(
            registro.get(
                "PREGUNTA_ID",
                "",
            )
            or ""
        ).strip()
        in preguntas_direccion_validas
        and _numero(
            registro.get(
                "INDICE_100"
            )
        )
        is not None
    ]

    direccion_por_poblacion = defaultdict(
        list
    )

    for registro in registros_direccion:
        carrera, modalidad = (
            _poblacion_direccion(
                registro
            )
        )

        if not carrera:
            continue

        direccion_por_poblacion[
            (
                carrera,
                modalidad,
            )
        ].append(
            registro
        )

    direccion_carreras = []
    direccion_preguntas = []

    for (
        carrera,
        modalidad,
    ), filas_totales in poblaciones_totales.items():

        grupo = direccion_por_poblacion.get(
            (
                carrera,
                modalidad,
            ),
            [],
        )

        filas_ruta = (
            poblaciones_ruta_direccion.get(
                (
                    carrera,
                    modalidad,
                ),
                set(),
            )
        )

        por_pregunta = defaultdict(
            list
        )

        for registro in grupo:
            pregunta_id = str(
                registro.get(
                    "PREGUNTA_ID",
                    "",
                )
                or ""
            ).strip()

            if pregunta_id:
                por_pregunta[
                    pregunta_id
                ].append(
                    registro
                )

        indices_pregunta = []

        for (
            pregunta_id,
            registros_pregunta,
        ) in por_pregunta.items():

            valores = [
                _numero(
                    registro.get(
                        "INDICE_100"
                    )
                )
                for registro
                in registros_pregunta
            ]

            valores = [
                valor
                for valor in valores
                if valor is not None
            ]

            indice_pregunta = _promedio(
                valores
            )

            if indice_pregunta is not None:
                indices_pregunta.append(
                    indice_pregunta
                )

            favorables = sum(
                1
                for valor in valores
                if valor >= 75
            )

            neutrales = sum(
                1
                for valor in valores
                if valor == 50
            )

            desfavorables = sum(
                1
                for valor in valores
                if valor < 50
            )

            total = len(
                valores
            )

            referencia = (
                registros_pregunta[0]
            )

            direccion_preguntas.append(
                {
                    "seccion_id":
                        "ESC_02_DIRECCION",
                    "carrera_base":
                        carrera,
                    "modalidad_academica":
                        modalidad,
                    "pregunta_id":
                        pregunta_id,
                    "texto_corto":
                        str(
                            referencia.get(
                                "TEXTO_CORTO",
                                "",
                            )
                            or ""
                        ).strip(),
                    "indice":
                        indice_pregunta,
                    "n":
                        total,
                    "favorable_pct":
                        round(
                            favorables
                            * 100
                            / total,
                            2,
                        )
                        if total
                        else None,
                    "neutral_pct":
                        round(
                            neutrales
                            * 100
                            / total,
                            2,
                        )
                        if total
                        else None,
                    "desfavorable_pct":
                        round(
                            desfavorables
                            * 100
                            / total,
                            2,
                        )
                        if total
                        else None,
                }
            )

        indice = _promedio(
            indices_pregunta
        )

        clave_estado_catalogo = (
            _clave_carrera_direccion(carrera),
            _clave_texto_personal(modalidad),
        )

        if indice is not None:
            estado = "con_datos"
        elif not filas_totales and clave_estado_catalogo in claves_catalogo_direccion:
            estado = "sin_encuestas"
        elif filas_ruta:
            estado = "sin_datos"
        else:
            estado = "no_aplica"

        filas_kpi = {
            registro.get(
                "FILA_ORIGEN"
            )
            for registro in grupo
            if registro.get(
                "FILA_ORIGEN"
            )
            not in (
                "",
                None,
            )
        }

        direccion_carreras.append(
            {
                "seccion_id":
                    "ESC_02_DIRECCION",
                "carrera_base":
                    carrera,
                "modalidad_academica":
                    modalidad,
                "indice":
                    indice,
                "encuestas":
                    len(
                        filas_kpi
                    ),
                "encuestas_total":
                    len(
                        filas_totales
                    ),
                "encuestas_ruta":
                    len(
                        filas_ruta
                    ),
                "preguntas_con_datos":
                    len(
                        indices_pregunta
                    ),
                "tiene_datos":
                    indice is not None,
                "aplica_direccion":
                    bool(
                        filas_ruta
                    ),
                "estado":
                    estado,
            }
        )

    orden_estado = {
        "con_datos": 0,
        "sin_datos": 1,
        "sin_encuestas": 2,
        "no_aplica": 3,
    }

    direccion_carreras.sort(
        key=lambda item: (
            orden_estado.get(
                item["estado"],
                9,
            ),
            -(
                item["indice"]
                if item["indice"]
                is not None
                else -1
            ),
            item["carrera_base"],
            item[
                "modalidad_academica"
            ],
        )
    )

    # ---------------------------------------------------------
    # Director(a) responsable por carrera/modalidad.
    # Se toma del catalogo oficial 04_PERSONAS_EVALUADAS para
    # no inferir nombres a partir de respuestas anonimas.
    # ---------------------------------------------------------

    directores_por_poblacion = defaultdict(dict)

    # El catalogo institucional confirmado es la fuente autoritativa.
    # La clave interna ignora el rol para evitar duplicar a la misma
    # persona como "Dirección de carrera" y "Director(a) de carrera".
    for carrera_catalogo, modalidad_catalogo, persona_catalogo in catalogo_direccion_aplicable:
        if not persona_catalogo:
            continue

        clave_poblacion = (
            _clave_carrera_direccion(carrera_catalogo),
            _clave_texto_personal(modalidad_catalogo),
        )
        clave_persona = _clave_texto_personal(persona_catalogo)
        directores_por_poblacion[clave_poblacion][clave_persona] = (
            persona_catalogo,
            "Dirección de carrera",
        )

    # El MASTER solo complementa poblaciones/personas que no estén ya
    # definidas en el catálogo institucional. Nunca crea un duplicado
    # por diferencias de escritura del rol.
    for fila in catalogo_personal:
        rol_catalogo = str(fila.get("ROL_EVALUADO", "") or "").strip()
        persona_catalogo = str(fila.get("PERSONA_EVALUADA", "") or "").strip()
        rol_clave = _clave_texto_personal(rol_catalogo)

        if not persona_catalogo or "director" not in rol_clave:
            continue

        carrera_catalogo = str(fila.get("CARRERA_BASE", "") or "").strip()
        modalidad_catalogo = str(fila.get("MODALIDAD_ACADEMICA", "") or "").strip()
        if not carrera_catalogo:
            continue

        clave_poblacion = (
            _clave_carrera_direccion(carrera_catalogo),
            _clave_texto_personal(modalidad_catalogo),
        )
        clave_persona = _clave_texto_personal(persona_catalogo)
        directores_por_poblacion[clave_poblacion].setdefault(
            clave_persona,
            (persona_catalogo, rol_catalogo or "Dirección de carrera"),
        )

    for item in direccion_carreras:
        directores = sorted(
            directores_por_poblacion.get(
                (
                    _clave_carrera_direccion(
                        item.get("carrera_base", "")
                    ),
                    _clave_texto_personal(
                        item.get("modalidad_academica", "")
                    ),
                ),
                {},
            ).values(),
            key=lambda x: (
                _clave_texto_personal(x[0]),
                _clave_texto_personal(x[1]),
            ),
        )
        item["directores"] = [
            {
                "persona_evaluada": persona,
                "rol_evaluado": rol,
            }
            for persona, rol in directores
        ]

    direccion_por_director = defaultdict(list)
    poblaciones_por_director = defaultdict(list)

    for item in direccion_carreras:
        carrera = str(item.get("carrera_base", "") or "").strip()
        modalidad = str(item.get("modalidad_academica", "") or "").strip()
        grupo = direccion_por_poblacion.get(
            (carrera, modalidad),
            [],
        )

        for director in item.get("directores", []):
            persona = str(
                director.get("persona_evaluada", "") or ""
            ).strip()
            rol = str(
                director.get("rol_evaluado", "") or ""
            ).strip()

            if not persona:
                continue

            clave_director = (persona, rol)
            # Crear la clave incluso si aún no tiene encuestas.
            direccion_por_director[clave_director]

            if item.get("estado") == "con_datos":
                direccion_por_director[clave_director].extend(
                    grupo
                )

            poblaciones_por_director[clave_director].append(
                {
                    "carrera_base": carrera,
                    "modalidad_academica": modalidad,
                    "indice": item.get("indice"),
                    "encuestas": item.get("encuestas", 0),
                    "preguntas_con_datos": item.get(
                        "preguntas_con_datos", 0
                    ),
                    "estado": item.get("estado", "sin_encuestas"),
                }
            )

    direccion_directores = []
    direccion_preguntas_director = []

    for (persona, rol), grupo in direccion_por_director.items():
        por_pregunta_director = defaultdict(list)

        for registro in grupo:
            pregunta_id = str(
                registro.get("PREGUNTA_ID", "") or ""
            ).strip()
            if pregunta_id:
                por_pregunta_director[pregunta_id].append(
                    registro
                )

        indices_director = []

        for pregunta_id, registros_pregunta in por_pregunta_director.items():
            valores = [
                _numero(registro.get("INDICE_100"))
                for registro in registros_pregunta
            ]
            valores = [
                valor for valor in valores
                if valor is not None
            ]

            indice_pregunta = _promedio(valores)
            if indice_pregunta is not None:
                indices_director.append(indice_pregunta)

            favorables = sum(1 for valor in valores if valor >= 75)
            neutrales = sum(1 for valor in valores if valor == 50)
            desfavorables = sum(1 for valor in valores if valor < 50)
            total = len(valores)
            referencia = registros_pregunta[0]

            direccion_preguntas_director.append(
                {
                    "seccion_id": "ESC_02_DIRECCION",
                    "persona_evaluada": persona,
                    "rol_evaluado": rol,
                    "pregunta_id": pregunta_id,
                    "texto_corto": str(
                        referencia.get("TEXTO_CORTO", "") or ""
                    ).strip(),
                    "indice": indice_pregunta,
                    "n": total,
                    "favorable_pct": (
                        round(favorables * 100 / total, 2)
                        if total else None
                    ),
                    "neutral_pct": (
                        round(neutrales * 100 / total, 2)
                        if total else None
                    ),
                    "desfavorable_pct": (
                        round(desfavorables * 100 / total, 2)
                        if total else None
                    ),
                }
            )

        filas_director = {
            registro.get("FILA_ORIGEN")
            for registro in grupo
            if registro.get("FILA_ORIGEN") not in ("", None)
        }

        poblaciones = sorted(
            poblaciones_por_director[(persona, rol)],
            key=lambda x: (
                x.get("carrera_base", ""),
                x.get("modalidad_academica", ""),
            ),
        )

        direccion_directores.append(
            {
                "persona_evaluada": persona,
                "rol_evaluado": rol,
                "indice": _promedio(indices_director),
                "evaluaciones": len(filas_director),
                "criterios_evaluados": len(indices_director),
                "carreras_a_cargo": len(
                    {
                        (
                            p.get("carrera_base", ""),
                            p.get("modalidad_academica", ""),
                        )
                        for p in poblaciones
                    }
                ),
                "poblaciones": poblaciones,
            }
        )

    direccion_directores.sort(
        key=lambda item: (
            -(
                item.get("indice")
                if item.get("indice") is not None
                else -1
            ),
            item.get("persona_evaluada", ""),
        )
    )

    comparativos_direccion = {
        "por_carrera_modalidad": direccion_carreras,
        "preguntas_por_carrera_modalidad": direccion_preguntas,
        "por_director": direccion_directores,
        "preguntas_por_director": direccion_preguntas_director,
    }


    # ---------------------------------
    # Comparativos generales por sección
    # Sección -> carrera -> modalidad
    # ---------------------------------

    secciones_por_carrera_modalidad = defaultdict(
        list
    )

    preguntas_por_carrera_modalidad = defaultdict(
        list
    )

    for registro in validos_kpi:
        seccion_id = str(
            registro.get(
                "SECCION_ID",
                "",
            )
            or ""
        ).strip()

        carrera_base = str(
            registro.get(
                "CARRERA_BASE",
                "",
            )
            or registro.get(
                "SERVICIO_PROGRAMA",
                "",
            )
            or ""
        ).strip()

        modalidad_academica = str(
            registro.get(
                "MODALIDAD_ACADEMICA",
                "",
            )
            or registro.get(
                "MODALIDAD_DETALLE",
                "",
            )
            or ""
        ).strip()

        pregunta_id = str(
            registro.get(
                "PREGUNTA_ID",
                "",
            )
            or ""
        ).strip()

        if not (
            seccion_id
            and carrera_base
            and pregunta_id
        ):
            continue

        clave_seccion = (
            seccion_id,
            carrera_base,
            modalidad_academica,
        )

        clave_pregunta = (
            seccion_id,
            carrera_base,
            modalidad_academica,
            pregunta_id,
        )

        secciones_por_carrera_modalidad[
            clave_seccion
        ].append(
            registro
        )

        preguntas_por_carrera_modalidad[
            clave_pregunta
        ].append(
            registro
        )

    comparativo_secciones_preguntas = []

    for (
        seccion_id,
        carrera_base,
        modalidad_academica,
        pregunta_id,
    ), grupo in preguntas_por_carrera_modalidad.items():

        ejemplo = grupo[0]

        indices = [
            _numero(
                registro.get(
                    "INDICE_100"
                )
            )
            for registro in grupo
        ]

        indices = [
            valor
            for valor in indices
            if valor is not None
        ]

        valores_base = [
            _numero(
                registro.get(
                    "VALOR_BASE"
                )
            )
            for registro in grupo
        ]

        valores_base = [
            valor
            for valor in valores_base
            if valor is not None
        ]

        n = len(indices)

        tipo_dato = str(
            ejemplo.get(
                "TIPO_DATO",
                "",
            )
            or ""
        ).strip().upper()

        if tipo_dato == "BINARIA":
            favorables = sum(
                1
                for valor in indices
                if valor == 100
            )

            neutrales = 0

            desfavorables = sum(
                1
                for valor in indices
                if valor == 0
            )

        else:
            favorables = sum(
                1
                for valor in valores_base
                if valor >= 4
            )

            neutrales = sum(
                1
                for valor in valores_base
                if valor == 3
            )

            desfavorables = sum(
                1
                for valor in valores_base
                if valor <= 2
            )

        comparativo_secciones_preguntas.append(
            {
                "seccion_id":
                    seccion_id,
                "seccion_nombre":
                    str(
                        ejemplo.get(
                            "SECCION_NOMBRE",
                            "",
                        )
                        or ""
                    ).strip(),
                "carrera_base":
                    carrera_base,
                "modalidad_academica":
                    modalidad_academica,
                "pregunta_id":
                    pregunta_id,
                "texto_corto":
                    str(
                        ejemplo.get(
                            "TEXTO_CORTO",
                            "",
                        )
                        or ""
                    ).strip(),
                "tipo_dato":
                    tipo_dato,
                "indice":
                    _promedio(
                        indices
                    ),
                "n":
                    n,
                "favorable_pct":
                    (
                        round(
                            favorables
                            * 100
                            / n,
                            2,
                        )
                        if n
                        else None
                    ),
                "neutral_pct":
                    (
                        round(
                            neutrales
                            * 100
                            / n,
                            2,
                        )
                        if n
                        else None
                    ),
                "desfavorable_pct":
                    (
                        round(
                            desfavorables
                            * 100
                            / n,
                            2,
                        )
                        if n
                        else None
                    ),
                "entra_seccion":
                    _es_si(
                        ejemplo.get(
                            "ENTRA_PROMEDIO_SECCION",
                            "",
                        )
                    ),
            }
        )

    comparativo_secciones_carreras = []

    for (
        seccion_id,
        carrera_base,
        modalidad_academica,
    ), grupo in secciones_por_carrera_modalidad.items():

        ejemplo = grupo[0]

        por_pregunta_grupo = defaultdict(
            list
        )

        entra_seccion_por_pregunta = {}

        for registro in grupo:
            pregunta_id = str(
                registro.get(
                    "PREGUNTA_ID",
                    "",
                )
                or ""
            ).strip()

            indice = _numero(
                registro.get(
                    "INDICE_100"
                )
            )

            if (
                not pregunta_id
                or indice is None
            ):
                continue

            por_pregunta_grupo[
                pregunta_id
            ].append(
                indice
            )

            entra_seccion_por_pregunta[
                pregunta_id
            ] = _es_si(
                registro.get(
                    "ENTRA_PROMEDIO_SECCION",
                    "",
                )
            )

        indices_preguntas = [
            _promedio(
                valores
            )
            for pregunta_id, valores
            in por_pregunta_grupo.items()
            if (
                valores
                and entra_seccion_por_pregunta.get(
                    pregunta_id,
                    False,
                )
            )
        ]

        indices_preguntas = [
            valor
            for valor in indices_preguntas
            if valor is not None
        ]

        filas = {
            registro.get(
                "FILA_ORIGEN"
            )
            for registro in grupo
            if registro.get(
                "FILA_ORIGEN"
            )
            not in (
                "",
                None,
            )
        }

        comparativo_secciones_carreras.append(
            {
                "seccion_id":
                    seccion_id,
                "seccion_nombre":
                    str(
                        ejemplo.get(
                            "SECCION_NOMBRE",
                            "",
                        )
                        or ""
                    ).strip(),
                "carrera_base":
                    carrera_base,
                "modalidad_academica":
                    modalidad_academica,
                "indice":
                    _promedio(
                        indices_preguntas
                    ),
                "encuestas":
                    len(
                        filas
                    ),
                "preguntas_con_datos":
                    len(
                        indices_preguntas
                    ),
                "tiene_datos":
                    bool(
                        indices_preguntas
                    ),
            }
        )

    comparativos_secciones = {
        "por_carrera_modalidad": sorted(
            comparativo_secciones_carreras,
            key=lambda item: (
                item["seccion_id"],
                -(
                    item["indice"]
                    if item["indice"]
                    is not None
                    else -1
                ),
                item["carrera_base"],
                item["modalidad_academica"],
            ),
        ),
        "preguntas_por_carrera_modalidad": sorted(
            comparativo_secciones_preguntas,
            key=lambda item: (
                item["seccion_id"],
                item["carrera_base"],
                item["modalidad_academica"],
                item["pregunta_id"],
            ),
        ),
    }

    # ---------------------------------------------------------
    # Exportacion de respuestas normalizadas.
    #
    # Se entrega en formato largo (una fila por respuesta a un
    # reactivo) y respeta los filtros aplicados antes de llegar
    # a este servicio. Se omiten celdas sin respuesta para que
    # la descarga sea utilizable directamente en Excel.
    # ---------------------------------------------------------

    respuestas_export = []

    for registro in registros:
        respuesta_original = registro.get(
            "RESPUESTA_ORIGINAL"
        )

        if respuesta_original in (
            "",
            None,
        ):
            respuesta_original = registro.get(
                "RESPUESTA_NORMALIZADA"
            )

        if str(
            respuesta_original
            if respuesta_original is not None
            else ""
        ).strip() == "":
            continue

        respuestas_export.append(
            {
                "Año": registro.get("ANIO", ""),
                "Marca temporal": registro.get("MARCA_TEMPORAL", ""),
                "Fila origen": registro.get("FILA_ORIGEN", ""),
                "Modalidad": (
                    registro.get("MODALIDAD_ACADEMICA", "")
                    or registro.get("MODALIDAD_DETALLE", "")
                    or registro.get("MODALIDAD_AGRUPADA", "")
                ),
                "Servicio / programa": registro.get("SERVICIO_PROGRAMA", ""),
                "Carrera base": (
                    registro.get("CARRERA_BASE", "")
                    or registro.get("SERVICIO_PROGRAMA", "")
                ),
                "Grado": registro.get("GRADO", ""),
                "Ciclo escolar": registro.get("CICLO_ESCOLAR", ""),
                "Turno": registro.get("TURNO", ""),
                "Edad": registro.get("EDAD", ""),
                "Sección": registro.get("SECCION_NOMBRE", ""),
                "Sección ID": registro.get("SECCION_ID", ""),
                "Dimensión": registro.get("DIMENSION", ""),
                "Subdimensión": registro.get("SUBDIMENSION", ""),
                "Pregunta": registro.get("TEXTO_CORTO", ""),
                "Pregunta ID": registro.get("PREGUNTA_ID", ""),
                "Pregunta canónica ID": registro.get("PREGUNTA_CANONICA_ID", ""),
                "Persona evaluada": registro.get("PERSONA_EVALUADA", ""),
                "Rol evaluado": registro.get("ROL_EVALUADO", ""),
                "Respuesta original": registro.get("RESPUESTA_ORIGINAL", ""),
                "Respuesta normalizada": registro.get("RESPUESTA_NORMALIZADA", ""),
                "Índice 100": registro.get("INDICE_100", ""),
                "Respuesta válida": registro.get("RESPUESTA_VALIDA", ""),
                "Incluye KPI": registro.get("INCLUIR_EN_KPI", ""),
                "Comparable histórico": registro.get("COMPARABLE_HISTORICO", ""),
                "Periodo ID": registro.get("PERIODO_ID", ""),
                "Versión instrumento": registro.get("VERSION_INSTRUMENTO", ""),
            }
        )

    return {
        "encuestas": len(
            filas_origen
        ),
        "perfil_respuestas": perfil_respuestas,
        "respuestas_normalizadas": len(
            registros
        ),
        "respuestas_kpi_validas": len(
            validos_kpi
        ),
        "preguntas_con_datos": len(
            preguntas
        ),
        "indice_general": indice_general,
        "secciones_general": len(
            secciones_general
        ),
        "secciones": sorted(
            secciones,
            key=lambda x: x[
                "seccion_id"
            ],
        ),
        "preguntas": preguntas,
        "comentarios": comentarios,
        "resumen_cualitativo": resumen_cualitativo,
        "comparativos_personal": comparativos_personal,
        "comparativos_direccion": comparativos_direccion,
        "comparativos_secciones": comparativos_secciones,
        "respuestas_export": respuestas_export,
        "personas": sorted(
            personas,
            key=lambda x: (
                x["seccion_id"],
                x["persona_evaluada"],
            ),
        ),
    }
