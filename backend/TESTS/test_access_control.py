from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pandas as pd
from fastapi.testclient import TestClient

from backend.CONFIG.auth import AuthSettings
from backend.CORE.auth.google_oidc import ISSUER, verified_identity
from backend.CORE.auth.policy import Identity, OBSERVATION_VIEW, QUALITY_COMMENTS, QUALITY_EXPORT, QUALITY_SIA
from backend.GATEWAY import auth_api, calidad_api, institutional_services_api, observation_api, results_report_api
from backend.GATEWAY.main import app
from backend.REPOSITORIES.auth.access import AccessRepository
from backend.REPOSITORIES.auth.sessions import SessionStore
from backend.SERVICES.calidad import kpi_service, results_report_source
from backend.TESTS import test_results_report_pdf as fixtures

BASE = "/api/encuesta-calidad"
OBS = "/api/observacion"
SUMMARY = f"{BASE}/preview/resumen"
PDF = f"{BASE}/informe-resultados.pdf"
ORIGIN = "https://udl.example.test"


def account(user, role, modules="ALL", service="", active="SI"):
    return {"EMAIL": f"{user}@example.test", "ROL": role, "MODULOS": modules,
            "SERVICIO_ASIGNADO": service, "ACTIVO": active}


class AccessControlTests(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(TemporaryDirectory())
        self.now = 100
        self.rows = [
            account("dg", "DG"), account("dc", "DC", "encuesta_calidad", "Derecho"),
            account("dc_observation", "DC", "observacion_clases", "Derecho"),
            account("dc_all", "DC", "ALL", "Derecho"),
            account("dc_both", "DC", "encuesta_calidad,observacion_clases", "Derecho"),
            account("dc_without_scope", "DC"),
            account("inactive", "DG", active="NO"), account("df", "DF"),
        ]
        self.config = AuthSettings(public_origin=ORIGIN, client_id="test-client", client_secret="test-only",
                                   access_sheet_id="fixture", session_seconds=480, access_ttl_seconds=30)
        self.repository = AccessRepository(lambda: self.rows, 30, clock=lambda: self.now)
        self.store = SessionStore(Path(self.directory) / "sessions.sqlite3", clock=lambda: self.now)
        self.enterContext(patch.object(auth_api, "auth_settings", self.config))
        self.enterContext(patch.object(auth_api, "session_store", self.store))
        self.enterContext(patch.object(auth_api, "access_repository", self.repository))
        self.client = self.enterContext(TestClient(app, base_url=ORIGIN, follow_redirects=False))
        self.addCleanup(results_report_source.clear_snapshots)
        results_report_source.clear_snapshots()

    def identity(self, username):
        return Identity(ISSUER, f"google-{username}", f"{username}@example.test", username)

    def session(self, username):
        token = self.store.create(self.identity(username), self.config.session_seconds)
        self.client.cookies.set(auth_api.cookie_name("session"), token)
        return token

    def google_callback(self, username):
        start = self.client.get("/api/auth/login?next_path=/encuesta-calidad")
        self.assertEqual(start.status_code, 303)
        params = parse_qs(urlsplit(start.headers["location"]).query)
        self.assertEqual(params["code_challenge_method"], ["S256"])
        self.assertTrue(params["nonce"][0])
        self.assertIn("HttpOnly", start.headers["set-cookie"])
        self.assertIn("Secure", start.headers["set-cookie"])
        with patch.object(auth_api, "exchange_code", return_value=self.identity(username)):
            return self.client.get("/api/auth/callback", params={"state": params["state"][0], "code": "fixture"})

    def mock_quality(self):
        datasets = {}
        for year, scores in [(2026, (25, 100, 75)), (2025, (50, 0, 100))]:
            mapping = fixtures.instrument(year)
            rows = []
            for i, (career, service, modality, score) in enumerate([
                ("Derecho", "Derecho", "Presencial", scores[0]),
                ("Nutrición", "Nutrición", "Presencial", scores[1]),
                ("Derecho", "Licenciatura Ejecutiva: Derecho", "Ejecutiva", scores[2]),
            ], 1):
                for row in fixtures.population(mapping, score, i, career, modality):
                    row.update(ANIO=year, FUENTE_ID="ESC_MAIN", SERVICIO_PROGRAMA=service, MODALIDAD_AGRUPADA="Escolarizadas")
                    if row["TIPO_DATO"] == "TEXTO":
                        row["RESPUESTA_ORIGINAL"] = f"Comentario reservado de {service}"
                    rows.append(row)
            datasets[str(year)] = rows, mapping

        def source(year, modality, _service):
            assert modality == "Escolarizadas"
            return datasets[year]

        def preview(modality, year):
            rows, _ = source(year, modality, "")
            return {"registros": rows, "qa": [{"privado": "diagnóstico interno"}], "fila_inicio": 2, "fila_fin": 4,
                    "fuente_id": "fixture", "periodo_id": year, "version_instrumento": "fixture"}

        people = [{"CARRERA_BASE": "Nutrición", "MODALIDAD_ACADEMICA": "Presencial",
                   "PERSONA_EVALUADA": "Persona fuera del alcance", "ROL_EVALUADO": "Asistente"}]
        self.enterContext(patch.object(calidad_api, "_obtener_base_preview", side_effect=preview))
        self.enterContext(patch.object(institutional_services_api, "_obtener_base_preview", side_effect=preview))
        self.enterContext(patch.object(institutional_services_api, "cargar_mapa_preguntas",
                                      side_effect=lambda source, year: datasets[str(year)][1]))
        self.enterContext(patch.object(kpi_service, "cargar_personas_evaluadas", return_value=people))
        self.enterContext(patch.object(results_report_source, "cargar_personas_evaluadas", return_value=people))
        self.enterContext(patch.object(results_report_source, "_fuente", side_effect=source))
        self.enterContext(patch.object(results_report_source, "_current_labels", side_effect=lambda rows, mapping, *_: (rows, mapping)))

    def test_anonymous_cannot_call_any_quality_endpoint(self):
        paths = [path for path in app.openapi()["paths"] if path.startswith(f"{BASE}/")]
        self.assertGreaterEqual(len(paths), 14)
        for path in paths:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(self.client.get("/api/observacion/resumen").status_code, 401)

    def test_valid_google_account_missing_from_access_is_forbidden(self):
        self.assertEqual(self.google_callback("unlisted").status_code, 403)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)

    def test_inactive_account_is_forbidden_even_after_google_login(self):
        response = self.google_callback("inactive")
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json(), {"detail": "Acceso no autorizado"})

    def test_authorized_login_sets_session_cookie_and_effective_grants(self):
        with self.assertLogs("backend.GATEWAY.auth_api", level="INFO") as diagnostic:
            response = self.google_callback("dg")
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], f"{ORIGIN}/")
        logged = " ".join(diagnostic.output)
        for marker in ("code_present=True", "state_matches=True", "identity_verified=True",
                       "access_granted=True", "session_created=True", "cookie_emitted=True"):
            self.assertIn(marker, logged)
        self.assertNotIn("fixture", logged)
        self.assertNotIn("dg@example.test", logged)
        cookie = next(line for line in response.headers.get_list("set-cookie") if "udl_session" in line)
        for attribute in ("Path=/", "HttpOnly", "Secure", "SameSite=lax", "Max-Age=480"):
            self.assertIn(attribute, cookie)
        me = self.client.get("/api/auth/me").json()
        self.assertEqual((me["rol"], me["alcance"]["tipo"]), ("DG", "global"))
        self.assertNotIn(QUALITY_SIA, me["permisos"])
        self.assertNotIn("sub", me)

    def test_http_cookies_are_allowed_only_for_explicit_loopback_configuration(self):
        local = replace(self.config, public_origin="http://localhost:5173", allow_local_http=True)
        with patch.object(auth_api, "auth_settings", local):
            response = self.client.get("/api/auth/login")
            cookie = response.headers["set-cookie"]
            self.assertTrue(cookie.startswith("udl_login="))
            self.assertIn("HttpOnly", cookie)
            self.assertNotIn("Secure", cookie)
        with self.assertRaises(ValueError):
            replace(self.config, public_origin="http://udl.example.test", allow_local_http=True)

    def test_login_flow_is_single_use_and_bound_to_cookie(self):
        start = self.client.get("/api/auth/login?next_path=//not-udl.test")
        state = parse_qs(urlsplit(start.headers["location"]).query)["state"][0]
        with patch.object(auth_api, "exchange_code", return_value=self.identity("dg")) as exchange:
            self.assertEqual(self.client.get("/api/auth/callback?state=other&code=fixture").status_code, 400)
            exchange.assert_not_called()
            params = {"state": state, "code": "fixture"}
            self.assertEqual(self.client.get("/api/auth/callback", params=params).headers["location"], ORIGIN + "/")
            self.client.cookies.set(auth_api.cookie_name("login"), state)
            self.assertEqual(self.client.get("/api/auth/callback", params=params).status_code, 400)
            exchange.assert_called_once()

    def test_claims_require_issuer_sub_verified_email_audience_and_nonce(self):
        claims = {"iss": ISSUER, "sub": "123", "email": "UDL@example.test", "email_verified": True,
                  "hd": "example.test", "aud": self.config.client_id, "nonce": "nonce"}
        self.assertEqual(verified_identity(claims, self.config, "nonce").email, "udl@example.test")
        for key, value in [("iss", "other"), ("sub", ""), ("email_verified", False), ("aud", "other"),
                           ("nonce", "other"), ("hd", ""), ("azp", "other")]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                verified_identity({**claims, key: value}, self.config, "nonce")

    def test_dg_can_view_global_and_export(self):
        self.mock_quality()
        self.session("dg")
        response = self.client.get(SUMMARY)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["encuestas"], 3)
        self.assertEqual(len(result["filtros"]["servicios"]), 3)
        self.assertIn("Nutrición", {row["Carrera base"] for row in result["respuestas_export"]})

    def test_comments_and_export_require_their_own_grants(self):
        self.mock_quality()
        self.session("dg")
        current = self.repository.lookup("dg@example.test")
        reduced = replace(current, permissions=current.permissions - {QUALITY_COMMENTS, QUALITY_EXPORT})
        with patch.object(self.repository, "lookup", return_value=reduced):
            response = self.client.get(SUMMARY)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["comentarios"], [])
            self.assertEqual(response.json()["qa"], [])
            self.assertNotIn("respuestas_export", response.json())
            response = self.client.get(f"{BASE}/centro-idiomas/actual")
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("Comentario reservado", response.text)

    def test_dc_filter_restricts_both_years_comments_options_catalog_and_export(self):
        self.mock_quality()
        self.session("dc")
        for params, period in [({}, "2026"), ({"periodo": "2025", "historico": "true"}, "2025"),
                               ({"carrera": "Derecho", "modalidad_academica": "Presencial"}, "2026")]:
            response = self.client.get(SUMMARY, params=params)
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertEqual(result["encuestas"], 1)
            self.assertEqual(result["periodo_id"], period)
            self.assertEqual(result["filtros"]["servicios"], ["Derecho"])
            self.assertTrue(result["comentarios"])
            self.assertEqual({row["texto"] for row in result["comentarios"]}, {"Comentario reservado de Derecho"})
            self.assertNotIn("respuestas_export", result)
            self.assertEqual(result["qa"], [])
            self.assertNotIn("Persona fuera del alcance", str(result))
            self.assertNotIn("Nutrición", str(result))
            self.assertNotIn("Ejecutiva", str(result))
        for path in ["/servicios-institucionales/actual", "/servicios-academicos/historico",
                     "/instalaciones/actual", "/ambiente-escolar/historico", "/satisfaccion-recomendacion/actual",
                     "/centro-idiomas/actual", "/educacion-continua/actual"]:
            with self.subTest(path=path):
                response = self.client.get(BASE + path)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertNotIn("Nutrición", response.text)
                self.assertNotIn("Ejecutiva", response.text)

    def test_dc_outside_population_and_explicit_global_are_forbidden(self):
        self.mock_quality()
        self.session("dc")
        for query in ["carrera=Nutrición", "servicio=Nutrición", "modalidad_academica=Ejecutiva", "alcance=global",
                      "servicio=", "carrera=Derecho&carrera=Nutrición", "global=true"]:
            for path in [SUMMARY, PDF, f"{BASE}/instalaciones/historico"]:
                with self.subTest(path=path, query=query):
                    self.assertEqual(self.client.get(f"{path}?{query}").status_code, 403)

    def test_missing_module_and_df_are_denied(self):
        for user in ["dc_without_scope", "df"]:
            self.session(user)
            self.assertEqual(self.client.get("/api/auth/me").json()["modulos"], [])
            self.assertEqual(self.client.get(SUMMARY).status_code, 403)
            self.assertEqual(self.client.get(PDF).status_code, 403)

    def mock_observation(self):
        entries = [
            ("Derecho", "Profesora compartida", "Clase", "15/01/2026"),
            ("Nutrición", "Profesora compartida", "Laboratorio", "15/02/2026"),
            ("Nutrición", "Docente externo", "Laboratorio", "15/02/2026"),
        ]
        rows = pd.DataFrame([{
            "Fecha": date, "Indica el servicio": service, "Nombre del docente": teacher,
            "Tipo de observación": kind, "Asignatura": "Seminario",
            "El docente va acorde con el programa del curso.": 70,
            "Se usaron estrategias para mantener la atención (dinámicas, pausas activas, preguntas detonadoras).": 30,
        } for service, teacher, kind, date in entries])
        cuts = pd.DataFrame([
            {"Corte": "Enero", "Fecha_inicio": "01/01/2026", "Fecha_fin": "31/01/2026"},
            {"Corte": "Febrero", "Fecha_inicio": "01/02/2026", "Fecha_fin": "28/02/2026"},
        ])
        self.enterContext(patch.object(observation_api.repository, "obtener_datos", return_value=(rows, cuts)))

    def test_dc_module_matrix_keeps_assigned_scope(self):
        for user, modules in [
            ("dc", ["encuesta_calidad"]), ("dc_observation", ["observacion_clases"]),
            ("dc_all", ["encuesta_calidad", "observacion_clases"]),
            ("dc_both", ["encuesta_calidad", "observacion_clases"]),
        ]:
            with self.subTest(user=user):
                self.session(user)
                me = self.client.get("/api/auth/me").json()
                self.assertEqual(me["modulos"], modules)
                self.assertEqual(me["alcance"], {"tipo": "asignado", "servicios": ["Derecho"]})
                self.assertEqual(OBSERVATION_VIEW in me["permisos"], "observacion_clases" in modules)
                self.assertNotIn("encuesta_calidad.ver_global", me["permisos"])

    def test_observation_dc_sees_only_assigned_rows_and_dg_keeps_global_counts(self):
        self.mock_observation()
        self.session("dc_all")
        self.assertEqual(self.client.get(f"{OBS}/filtros").json(),
                         {"cortes": ["Enero"], "servicios": ["Derecho"], "tipos": ["Clase"]})
        for path in ["/resumen", "/tendencia", "/casos-prioritarios", "/insights", "/consulta"]:
            with self.subTest(path=path):
                response = self.client.get(OBS + path)
                self.assertEqual(response.status_code, 200)
                self.assertNotIn("Nutrición", response.text)
                self.assertNotIn("Febrero", response.text)
        self.assertEqual(self.client.get(f"{OBS}/resumen").json()["observaciones"], 1)
        self.assertEqual(self.client.get(f"{OBS}/resumen", params={"carrera": "DERECHO"}).json()["observaciones"], 1)
        teacher = self.client.get(f"{OBS}/docente/Profesora compartida").json()
        self.assertEqual([row["servicio"] for row in teacher["historial"]], ["Derecho"])
        for path in ["/docente/Profesora compartida/pdf", "/reporte-ejecutivo/pdf"]:
            response = self.client.get(OBS + path)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.content.startswith(b"%PDF-"))
        self.session("dg")
        self.assertEqual(self.client.get(f"{OBS}/resumen").json()["observaciones"], 3)
        self.assertEqual(self.client.get(f"{OBS}/docente/Profesora compartida").json()["observaciones"], 2)

    def test_observation_dc_denies_outside_selectors_and_unassigned_teachers(self):
        self.mock_observation()
        self.session("dc_all")
        for path in ["/resumen", "/filtros", "/tendencia", "/casos-prioritarios", "/insights", "/consulta",
                     "/buscar-docentes?nombre=Profesora", "/docente/Profesora compartida",
                     "/docente/Profesora compartida/pdf", "/reporte-ejecutivo/pdf"]:
            with self.subTest(path=path):
                url = OBS + path + ("&" if "?" in path else "?") + "servicio=Nutrición"
                self.assertEqual(self.client.get(url).status_code, 403)
        for path in ["/resumen?alcance=global", "/resumen?carrera=Nutrición",
                     "/resumen?servicio=Derecho&servicio=Nutrición", "/resumen?servicio=Todos los servicios",
                     "/consulta?docente=Docente externo", "/buscar-docentes?nombre=Docente externo",
                     "/docente/Docente externo", "/docente/Docente externo/pdf"]:
            self.assertEqual(self.client.get(OBS + path).status_code, 403)
        for payload in [
            {"pregunta": "Resultados de Nutrición"},
            {"pregunta": "Reporte de Docente externo"},
            {"pregunta": "¿Qué aspectos mejoró?", "contexto": {"ultimo_docente": "Docente externo"}},
            {"pregunta": "Dame un resumen", "contexto": {"ultimo_servicio": "Nutrición"}},
        ]:
            self.assertEqual(self.client.post(f"{OBS}/chat", json=payload).status_code, 403)
        chat = self.client.post(f"{OBS}/chat", json={"pregunta": "Dame un resumen"})
        self.assertEqual(chat.status_code, 200)
        self.assertNotIn("Nutrición", chat.text)

    def test_observation_dc_permission_revocation_removes_api_access(self):
        self.session("dc_all")
        me = self.client.get("/api/auth/me").json()
        self.assertIn(OBSERVATION_VIEW, me["permisos"])
        next(row for row in self.rows if row["EMAIL"] == "dc_all@example.test")["MODULOS"] = "encuesta_calidad"
        self.now += self.config.access_ttl_seconds
        self.assertNotIn(OBSERVATION_VIEW, self.client.get("/api/auth/me").json()["permisos"])
        self.assertEqual(self.client.get(f"{OBS}/filtros").status_code, 403)

    def test_global_pdf_cache_never_serves_a_scoped_session(self):
        self.mock_quality()
        with patch.object(results_report_api, "render_report", wraps=results_report_api.render_report) as render:
            self.session("dg")
            dg = self.client.get(PDF)
            self.assertEqual(dg.status_code, 200)
            self.assertEqual(render.call_args.args[0]["surveys"], 3)
            self.session("dc")
            dc = self.client.get(PDF)
            self.assertEqual(dc.status_code, 200)
            self.assertEqual(render.call_args.args[0]["surveys"], 1)
            self.assertIn("Derecho_Presencial", dc.headers["content-disposition"])
            self.assertNotEqual(dc.content, dg.content)
            self.assertEqual(self.client.get(PDF).content, dc.content)
            self.assertEqual(self.client.get(f"{PDF}?carrera=Nutrición").status_code, 403)
            self.assertEqual(render.call_count, 2)

    def test_logout_revokes_a_replayed_cookie_and_rejects_cross_origin(self):
        token = self.session("dg")
        self.assertEqual(self.client.post("/api/auth/logout", headers={"Origin": "https://other.test"}).status_code, 403)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 200)
        self.assertEqual(self.client.post("/api/auth/logout", headers={"Origin": ORIGIN}).status_code, 204)
        self.client.cookies.set(auth_api.cookie_name("session"), token)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)

    def test_expiry_and_access_changes_take_effect_on_the_server(self):
        self.session("dg")
        self.assertEqual(self.client.get("/api/auth/me").status_code, 200)
        self.rows[0]["ACTIVO"] = "NO"
        self.now += 29
        self.assertEqual(self.client.get("/api/auth/me").status_code, 200)
        self.now += 1
        self.assertEqual(self.client.get("/api/auth/me").status_code, 403)
        self.now += self.config.session_seconds
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)
