"""Transport, isolation, TTL, concurrency and filter-safe PDF caching regressions."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from gspread.exceptions import APIError
from backend.FUENTES.calidad.read_snapshot import ReadSnapshot, SnapshotHTTPClient, active_snapshot, reading
from backend.FUENTES.calidad.google_sheets import abrir_spreadsheet


def response(code=200):
    return SimpleNamespace(ok=code == 200, status_code=code,
                           json=lambda: {"error": {"code": code, "message": "quota"}})


class SnapshotTransportTests(unittest.TestCase):
    def client(self, answers):
        session = Mock()
        session.request.side_effect = answers
        return SnapshotHTTPClient(None, session=session), session

    def test_duplicate_gets_reuse_response_but_render_options_remain_distinct(self):
        first, second = response(), response()
        client, session = self.client([first, second])
        snapshot = ReadSnapshot()
        with reading(snapshot):
            self.assertIs(client.request("get", "https://sheets/range", params={"option": "formatted"}), first)
            self.assertIs(client.request("get", "https://sheets/range", params={"option": "formatted"}), first)
            self.assertIs(client.request("get", "https://sheets/range", params={"option": "raw"}), second)
        self.assertEqual((session.request.call_count, snapshot.network_requests, snapshot.cache_hits), (2, 2, 1))
        self.assertIsNone(active_snapshot())

    def test_non_pdf_reads_are_not_cached(self):
        client, session = self.client([response(), response()])
        client.request("get", "https://sheets/range")
        client.request("get", "https://sheets/range")
        self.assertEqual(session.request.call_count, 2)

    def test_one_bounded_429_retry_then_cached_success(self):
        client, session = self.client([response(429), response()])
        snapshot = ReadSnapshot()
        with reading(snapshot), patch("backend.FUENTES.calidad.read_snapshot.time.sleep") as sleep, self.assertLogs("backend.FUENTES.calidad.read_snapshot", level="WARNING"):
            client.request("get", "https://sheets/range")
            client.request("get", "https://sheets/range")
        self.assertEqual((session.request.call_count, snapshot.retries), (2, 1))
        sleep.assert_called_once_with(2)

    def test_second_429_propagates_and_no_failed_response_is_cached(self):
        client, session = self.client([response(429), response(429), response()])
        snapshot = ReadSnapshot()
        with reading(snapshot), patch("backend.FUENTES.calidad.read_snapshot.time.sleep"), self.assertLogs("backend.FUENTES.calidad.read_snapshot", level="WARNING"):
            with self.assertRaises(APIError):
                client.request("get", "https://sheets/range")
        self.assertEqual(session.request.call_count, 2)
        self.assertFalse(snapshot.responses)

    def test_permission_failure_is_not_retried(self):
        client, session = self.client([response(403)])
        with reading(ReadSnapshot()), patch("backend.FUENTES.calidad.read_snapshot.time.sleep") as sleep:
            with self.assertRaises(APIError):
                client.request("get", "https://sheets/range")
        self.assertEqual(session.request.call_count, 1)
        sleep.assert_not_called()

    def test_workbook_open_is_reused_only_inside_snapshot(self):
        client = Mock()
        with reading(ReadSnapshot()):
            first = abrir_spreadsheet(client, "master")
            self.assertIs(abrir_spreadsheet(client, "master"), first)
            abrir_spreadsheet(client, "archive")
        self.assertEqual(client.open_by_key.call_count, 2)
        abrir_spreadsheet(client, "master")
        self.assertEqual(client.open_by_key.call_count, 3)


class PdfSnapshotCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from backend.TESTS.test_results_report_pdf import ResultsReportTests
        ResultsReportTests.setUpClass()
        cls.fixture = ResultsReportTests
        with patch("backend.REPOSITORIES.calidad.calidad_repository.CalidadRepository"):
            from backend.GATEWAY import results_report_api
            from backend.SERVICES.calidad import results_report_source
        cls.api, cls.sources = results_report_api, results_report_source

    def setUp(self):
        self.sources.clear_snapshots()
        self.addCleanup(self.sources.clear_snapshots)
        patcher = patch.dict(self.sources.calidad_api._cache_expira, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def source(self, year, source, service):
        self.assertEqual(service, "")  # Share raw source only; never filtered results.
        return (self.fixture.now, self.fixture.now_map) if year == "2026" else (self.fixture.old, self.fixture.old_map)

    def test_two_global_and_two_filtered_requests_share_source_not_results(self):
        with patch.object(self.sources, "_fuente", side_effect=self.source) as source, patch.object(self.api, "render_report", wraps=self.api.render_report) as render:
            first = self.api.download_report()
            second = self.api.download_report()
            filtered = self.api.download_report(carrera="Derecho", modalidad_academica="Presencial")
            repeated = self.api.download_report(carrera="Derecho", modalidad_academica="Presencial")
        self.assertEqual(source.call_count, 2)
        self.assertEqual(render.call_count, 2)
        self.assertEqual(first.body, second.body)
        self.assertEqual(filtered.body, repeated.body)
        self.assertNotEqual(first.body, filtered.body)
        self.assertIn("Derecho_Presencial", filtered.headers["content-disposition"])
        self.assertEqual(first.headers["x-pdf-snapshot-cache"], "MISS")
        self.assertEqual(filtered.headers["x-pdf-result-cache"], "MISS")
        self.assertEqual(repeated.headers["x-pdf-result-cache"], "HIT")

    def test_every_filter_participates_in_pdf_result_key(self):
        scopes = [{}, {"servicio": "Derecho"}, {"carrera": "Derecho"}, {"modalidad_academica": "Presencial"},
                  {"idioma": "Inglés"}, {"tipo_curso": "Intensivo"}, {"participacion": "si"}]
        with patch.object(self.sources, "_fuente", side_effect=self.source):
            for scope in scopes:
                first, second = self.api.download_report(**scope), self.api.download_report(**scope)
                self.assertEqual(first.headers["x-pdf-result-cache"], "MISS", scope)
                self.assertEqual(second.headers["x-pdf-result-cache"], "HIT", scope)
                self.assertEqual(first.body, second.body)

    def test_projections_do_not_mutate_shared_normalized_inputs(self):
        original = deepcopy((self.fixture.now, self.fixture.now_map, self.fixture.old, self.fixture.old_map))
        with patch.object(self.sources, "_fuente", side_effect=self.source):
            self.api.download_report()
            self.api.download_report(carrera="Derecho", modalidad_academica="Presencial")
        self.assertEqual(original, (self.fixture.now, self.fixture.now_map, self.fixture.old, self.fixture.old_map))

    def test_expired_snapshot_is_reloaded(self):
        with patch.object(self.sources, "_fuente", side_effect=self.source) as source, patch.object(self.sources.time, "monotonic", return_value=100) as clock:
            with self.sources.source_snapshot() as first:
                identifier = first.snapshot.identifier
            clock.return_value = 161
            with self.sources.source_snapshot() as second:
                self.assertFalse(second.hit)
                self.assertNotEqual(identifier, second.snapshot.identifier)
        self.assertEqual(source.call_count, 4)

    def test_source_snapshot_does_not_extend_existing_dataset_expiration(self):
        with patch.object(self.sources, "_fuente", side_effect=self.source), patch.object(self.sources.time, "monotonic", return_value=100), patch.dict(self.sources.calidad_api._cache_expira, {"2026:Escolarizadas": 110, "2025:Escolarizadas": 120}):
            with self.sources.source_snapshot() as access:
                self.assertEqual(access.snapshot.expires, 110)

    def test_failed_refresh_never_serves_expired_pdf(self):
        with patch.object(self.sources, "_fuente", side_effect=self.source), patch.object(self.sources.time, "monotonic", return_value=100) as clock:
            self.api.download_report()
            clock.return_value = 161
            with patch.object(self.sources, "_fuente", side_effect=RuntimeError("source unavailable")):
                with self.assertRaises(RuntimeError):
                    self.api.download_report()
        self.assertFalse(self.sources._snapshots)

    def test_concurrent_requests_load_source_once_and_restore_context(self):
        def request(_):
            with self.sources.source_snapshot() as access:
                self.assertIs(active_snapshot(), access.snapshot.transport)
                identifier = access.snapshot.identifier
            self.assertIsNone(active_snapshot())
            return identifier
        with patch.object(self.sources, "_fuente", side_effect=self.source) as source:
            with ThreadPoolExecutor(max_workers=3) as executor:
                identifiers = list(executor.map(request, range(3)))
        self.assertEqual(len(set(identifiers)), 1)
        self.assertEqual(source.call_count, 2)
