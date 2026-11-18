from backend.CORE.auth.policy import Identity, grants_for
from backend.GATEWAY.auth_api import Principal, current_principal


def authorize_dg(app):
    """Give an isolated gateway test the same grants as an authenticated DG."""
    principal = Principal(
        Identity("https://accounts.google.com", "fixture", "dg@example.test", "DG de prueba"),
        grants_for({"ROL": "DG", "ACTIVO": "SI", "MODULOS": "encuesta_calidad"}),
    )
    app.dependency_overrides[current_principal] = lambda: principal
