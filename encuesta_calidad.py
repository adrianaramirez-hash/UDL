@st.cache_data(show_spinner=False)
def _load_from_gsheets(sheet_id: str):
    # Auth desde secrets
    sa = dict(st.secrets["gcp_service_account"])
    gc = gspread.service_account_from_dict(sa)

    sh = gc.open_by_key(sheet_id)

    # Normalizador para comparar títulos de pestañas
    def norm(x: str) -> str:
        return str(x).strip().lower().replace(" ", "").replace("_", "")

    # Mapa de pestañas que necesitamos
    targets = {
        "Respuestas": "Respuestas",
        "Mapa_Preguntas": "Mapa_Preguntas",
        "Catalogo_Servicio": "Catalogo_Servicio",
    }
    targets_norm = {k: norm(v) for k, v in targets.items()}

    # Lista real de pestañas visibles para el Service Account
    all_ws = sh.worksheets()
    titles = [ws.title for ws in all_ws]
    titles_norm = {norm(t): t for t in titles}

    # Busca por normalizado (tolerante)
    missing = []
    resolved = {}
    for key, tnorm in targets_norm.items():
        if tnorm in titles_norm:
            resolved[key] = titles_norm[tnorm]
        else:
            missing.append(targets[key])

    if missing:
        # Esto te dirá si la app está abriendo otro Sheet o si hay un detalle de nombre
        raise ValueError(
            "No encontré estas pestañas: "
            + ", ".join(missing)
            + " | Pestañas disponibles para el service account: "
            + ", ".join(titles)
        )

    ws_resp = sh.worksheet(resolved["Respuestas"])
    ws_map = sh.worksheet(resolved["Mapa_Preguntas"])
    ws_cat = sh.worksheet(resolved["Catalogo_Servicio"])

    df = pd.DataFrame(ws_resp.get_all_records())
    mapa = pd.DataFrame(ws_map.get_all_records())
    catalogo = pd.DataFrame(ws_cat.get_all_records())

    return df, mapa, catalogo
