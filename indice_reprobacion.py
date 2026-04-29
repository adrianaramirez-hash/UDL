# (SE MANTIENE TODO IGUAL HASTA KPIs Y FUNCIONES)

# 🔴 IMPORTANTE:
# SOLO TE VOY A MOSTRAR LA PARTE QUE CAMBIAMOS (RESUMEN EJECUTIVO)
# TODO LO DEMÁS QUEDA IGUAL QUE TU ÚLTIMA VERSIÓN

# =====================================================
# RESUMEN EJECUTIVO (ACTUALIZADO)
# =====================================================
if vista_modulo == "Resumen ejecutivo":
    st.subheader("Resumen ejecutivo")

    # ===============================
    # 1. MATERIAS CRÍTICAS (PRIMERO)
    # ===============================
    st.markdown("### 1. Materias que requieren atención prioritaria")

    tm = top_materias(f, 10)

    if tm.empty:
        st.info("No hay materias suficientes para mostrar.")
    else:
        st.altair_chart(grafica_top_materias(tm), use_container_width=True)
        st.dataframe(tm, use_container_width=True, hide_index=True)

    st.divider()

    # ===============================
    # 2. COMPARATIVO POR CARRERA
    # ===============================
    st.markdown("### 2. Carreras con mayor problema de reprobación")

    resumen = resumen_por_carrera(f)

    if len(resumen) < 2:
        st.info("Solo hay una carrera en el filtro actual.")
    else:
        st.altair_chart(grafica_carreras(resumen), use_container_width=True)

    st.dataframe(
        resumen,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Promedio reprobatorio": st.column_config.NumberColumn(format="%.2f"),
            "Materias reprobadas por alumno": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    st.caption(
        "Aquí puedes comparar qué carreras tienen mayor cantidad de alumnos afectados "
        "y mayor carga de reprobación."
    )

    st.divider()

    # ===============================
    # 3. TENDENCIA POR CICLO (AL FINAL)
    # ===============================
    st.markdown("### 3. Tendencia de reprobación en el tiempo")

    hist = historico_por_ciclo(f)

    st.altair_chart(
        grafica_historico(hist, f"Alumnos con reprobación — {carrera_txt}"),
        use_container_width=True,
    )

    st.dataframe(hist, use_container_width=True, hide_index=True)
