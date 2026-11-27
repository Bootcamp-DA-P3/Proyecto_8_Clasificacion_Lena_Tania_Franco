"""Aplicación Streamlit — Predictor de inclusión financiera
Proyecto 8: Clasificación binaria con Global Findex 2025.

Predice si una persona tiene cuenta financiera (account = 1) o no (account = 0)
a partir de su perfil demográfico y de contexto.

Uso:
    streamlit run app/app.py
"""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# ── Rutas del proyecto ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
DATA_RAW = ROOT / "data" / "raw" / "findex_microdata_2025_labelled_update112425.csv"
MODELS_DIR = ROOT / "models"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
MODEL_PATH = MODELS_DIR / "best_model.joblib"

sys.path.insert(0, str(SRC_DIR))
from preprocess import (  # noqa: E402
    limpiar,
    NUM_FEATURES,
    BIN_FEATURES,
    CAT_FEATURES,
)

# ── Regiones (7 regiones del Banco Mundial en Findex 2025) ───────────────────
REGIONES = [
    "High income",
    "Sub-Saharan Africa (excluding high income)",
    "Europe & Central Asia (excluding high income)",
    "Latin America & Caribbean (excluding high income)",
    "East Asia & Pacific (excluding high income)",
    "Middle East & North Africa (excluding high income)",
    "South Asia",
]


# ── Carga del mapeo economías → región (cacheado) ───────────────────────────
@st.cache_data(show_spinner="Cargando lista de economías…")
def cargar_economias() -> pd.DataFrame:
    """Lee las columnas mínimas del CSV para construir el desplegable de países."""
    df = pd.read_csv(
        DATA_RAW,
        usecols=["economycode", "economy", "regionwb"],
        low_memory=False,
    )
    return (
        df.drop_duplicates(subset="economycode")
        .sort_values("economy")
        .reset_index(drop=True)
    )


# ── Carga del modelo y preprocesador (cacheados) ─────────────────────────────
@st.cache_resource(show_spinner=False)
def cargar_modelo():
    """Carga el modelo final (o crea un modelo dummy si no existe)."""
    import json
    import joblib
    from sklearn.dummy import DummyClassifier

    preprocesador = None
    if PREPROCESSOR_PATH.exists():
        preprocesador = joblib.load(PREPROCESSOR_PATH)

    # Buscar modelo final (modelo_final.joblib o best_model.joblib)
    ruta_modelo = None
    for cand in [MODELS_DIR / "modelo_final.joblib", MODELS_DIR / "best_model.joblib"]:
        if cand.exists():
            ruta_modelo = cand
            break

    meta = {}
    meta_path = MODELS_DIR / "modelo_final_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    if ruta_modelo is not None:
        model = joblib.load(ruta_modelo)
        es_dummy = False
        nombre_modelo = meta.get("modelo", "Modelo entrenado")
        umbral = meta.get("umbral_sin_cuenta", 0.5)
    else:
        model = DummyClassifier(strategy="most_frequent")
        n_features = len(NUM_FEATURES) + len(BIN_FEATURES) + len(CAT_FEATURES)
        X_dummy = pd.DataFrame(
            [[0] * n_features] * 100, columns=NUM_FEATURES + BIN_FEATURES + CAT_FEATURES
        )
        y_dummy = [0] * 26 + [1] * 74
        model.fit(X_dummy, y_dummy)
        es_dummy = True
        nombre_modelo = "DummyClassifier (prueba)"
        umbral = 0.5

    return model, preprocesador, es_dummy, nombre_modelo, umbral, meta


# ── Función de predicción ─────────────────────────────────────────────────────
def predecir(
    fila_raw: dict, model, preprocesador, umbral: float = 0.5
) -> tuple[int, float]:
    """Recibe el dict crudo del formulario y devuelve (clase, probabilidad_clase_1)."""
    # 1. DataFrame con los códigos originales del CSV
    df_raw = pd.DataFrame([fila_raw])

    # 2. Recodificación con `limpiar` de src/preprocess.py (female, emp_in, urbanicity → urban)
    df_limpio = limpiar(df_raw)

    # 3. Selección de features en el orden esperado
    X = df_limpio[NUM_FEATURES + BIN_FEATURES + CAT_FEATURES]

    # 4. Comprobar si el modelo es un Pipeline completo que ya incluye el preprocesador
    es_pipeline_completo = hasattr(model, "named_steps") and (
        "prep" in model.named_steps or "preprocessor" in model.named_steps
    )

    if es_pipeline_completo:
        X_input = X
    elif preprocesador is not None:
        try:
            X_input = preprocesador.transform(X)
        except Exception:
            X_input = X
    else:
        X_input = X

    # 5. Predicción con probabilidades y umbral
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_input)[0]
        # Identificar índices de las clases
        clases = list(model.classes_) if hasattr(model, "classes_") else [0, 1]
        idx_0 = clases.index(0) if 0 in clases else 0
        idx_1 = clases.index(1) if 1 in clases else 1

        p_sin_cuenta = float(probs[idx_0])
        p_con_cuenta = float(probs[idx_1])

        # Decisión con umbral calibrado sobre P(sin cuenta)
        if p_sin_cuenta >= umbral:
            clase = 0
        else:
            clase = 1
        prob = p_con_cuenta
    else:
        clase = int(model.predict(X_input)[0])
        prob = 1.0 if clase == 1 else 0.0

    return clase, prob


# ═════════════════════════════════════════════════════════════════════════════
# INTERFAZ STREAMLIT
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Predictor Inclusión Financiera",
    page_icon="🏦",
    layout="centered",
)

# ── Cabecera ─────────────────────────────────────────────────────────────────
st.title("🏦 Predictor de Inclusión Financiera")
st.markdown(
    "Esta herramienta estima la probabilidad de que una persona tenga una "
    "**cuenta financiera** (banco, cooperativa o cuenta móvil) a partir de "
    "su perfil demográfico y de contexto. Datos: **Global Findex 2025**."
)

# ── Carga de recursos ────────────────────────────────────────────────────────
model, preprocesador, es_dummy, nombre_modelo, umbral, meta = cargar_modelo()
economias_df = cargar_economias()

if es_dummy:
    st.warning(
        "🧪 **Modelo de prueba** — Los resultados no son predictivos. "
        "El modelo final se integrará cuando exista `models/modelo_final.joblib`.",
        icon="⚠️",
    )
else:
    st.info(
        f"🎯 **Modelo activo:** `{nombre_modelo}` · "
        f"**Umbral óptimo P(sin cuenta):** `{umbral:.3f}`",
        icon="ℹ️",
    )

st.divider()

# ── Formulario ───────────────────────────────────────────────────────────────
st.subheader("📋 Perfil de la persona")

with st.form("formulario_prediccion"):
    col1, col2 = st.columns(2)

    with col1:
        edad = st.number_input(
            "Edad",
            min_value=15,
            max_value=100,
            value=35,
            step=1,
            help="Rango válido: 15–100 años.",
        )

        sexo = st.radio(
            "Sexo",
            options=[1, 2],
            format_func=lambda x: "Mujer" if x == 1 else "Hombre",
            horizontal=True,
        )

        nivel_edu = st.selectbox(
            "Nivel educativo",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "1 — Primaria o menos",
                2: "2 — Secundaria",
                3: "3 — Terciaria o más",
            }[x],
        )

        quintil_ing = st.selectbox(
            "Quintil de ingreso (dentro de su país)",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: f"Q{x} — {'Más bajo' if x == 1 else 'Más alto' if x == 5 else ''}".rstrip(" —"),
        )

    with col2:
        situacion_lab = st.radio(
            "Situación laboral",
            options=[1, 2],
            format_func=lambda x: "En la fuerza laboral" if x == 1 else "Fuera de la fuerza laboral",
        )

        zona = st.radio(
            "Zona de residencia",
            options=[1, 2],
            format_func=lambda x: "Rural" if x == 1 else "Urbana",
            horizontal=True,
        )

        internet = st.radio(
            "¿Usa internet?",
            options=[1, 0],
            format_func=lambda x: "Sí" if x == 1 else "No",
            horizontal=True,
        )

        # ── Región → País (cascada) ─────────────────────────────────────────
        region = st.selectbox("Región del Banco Mundial", options=REGIONES)

    # País filtrado por región (ocupa el ancho completo)
    paises_region = economias_df[economias_df["regionwb"] == region]
    opciones_pais = list(zip(paises_region["economycode"], paises_region["economy"]))

    if opciones_pais:
        pais_code = st.selectbox(
            "País (economía)",
            options=[c for c, _ in opciones_pais],
            format_func=lambda c: next(
                (n for code, n in opciones_pais if code == c), c
            ),
        )
    else:
        pais_code = st.text_input("Código de economía (ISO-3)", value="USA")

    submitted = st.form_submit_button("🔮 Predecir", use_container_width=True)

# ── Resultado ─────────────────────────────────────────────────────────────────
if submitted:
    fila_raw = {
        "age": edad,
        "female": sexo,          # 1=Mujer, 2=Hombre → limpiar() convierte a 1/0
        "educ": nivel_edu,
        "inc_q": quintil_ing,
        "emp_in": situacion_lab, # 1=En fuerza, 2=Fuera → limpiar() convierte a 1/0
        "urbanicity": zona,      # 1=Rural, 2=Urbano → limpiar() convierte a 0/1 y renombra a urban
        "internet_use": internet, # ya 0/1
        "regionwb": region,
        "economycode": pais_code,
    }

    with st.spinner("Calculando predicción…"):
        clase, prob = predecir(fila_raw, model, preprocesador, umbral)

    st.divider()
    st.subheader("📊 Resultado")

    if clase == 1:
        st.success(
            f"### ✅ Bancarizado/a\n"
            f"Probabilidad estimada de tener cuenta financiera: **{prob:.1%}**"
        )
    else:
        st.error(
            f"### ❌ No bancarizado/a\n"
            f"Probabilidad estimada de tener cuenta financiera: **{prob:.1%}**"
        )

    # Barra de probabilidad
    st.progress(prob, text=f"Probabilidad de inclusión financiera: {prob:.1%}")

    with st.expander("Ver datos enviados al modelo"):
        st.json(fila_raw)

    if es_dummy:
        st.caption(
            "⚠️ Resultado generado con modelo dummy (siempre predice la clase mayoritaria). "
            "Reemplazar con `models/modelo_final.joblib` para predicciones reales."
        )

# ── Pie de página ─────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Proyecto 8 · Bootcamp Data Analytics · "
    "Global Findex 2025 — Banco Mundial · "
    "Equipo: Lena, Tania, Franco"
)
