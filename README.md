# Proyecto 8: Clasificación de inclusión financiera

Equipo: Lena, Tania, Franco

Modelo de clasificación que predice si una persona tiene cuenta financiera (`account`) a partir de su perfil, usando los microdatos de Global Findex 2025.

## 🚀 Aplicación web (Streamlit)

La aplicación interactiva está desplegada y disponible públicamente:
👉 **[Predictor de Inclusión Financiera en Streamlit Cloud](https://findex-financial-inclusion-predictor.streamlit.app/)**

Para ejecutarla en local:
```bash
streamlit run app/app.py
```

## Estructura

```
data/raw/         CSV original de Findex 2025
data/processed/   Split train/test generado por src/preprocess.py (no se sube)
notebooks/        EDA y experimentos
src/              Código de preprocesado y entrenamiento
app/              App de predicción (Streamlit)
models/           Modelos entrenados
reports/          Informe técnico y presentaciones
tests/            Tests
docs/             Plan de tareas y reglas del equipo
scripts/          Utilidades (creación del tablero Trello)
```

## Documentación

- [Plan de tareas](docs/TAREAS.md)
- [Cómo trabajamos: ramas, commits y PRs](docs/CONTRIBUTING.md)
