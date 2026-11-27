# Proyecto 8: Plan de tareas del equipo

Equipo: Lena, Tania, Franco

## Problema propuesto

Predecir si una persona tiene cuenta financiera (`account`: 1 = sí, 0 = no) a partir de su perfil: edad, género, educación, ingreso, empleo, zona urbana/rural, región o país y uso de internet.

**Uso de negocio:** detectar perfiles excluidos del sistema financiero para orientar campañas de inclusión financiera.

**Dataset:** Global Findex 2025 (`data/raw/findex_microdata_2025_labelled_update112425.csv`), 144.090 filas, 199 columnas, 140 economías.

**Por qué `account`:**

- No tiene nulos y es binaria.
- Desbalance moderado (74% sí / 26% no), así que se evalúa con F1, recall y ROC-AUC, no solo accuracy.

**Riesgo principal (leakage):** `account_fin`, `account_mob`, `dig_account` y casi todas las `fin*` se derivan de tener cuenta. Se excluyen de las features. La lista definitiva está en `src/preprocess.py`, que usa una lista blanca de columnas.

## Requisitos clave (README Factoria)

- Overfitting < 5%.
- App que reciba datos y devuelva la clase predicha (Streamlit, Gradio, Dash o API).
- Repo con ramas y commits limpios, informe técnico, presentación de negocio, presentación técnica y tablero Trello.
- Métricas del informe: accuracy, precision, recall, F1, curva ROC, matriz de confusión, feature importance y análisis de errores.

---

## Fase 0: Todos juntos (1 sesión)

- [ ] Confirmar el target (`account`).
- [ ] Cerrar la lista de columnas prohibidas por leakage.
- [ ] Fijar convenciones:
  - `random_state=42` en todo.
  - Split estratificado 80/20, único y guardado en `data/processed`.
  - Overfitting = diferencia de F1 entre train y test, debe ser < 5%.
- [x] Crear la estructura del repo:

```
data/raw/
data/processed/
notebooks/
src/
app/
models/
reports/
tests/
docs/
```

---

## Franco: Datos, repo e infraestructura

### Nivel esencial

- [x] Configurar ramas: `main`, `dev` y una rama por feature (`feature/eda`, `feature/modelo`, etc.).
- [x] Definir reglas de commits y crear el tablero Trello.
- [x] Decidir qué hacer con el CSV de 50 MB: Git LFS o `.gitignore` con instrucciones de descarga.
- [x] `src/preprocess.py`:
  - Selección de columnas (sin leakage).
  - Tratamiento de nulos.
  - Recodificar `female` (1/2 a 0/1).
  - Encoding de categóricas.
  - Split train/test guardado en `data/processed`.

### Nivel medio

- [ ] Pipeline de ingesta de datos para reentrenamiento.
- [ ] Sistema de feedback de la app en producción.

### Nivel avanzado

- [ ] Dockerizar la app.
- [ ] Base de datos para los datos que recoja la app (SQL o MongoDB).
- [ ] Despliegue en la nube (Render o similar).
- [x] Tests unitarios del preprocesado (`tests/test_preprocess.py`, se ejecutan con `pytest`).

---

## Lena: EDA, visualización y negocio

### Nivel esencial

- [ ] Notebook de EDA (`notebooks/01_eda.ipynb`):
  - Distribución del target.
  - Histogramas de variables numéricas.
  - Matriz de correlación.
  - Nulos por columna.
  - Target cruzado con género, ingreso, educación y región.
- [ ] Extraer insights de negocio (qué perfiles concentran a los no bancarizados).
- [ ] App en Streamlit (`app/`):
  - Formulario con el perfil de la persona.
  - Resultado: clase predicha y probabilidad.
  - Empezar con un modelo dummy hasta tener el modelo final.
- [ ] Presentación de negocio.

---

## Tania: Modelado y evaluación

### Nivel esencial

- [ ] Baseline: regresión logística con `class_weight="balanced"`.
- [ ] Comparar modelos: árbol de decisión, Random Forest, Gradient Boosting, XGBoost o LightGBM.
- [ ] Métricas para el informe: accuracy, precision, recall, F1, curva ROC, matriz de confusión, feature importance, análisis de errores.
- [ ] Verificar overfitting < 5%.
- [ ] Guardar el modelo final en `models/` con `joblib`.

### Nivel medio

- [ ] Validación cruzada (K-Fold estratificado).
- [ ] Tuning de hiperparámetros (GridSearch u Optuna).

### Nivel avanzado

- [ ] Tests unitarios de métricas mínimas del modelo.

---

## Compartido (cierre)

- [ ] Informe técnico: cada uno escribe su sección.
- [ ] Presentación técnica: cada uno explica su parte del código.
- [ ] Revisión cruzada de PRs antes de mergear a `main`.

---

## Dependencias entre tareas

| Tarea | Depende de | Mientras tanto |
|---|---|---|
| EDA (Lena) | Nada | Trabajar sobre el CSV crudo |
| Modelado (Tania) | Preprocesado (Franco), ya disponible | Usar `train_raw.csv` / `train.csv` generados por `src/preprocess.py` |
| App (Lena) | Modelo final (Tania) | Usar un modelo dummy |
| Docker y deploy (Franco) | App (Lena) | Preparar Dockerfile y base de datos |

## Niveles de entrega

No hay rúbrica con puntajes: la guía son los niveles del README de Factoría (esencial, medio, avanzado, experto). Hay que cumplir lo esencial completo y avanzar en los siguientes niveles hasta donde el equipo decida llegar.

- [ ] Nivel esencial.
- [ ] Nivel medio.
- [ ] Nivel avanzado.
- [ ] Nivel experto.

**Nivel objetivo del equipo:** por definir.

### Nivel experto (aún sin asignar)

- [ ] Experimentar con redes neuronales o desplegarlas.
- [ ] MLOps: test A/B, monitoreo de data drift y reemplazo automático del modelo solo si el nuevo supera métricas predefinidas.

## Pendiente

- Fechas de entrega: el README de Factoría no las trae. Ajustar este plan cuando estén.
