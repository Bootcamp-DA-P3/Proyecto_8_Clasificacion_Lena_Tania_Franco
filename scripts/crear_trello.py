"""Crea el tablero Trello del Proyecto 8 a partir de docs/TAREAS.md.

Uso:
    python scripts/crear_trello.py            # vista previa, no crea nada
    python scripts/crear_trello.py --crear    # crea el tablero en Trello

Requiere variables de entorno (no se guardan en el repo):
    TRELLO_KEY    API key   (https://trello.com/power-ups/admin)
    TRELLO_TOKEN  token     (enlace "Token" junto a la API key)
"""

import json
import os
import sys
import urllib.parse
import urllib.request

API = "https://api.trello.com/1"
NOMBRE_TABLERO = "Proyecto 8 - Clasificacion Findex (Lena, Tania, Franco)"

LISTAS = ["Backlog", "Por hacer", "En curso", "En revision (PR)", "Hecho"]

ETIQUETAS = {
    "Franco": "blue",
    "Lena": "pink",
    "Tania": "purple",
    "Todos": "black",
    "Esencial": "green",
    "Medio": "yellow",
    "Avanzado": "red",
    "Bloqueada": "orange",
}

# (lista, titulo, etiquetas, descripcion, checklist)
TARJETAS = [
    # Fase 0
    ("Por hacer", "Fase 0: Confirmar target (account)", ["Todos", "Esencial"],
     "account: 1 = tiene cuenta, 0 = no. Binaria, sin nulos, 74/26.", []),
    ("Por hacer", "Fase 0: Cerrar lista de columnas prohibidas (leakage)", ["Todos", "Esencial"],
     "Revisar codebook Findex. Excluir account_fin, account_mob, dig_account y casi todas las fin*.", []),
    ("Por hacer", "Fase 0: Fijar convenciones", ["Todos", "Esencial"], "",
     ["random_state=42 en todo",
      "Split estratificado 80/20 unico, guardado en data/processed",
      "Overfitting = diferencia F1 train vs test < 5%"]),
    ("Por hacer", "Fase 0: Crear estructura del repo", ["Todos", "Esencial"], "",
     ["data/raw/", "data/processed/", "notebooks/", "src/", "app/",
      "models/", "reports/", "tests/", "docs/"]),

    # Franco
    ("Por hacer", "Configurar ramas (main, dev, feature/*)", ["Franco", "Esencial"],
     "Una rama por feature: feature/eda, feature/modelo, etc.", []),
    ("Por hacer", "Reglas de commits + tablero Trello", ["Franco", "Esencial"], "", []),
    ("Por hacer", "Decidir manejo del CSV de 50 MB", ["Franco", "Esencial"],
     "Git LFS o .gitignore con instrucciones de descarga.", []),
    ("Por hacer", "src/preprocess.py", ["Franco", "Esencial"],
     "Desbloquea el modelado de Tania.",
     ["Seleccion de columnas (sin leakage)", "Tratamiento de nulos",
      "Recodificar female (1/2 a 0/1)", "Encoding de categoricas",
      "Split train/test guardado en data/processed"]),
    ("Backlog", "Pipeline de ingesta para reentrenamiento", ["Franco", "Medio"], "", []),
    ("Backlog", "Sistema de feedback de la app en produccion", ["Franco", "Medio"], "", []),
    ("Backlog", "Dockerizar la app", ["Franco", "Avanzado", "Bloqueada"],
     "Depende de: App (Lena). Mientras tanto preparar Dockerfile.", []),
    ("Backlog", "Base de datos para datos de la app (SQL o MongoDB)", ["Franco", "Avanzado"], "", []),
    ("Backlog", "Despliegue en la nube (Render o similar)", ["Franco", "Avanzado", "Bloqueada"],
     "Depende de: App (Lena).", []),
    ("Backlog", "Tests unitarios del preprocesado", ["Franco", "Avanzado"], "", []),

    # Lena
    ("Por hacer", "Notebook EDA (notebooks/01_eda.ipynb)", ["Lena", "Esencial"],
     "No depende de nada: trabajar sobre el CSV crudo.",
     ["Distribucion del target", "Histogramas de variables numericas",
      "Matriz de correlacion", "Nulos por columna",
      "Target cruzado con genero, ingreso, educacion y region"]),
    ("Por hacer", "Insights de negocio (perfiles no bancarizados)", ["Lena", "Esencial"], "", []),
    ("Por hacer", "App Streamlit (app/)", ["Lena", "Esencial"],
     "Depende de: modelo final (Tania). Mientras tanto usar modelo dummy.",
     ["Formulario con el perfil de la persona",
      "Resultado: clase predicha y probabilidad",
      "Empezar con modelo dummy"]),
    ("Por hacer", "Presentacion de negocio", ["Lena", "Esencial"], "", []),

    # Tania
    ("Por hacer", "Baseline: regresion logistica (class_weight=balanced)", ["Tania", "Esencial", "Bloqueada"],
     "Depende de: preprocess.py (Franco). Mientras tanto preparar plantilla de metricas.", []),
    ("Por hacer", "Comparar modelos", ["Tania", "Esencial"], "",
     ["Arbol de decision", "Random Forest", "Gradient Boosting", "XGBoost o LightGBM"]),
    ("Por hacer", "Metricas para el informe", ["Tania", "Esencial"], "",
     ["Accuracy", "Precision", "Recall", "F1", "Curva ROC",
      "Matriz de confusion", "Feature importance", "Analisis de errores"]),
    ("Por hacer", "Verificar overfitting < 5%", ["Tania", "Esencial"], "", []),
    ("Por hacer", "Guardar modelo final en models/ (joblib)", ["Tania", "Esencial"], "", []),
    ("Backlog", "Validacion cruzada (K-Fold estratificado)", ["Tania", "Medio"], "", []),
    ("Backlog", "Tuning de hiperparametros (GridSearch u Optuna)", ["Tania", "Medio"], "", []),
    ("Backlog", "Tests unitarios de metricas minimas del modelo", ["Tania", "Avanzado"], "", []),

    # Compartido
    ("Backlog", "Informe tecnico", ["Todos", "Esencial"], "Cada uno escribe su seccion.",
     ["Seccion Franco", "Seccion Lena", "Seccion Tania"]),
    ("Backlog", "Presentacion tecnica", ["Todos", "Esencial"], "Cada uno explica su parte del codigo.", []),
    ("Backlog", "Revision cruzada de PRs antes de mergear a main", ["Todos", "Esencial"], "", []),
]


def llamar(metodo, ruta, **params):
    params["key"] = os.environ["TRELLO_KEY"]
    params["token"] = os.environ["TRELLO_TOKEN"]
    url = f"{API}{ruta}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, method=metodo)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def vista_previa():
    print(f"Tablero: {NOMBRE_TABLERO}")
    print(f"Listas: {', '.join(LISTAS)}")
    print(f"Etiquetas: {', '.join(f'{k} ({v})' for k, v in ETIQUETAS.items())}")
    for lista in LISTAS:
        tarjetas = [t for t in TARJETAS if t[0] == lista]
        if not tarjetas:
            continue
        print(f"\n[{lista}] {len(tarjetas)} tarjetas")
        for _, titulo, etiquetas, _, checklist in tarjetas:
            extra = f" ({len(checklist)} items)" if checklist else ""
            print(f"  - {titulo}  <{', '.join(etiquetas)}>{extra}")
    print(f"\nTotal: {len(TARJETAS)} tarjetas. Ejecuta con --crear para crearlo.")


def crear():
    for var in ("TRELLO_KEY", "TRELLO_TOKEN"):
        if not os.environ.get(var):
            sys.exit(f"Falta la variable de entorno {var}.")

    tablero = llamar("POST", "/boards", name=NOMBRE_TABLERO,
                     defaultLists="false", defaultLabels="false")
    id_tablero = tablero["id"]
    print(f"Tablero creado: {tablero['url']}")

    ids_listas = {}
    for pos, nombre in enumerate(LISTAS, start=1):
        ids_listas[nombre] = llamar("POST", "/lists", name=nombre,
                                    idBoard=id_tablero, pos=pos * 1000)["id"]

    ids_etiquetas = {}
    for nombre, color in ETIQUETAS.items():
        ids_etiquetas[nombre] = llamar("POST", "/labels", name=nombre,
                                       color=color, idBoard=id_tablero)["id"]

    for lista, titulo, etiquetas, desc, checklist in TARJETAS:
        tarjeta = llamar("POST", "/cards", idList=ids_listas[lista], name=titulo,
                         desc=desc, pos="bottom",
                         idLabels=",".join(ids_etiquetas[e] for e in etiquetas))
        if checklist:
            id_check = llamar("POST", "/checklists", idCard=tarjeta["id"],
                              name="Checklist")["id"]
            for item in checklist:
                llamar("POST", f"/checklists/{id_check}/checkItems", name=item)
        print(f"  + {titulo}")

    print(f"\nListo. {len(TARJETAS)} tarjetas. Abrir: {tablero['url']}")


if __name__ == "__main__":
    crear() if "--crear" in sys.argv else vista_previa()
