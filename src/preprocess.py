"""Preprocesado y split train/test del proyecto de clasificacion Findex 2025.

Objetivo: predecir `account` (1 = tiene cuenta financiera) a partir de
caracteristicas demograficas y de contexto.

Nota sobre fuga de datos (LEAKAGE)
----------------------------------
Se usa una LISTA BLANCA de columnas: solo las variables definidas en
NUM_FEATURES, BIN_FEATURES y CAT_FEATURES entran al modelo. Todo lo demas se
descarta. Esto excluye, entre otras, las columnas fin*, con*, account_fin,
account_mob, dig_account, borrowed, saved, receive_*, pay_*, anydigpayment,
merchantpay_dig y domestic_remittances, porque estan derivadas de tener (o
usar) una cuenta, y por tanto revelan el target. Tambien se excluyen los
identificadores y pesos (wpid_random, wgt, year, pop_adult, nombre de
economia).

Salidas
-------
- data/processed/train_raw.csv y test_raw.csv: datos limpios SIN codificar,
  con el target, para que cada integrante arme sus propios Pipelines.
- data/processed/train.csv y test.csv: features codificadas + `account` al final.
- models/preprocessor.joblib: ColumnTransformer ajustado solo con train.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "account"

# Variables numericas / ordinales (se imputan con la mediana y se escalan).
NUM_FEATURES = ["age", "educ", "inc_q"]
# Variables binarias ya recodificadas a 0/1 (imputacion por moda + indicador).
BIN_FEATURES = ["female", "emp_in", "urban", "internet_use"]
# Variables categoricas nominales (one-hot).
CAT_FEATURES = ["regionwb", "economycode"]

# Nombres de columnas tal como vienen en el CSV original (urbanicity se
# renombra a urban en `limpiar`).
RAW_FEATURES = [
    "age",
    "female",
    "educ",
    "inc_q",
    "emp_in",
    "urbanicity",
    "internet_use",
    "regionwb",
    "economycode",
]

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "findex_microdata_2025_labelled_update112425.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"


def cargar_datos(path: Path = RAW_PATH) -> pd.DataFrame:
    """Lee el CSV original cargando solo el target y las columnas de la lista blanca."""
    return pd.read_csv(path, usecols=[TARGET] + RAW_FEATURES, low_memory=False)


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica los recodificados a los codigos originales del CSV.

    - female: {1: 1, 2: 0} (1 = mujer).
    - emp_in: {1: 1, 2: 0} (1 = en la fuerza laboral), conserva NaN.
    - urbanicity: {1: 0, 2: 1} y se renombra a `urban` (1 = urbano), conserva NaN.

    Funciona sin la columna `account` (por ejemplo, con la fila de un formulario)
    y solo transforma las columnas presentes. Debe aplicarse una sola vez sobre
    codigos originales: aplicarla a datos ya limpios invertiria las variables.
    """
    df = df.copy()
    if "female" in df.columns:
        df["female"] = df["female"].map({1: 1, 2: 0})
    if "emp_in" in df.columns:
        df["emp_in"] = df["emp_in"].map({1: 1, 2: 0})
    if "urbanicity" in df.columns:
        df["urbanicity"] = df["urbanicity"].map({1: 0, 2: 1})
        df = df.rename(columns={"urbanicity": "urban"})
    return df


def construir_preprocesador() -> ColumnTransformer:
    """Devuelve el ColumnTransformer sin ajustar.

    - Numericas: mediana + StandardScaler.
    - Binarias: moda + indicador de faltante (para emp_in, faltante = China).
    - Categoricas: OneHotEncoder que ignora categorias desconocidas.
    """
    numericas = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    binarias = SimpleImputer(strategy="most_frequent", add_indicator=True)
    categoricas = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return ColumnTransformer(
        transformers=[
            ("num", numericas, NUM_FEATURES),
            ("bin", binarias, BIN_FEATURES),
            ("cat", categoricas, CAT_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def main() -> None:
    """Carga, limpia, divide, guarda los datos y ajusta el preprocesador."""
    df = limpiar(cargar_datos())
    X = df[NUM_FEATURES + BIN_FEATURES + CAT_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Datos limpios sin codificar, con el target al final.
    train_raw = X_train.assign(**{TARGET: y_train}).reset_index(drop=True)
    test_raw = X_test.assign(**{TARGET: y_test}).reset_index(drop=True)
    train_raw.to_csv(PROCESSED_DIR / "train_raw.csv", index=False)
    test_raw.to_csv(PROCESSED_DIR / "test_raw.csv", index=False)

    # El preprocesador se ajusta solo con train para evitar fuga de informacion.
    preprocesador = construir_preprocesador()
    Z_train = preprocesador.fit_transform(X_train)
    Z_test = preprocesador.transform(X_test)
    columnas = list(preprocesador.get_feature_names_out())

    train = pd.DataFrame(Z_train, columns=columnas)
    train[TARGET] = y_train.to_numpy()
    test = pd.DataFrame(Z_test, columns=columnas)
    test[TARGET] = y_test.to_numpy()
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)

    joblib.dump(preprocesador, PREPROCESSOR_PATH)

    print(f"Filas totales: {len(df)}")
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"Proporcion de {TARGET}=1 en train: {y_train.mean():.4f}")
    print(f"Proporcion de {TARGET}=1 en test:  {y_test.mean():.4f}")
    print(f"Numero de features de salida: {len(columnas)}")
    print("Archivos generados:")
    for ruta in [
        PROCESSED_DIR / "train_raw.csv",
        PROCESSED_DIR / "test_raw.csv",
        PROCESSED_DIR / "train.csv",
        PROCESSED_DIR / "test.csv",
        PREPROCESSOR_PATH,
    ]:
        print(f"  {ruta.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
