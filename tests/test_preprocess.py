"""Tests unitarios de src/preprocess.py (datos sinteticos, sin leer el CSV)."""

import numpy as np
import pandas as pd
import pytest

from src import preprocess as pp


@pytest.fixture
def df_crudo():
    return pd.DataFrame(
        {
            "age": [25, 40, np.nan, 60],
            "female": [1, 2, 2, 1],
            "educ": [1, 2, 3, np.nan],
            "inc_q": [1, 2, 3, 4],
            "emp_in": [1, 2, np.nan, 1],
            "urbanicity": [1, 2, 2, np.nan],
            "internet_use": [0, 1, 1, 0],
            "regionwb": ["LAC", "MENA", "SSA", "LAC"],
            "economycode": ["CHL", "EGY", "NER", "CHL"],
            "account": [1, 0, 0, 1],
        }
    )


def test_limpiar_recodifica_female(df_crudo):
    out = pp.limpiar(df_crudo)
    assert out["female"].tolist() == [1, 0, 0, 1]  # 1 = mujer


def test_limpiar_recodifica_emp_in_y_conserva_nan(df_crudo):
    out = pp.limpiar(df_crudo)
    assert out["emp_in"].iloc[:2].tolist() == [1, 0]
    assert np.isnan(out["emp_in"].iloc[2])


def test_limpiar_renombra_urbanicity_a_urban(df_crudo):
    out = pp.limpiar(df_crudo)
    assert "urban" in out.columns and "urbanicity" not in out.columns
    assert out["urban"].iloc[:3].tolist() == [0, 1, 1]  # 1 = urbano
    assert np.isnan(out["urban"].iloc[3])


def test_limpiar_no_modifica_el_original(df_crudo):
    copia = df_crudo.copy()
    pp.limpiar(df_crudo)
    pd.testing.assert_frame_equal(df_crudo, copia)


def test_limpiar_funciona_sin_target_ni_otras_columnas():
    fila = pd.DataFrame({"female": [2], "urbanicity": [2]})
    out = pp.limpiar(fila)
    assert out.loc[0, "female"] == 0 and out.loc[0, "urban"] == 1


def test_lista_blanca_no_incluye_leakage():
    usadas = set(pp.NUM_FEATURES + pp.BIN_FEATURES + pp.CAT_FEATURES)
    prohibidas = {"account", "account_fin", "account_mob", "dig_account", "borrowed", "saved"}
    assert not usadas & prohibidas
    assert not any(c.startswith(("fin", "con")) for c in usadas)


def test_preprocesador_salida_sin_nulos_y_con_nombres(df_crudo):
    df = pp.limpiar(df_crudo)
    X = df[pp.NUM_FEATURES + pp.BIN_FEATURES + pp.CAT_FEATURES]
    pre = pp.construir_preprocesador()
    Z = pre.fit_transform(X)
    assert not np.isnan(Z).any()
    assert Z.shape[0] == len(X)
    assert Z.shape[1] == len(pre.get_feature_names_out())


def test_preprocesador_ignora_categorias_desconocidas(df_crudo):
    df = pp.limpiar(df_crudo)
    cols = pp.NUM_FEATURES + pp.BIN_FEATURES + pp.CAT_FEATURES
    pre = pp.construir_preprocesador().fit(df[cols])
    nuevo = df[cols].iloc[[0]].copy()
    nuevo["economycode"] = "XXX"
    Z = pre.transform(nuevo)
    assert Z.shape[0] == 1


def test_constantes_del_proyecto():
    assert pp.RANDOM_STATE == 42
    assert pp.TEST_SIZE == 0.2
    assert pp.TARGET == "account"
