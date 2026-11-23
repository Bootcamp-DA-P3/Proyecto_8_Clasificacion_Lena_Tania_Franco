# Insights de Negocio — Inclusión Financiera (Global Findex 2025)

**Proyecto:** Proyecto 8 — Clasificación de inclusión financiera (`account`)  
**Autora:** Lena  
**Fuente analizada:** [notebooks/01_eda.ipynb](../notebooks/01_eda.ipynb) sobre `data/raw/findex_microdata_2025_labelled_update112425.csv` (144,090 personas encuestadas en 140 economías, año 2024).

---

## 1. Resumen ejecutivo

- **Cobertura actual (`account`):** el **73.78%** de las personas encuestadas (`106,312`) declara tener una cuenta financiera o de dinero móvil, mientras que el **26.22%** (`37,778` personas) permanece **no bancarizado** (tasa ponderada por `wgt`: **71.05%** con cuenta).
- **Desbalance moderado (~2.81 : 1):** desde el punto de vista de negocio, el foco de las campañas de inclusión está en ese **26.22%** excluido (`account = 0`), por lo que los modelos predictivos deben optimizar **F1, Recall y ROC-AUC** y no solo accuracy.
- **Hallazgo estratégico clave:** la falta de cuenta no equivale a desconexión tecnológica. El **75.5%** de las personas sin cuenta (`28,532` de `37,778`) **ya posee un teléfono móvil** y casi la mitad (**48.6%**, `18,341` personas) cuenta con **móvil e internet**.

---

## 2. Hallazgos principales sobre los perfiles no bancarizados

### 2.1 La educación es el predictor sociodemográfico más fuerte
- **Correlación más alta del perfil limpio:** `educ` (`r_Spearman = +0.337`).
- Personas con **primaria o menos (`educ = 1`)** sufren una exclusión del **48.5%** (`18,626` personas) y concentran casi la **mitad (49.3%)** de todo el universo no bancarizado.
- Con secundaria (`educ = 2`) la exclusión cae al **22.8%** (`17,020`, `45.1%` del total), y con educación terciaria (`educ = 3`) baja al **6.3%** (`1,918`, `5.1%` del total): una brecha de **42.2 puntos porcentuales**.
- **Efecto dominante sobre el ingreso:** una persona con educación primaria en el quintil más rico (`Q5`) mantiene un **42.3%** de exclusión, casi **4 veces más** que una persona con educación terciaria en el quintil más pobre (`Q1`, **10.9%**).

### 2.2 Conectividad digital: móvil e internet marcan la frontera
- Quienes **no usan internet** (`internet_use = 0`) tienen un **50.7%** de exclusión (`18,028`, `47.7%` del total no bancarizado) frente al **18.2%** entre usuarios de internet.
- Quienes **no tienen teléfono móvil propio** (`con1 = 2`) alcanzan la mayor tasa de riesgo individual del perfil (**61.9%** sin cuenta, aunque representan el **24.4%** del volumen total por su menor tamaño muestral).

### 2.3 Concentración geográfica y brecha regional
- El **91.8%** de todas las personas sin cuenta (`34,663` de `37,778`) vive fuera de las economías de ingresos altos (`High income`, donde la exclusión es solo del **6.7%**).
- **Regiones con mayor exclusión:**
  1. **Oriente Medio y Norte de África — MENA (no altos ingresos):** **53.7%** sin cuenta.
  2. **América Latina y el Caribe (no altos ingresos):** **39.9%**.
  3. **África Subsahariana (no altos ingresos):** **37.9%**.
  4. **Asia del Sur:** **30.2%**.
  5. **Asia Oriental y Pacífico (no altos ingresos):** **26.7%**.
  6. **Europa y Asia Central (no altos ingresos):** **24.3%**.
- **Economías más rezagadas en la muestra:** Níger (`79.4%` sin cuenta), Líbano (`75.4%`), Madagascar (`73.3%`), Nicaragua (`72.9%`) y Chad (`72.9%`).

### 2.4 Brecha de género: moderada en el agregado, severa por región
- Globalmente, el **29.8%** de las mujeres no tiene cuenta frente al **22.2%** de los hombres (**7.6 pp** de brecha). Las mujeres representan el **59.7%** (`22,539`) del total de no bancarizados.
- Esta brecha es muy desigual geográficamente: alcanza **18.6 pp en MENA** (`37.3%` de inclusión en mujeres vs `55.8%` en hombres) y ronda los **9.2–9.4 pp** en África Subsahariana (`9.4 pp`), América Latina (`9.3 pp`) y Asia del Sur (`9.2 pp`), mientras casi desaparece en Asia Oriental y Pacífico (`0.7 pp`) y en `High income` (`1.5 pp`).
- La intersección **mujer fuera de la fuerza laboral** reúne a `14,092` personas sin cuenta (**37.3%** del total no bancarizado, con **39.0%** de exclusión).

### 2.5 Juventud, ámbito rural, ingresos y empleo
- **Edad (`15–24` años):** presentan **41.3%** de exclusión (`10,438` jóvenes, **27.6%** de todos los no bancarizados), prácticamente el doble que cualquier tramo a partir de 25 años (`21.4%–24.7%`).
- **Zona rural (`urbanicity = 1`):** tiene **30.5%** de exclusión frente al **21.4%** urbano y concentra el **64.2%** (`24,240`) de las personas sin cuenta. Como excepción regional, en Asia del Sur la exclusión urbana (`34.1%`) supera ligeramente a la rural (`28.1%`).
- **Fuera de la fuerza laboral (`emp_in = 2`):** **35.5%** de exclusión (`19,948`, **52.8%** del total sin cuenta) frente a **20.7%** en personas activas laboralmente.
- **Ingresos (`inc_q` Q1–Q2):** el 40% de menor ingreso muestra **33.9%** de exclusión (`16,750`, **44.3%** del total sin cuenta), desde **37.2%** en `Q1` hasta **17.6%** en `Q5`.
- **Segmento compuesto crítico (`Educación primaria + Rural`):** tasa de exclusión del **51.2%** (`13,604` personas, **36.0%** de todos los no bancarizados).

---

## 3. Palancas accionables para campañas de inclusión financiera

Segmentando las **37,778 personas sin cuenta** según su acceso digital real (`con1` e `internet_use`), se recomiendan tres estrategias diferenciadas:

| Segmento de no bancarizados | Volumen (`n`) | % sobre sin cuenta | Canal y estrategia recomendada |
|---|---:|---:|---|
| **Móvil + Internet (*Digital-ready*)** | 18,341 | **48.6%** | **Onboarding 100% digital:** billeteras móviles, apertura remota con KYC simplificado y foco en **jóvenes (15–24 años)** y entornos urbanos/semiurbanos. |
| **Solo móvil, sin internet** | 10,191 | **27.0%** | **Banca móvil básica (USSD/SMS) y redes de agentes:** ideal para **zonas rurales** y personas trabajadoras sin tarifa de datos o smartphone. |
| **Sin móvil ni internet (Desconectado)** | 7,812 | **20.7%** | **Inclusión presencial y educación financiera:** corresponsales comunitarios, cuentas básicas sin comisiones, apoyo en alfabetización financiera y programas dirigidos a **mujeres fuera del mercado laboral** y población con **primaria o menos**. |

*(El 3.8% restante corresponde a personas con internet sin móvil propio — `1,393`, `3.69%` — o NS/NC en `con1` — `41`, `0.11%`.)*

---

## 4. Handoff técnico para Fase 0, Preprocesado y Modelado

- **Alineación con [`src/preprocess.py`](../src/preprocess.py) (Franco):**
  - El uso de lista blanca (`age`, `educ`, `inc_q`, `female`, `emp_in`, `urbanicity` -> `urban`, `internet_use`, `regionwb`, `economycode`) elimina el **data leakage** demostrado en el EDA (como `account_fin` `r=+0.842`, `anydigpayment` `r=+0.818`, `dig_account` `r=+0.724`, `fin2` `r=-0.631` y las **51 columnas `fin*`** con nulos estructuralmente ligados al target).
  - La recodificación `1/2 -> 1/0` y los indicadores de ausencia (`add_indicator=True`) en binarias capturan correctamente que `emp_in` nulo corresponde al 100% a China (`4,020` filas).
- **Puntos de vigilancia para Modelado (Tania) e iteraciones:**
  1. **`con1` (tenencia de móvil):** tiene `r_Spearman = +0.276` con `account` y 0% nulos (salvo 102 filas con códigos `3/4` NS/NC). Merece evaluarse en una iteración v2 como feature adicional sin leakage.
  2. **`inc_q` en Lesotho:** los `1,020` nulos (`0.71%`) pertenecen íntegramente a Lesotho; la imputación por mediana global asigna quintil `3`.
  3. **`age == 100`:** existen `279` registros con edad 100 (posible *top-coding* en países de ingresos altos).
  4. **Overfitting por `economycode`:** introduce ~140 columnas one-hot; verificar que la brecha de F1 entre train y test se mantenga **< 5%** frente a modelos que solo usen `regionwb`.
