# Cómo trabajamos en el repo

Reglas del equipo (Lena, Tania, Franco) para ramas, commits y PRs.

## Ramas

| Rama | Uso |
|---|---|
| `main` | Versión estable y entregable. Solo recibe merges desde `dev`. |
| `dev` | Integración. Aquí se mergean las features. |
| `feature/<tarea>` | Una rama por tarjeta de Trello. Sale de `dev` y vuelve a `dev`. |

Ejemplos: `feature/eda`, `feature/preprocess`, `feature/modelo-baseline`, `feature/app-streamlit`.

Nunca se hace commit directo a `main` ni a `dev`.

## Flujo de una tarea

```bash
git checkout dev
git pull
git checkout -b feature/eda
# ... trabajar y hacer commits ...
git push -u origin feature/eda
```

1. Mover la tarjeta a **En curso**.
2. Crear la rama desde `dev` actualizado.
3. Hacer commits pequeños.
4. Abrir un PR hacia `dev` y pegar el link del PR en la tarjeta.
5. Mover la tarjeta a **En revisión (PR)**.
6. Otra persona del equipo revisa y aprueba. Quien abrió el PR hace el merge.
7. Mover la tarjeta a **Hecho** y borrar la rama.

Al cerrar cada fase se abre un PR de `dev` a `main`.

## Mensajes de commit

Formato: `tipo: descripción corta en presente`

| Tipo | Para qué |
|---|---|
| `feat` | Funcionalidad nueva (app, pipeline, script) |
| `fix` | Corrección de un error |
| `eda` | Análisis exploratorio y gráficos |
| `model` | Entrenamiento, tuning y evaluación de modelos |
| `docs` | README, informe, documentación |
| `refactor` | Reorganizar código sin cambiar el resultado |
| `test` | Tests |
| `chore` | Configuración, dependencias, estructura |

Ejemplos:

```
feat: añade formulario de perfil en la app
fix: corrige recodificación de female
eda: añade cruce del target con ingreso
model: compara random forest y xgboost
docs: documenta columnas excluidas por leakage
```

Reglas:

- Un commit por cambio lógico. Nada de "cambios varios" o "update".
- Máximo unos 50 caracteres en la primera línea.
- Si hace falta explicar el porqué, se deja una línea en blanco y se escribe debajo.

## Qué no se sube

- Tokens, contraseñas ni archivos `.env`.
- Datos procesados (`data/processed/`) ni modelos pesados sin acordarlo antes.
- Outputs pesados de los notebooks: limpiar las salidas grandes antes del commit.

## Datos

El CSV de Findex 2025 (49 MB) está versionado en `data/raw/`. Está por debajo del límite de 100 MB de GitHub, así que no hace falta Git LFS. Los archivos que genere `src/preprocess.py` van a `data/processed/` y no se suben: cada uno los regenera ejecutando el script.
