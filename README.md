# Sistema de predicción de riesgos con IA

Aplicación y análisis de datos que estiman qué tipo de lesión es más probable en un trabajador informal de Bogotá según su edad, sexo, ocupación y agente de riesgo.

Proyecto desarrollado por **Viviana Patiño Ballén** en el programa del SENA *Transformación de datos en modelos de inteligencia artificial* (código 21710119).

## Datos

- **Dataset:** «Enfermedades derivadas de la ocupación en Unidades de Trabajo Informal (UTI) en Bogotá D.C.», Secretaría Distrital de Salud (SIVISTRA).
- **Fuente:** [Datos Abiertos Bogotá](https://datosabiertos.bogota.gov.co/dataset/enfermedades-derivadas-de-la-ocupacion-en-unidades-de-trabajo-informal-uti-en-bogota-d-c)
- **Archivo:** `obs_salud1.csv` (22.438 registros × 14 variables, separador `;`, codificación Latin-1).

## Estructura

| Archivo | Qué hace |
|---|---|
| `dashboard.py` | AA3-EV02: tablero interactivo (dataset, exploración, partición, pipeline, predicción y seguridad) |
| `app.py` | Aplicación web (Streamlit) que muestra las 3 lesiones más probables |
| `modelo.py` | Carga, limpieza y pipeline del modelo que usa la aplicación |
| `preparacion_datos.py` | AA2-EV01: depuración, codificación y escalado |
| `aa2_ev02_reduccion.py` | AA2-EV02: reducción de dimensionalidad (PCA y selección de atributos) |
| `aa3_ev01_visualizacion.py` | AA3-EV01: exploración visual, partición 70/15/15, pipeline y anonimización |
| `informes/` | Informes en PDF de cada evidencia |

## Cómo ejecutarlo

```bash
pip install -r requirements.txt

streamlit run dashboard.py           # tablero AA3-EV02
streamlit run app.py                 # aplicación web
python preparacion_datos.py          # AA2-EV01
python aa2_ev02_reduccion.py         # AA2-EV02
python aa3_ev01_visualizacion.py     # AA3-EV01
```

Cada script genera sus figuras y un archivo `resultados_*.json` con todas las cifras que citan los informes.

## Privacidad

El dataset no tiene identificadores directos, pero la combinación de edad, sexo, localidad, ocupación y año hace único al 71,7 % de los registros. `aa3_ev01_visualizacion.py` genera `obs_salud_anonimizado.csv` (k ≥ 5), la versión recomendada para compartir fuera del proyecto. Los resultados del modelo apoyan la prevención y no reemplazan una valoración clínica.
