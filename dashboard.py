"""AA3-EV02 · Tablero de riesgo laboral en trabajadores informales de Bogotá.

Muestra el dataset, la exploración interactiva, la partición 70/15/15, el pipeline de
preprocesamiento, la predicción del modelo y las medidas de seguridad.
Uso: streamlit run dashboard.py
"""
from pathlib import Path
import numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler

st.set_page_config(page_title='Riesgo laboral informal · Bogotá', page_icon='🩺', layout='wide')
RAIZ = Path(__file__).parent
COL = {'Año': 'Anio', 'Localidad de ocurrencia del caso': 'Localidad', 'Régimen aseguramiento en salud': 'Regimen',
       'Ocupación': 'Ocupacion', 'Tipo UTI': 'TipoUTI', 'Clase UTI': 'ClaseUTI', 'Nivel de ingresos': 'Ingresos',
       'Forma de Pago': 'FormaPago', 'Síntoma': 'Sintoma', 'Agente probablemente asociado': 'Agente',
       'Tipo de lesión o Sistema Comprometido': 'Lesion'}
ESC = ["No Fue A La Escuela", "Primaria Incompleta", "Primaria Completa", "Secundaria Incompleta", "Secundaria Completa",
       "Técnico Pos Secundaria Incompleto", "Técnico Pos Secundaria", "Técnico Pos Secundaria Completo",
       "Universidad Incompleta", "Universidad Completa", "Posgrado Incompleto", "Posgrado Completo"]
ING = ["Menos de 1 SMMLV", "1 SMMLV", "Entre 1 y 2 SMMLV", "2 y MÁS SMMLV"]
NOM = ["Sexo", "Localidad", "Regimen", "Ocupacion", "TipoUTI", "ClaseUTI", "FormaPago", "Agente"]
FEAT = NOM + ['Escolaridad', 'Ingresos', 'Edad', 'Anio']
COLORES = {'Entrenamiento': '#2a78d6', 'Validación': '#eb6834', 'Prueba': '#1baf7a'}


@st.cache_data
def cargar():
    crudo = pd.read_csv(RAIZ / 'obs_salud1.csv', sep=';', encoding='latin-1')
    crudo.columns = [c.strip() for c in crudo.columns]
    crudo = crudo.rename(columns=COL)
    d = crudo.copy()
    for c in d.columns:
        if c in ('Anio', 'Edad'): continue
        d[c] = d[c].astype('string').str.strip().str.replace(r'\s+', ' ', regex=True)
        d.loc[d[c].isin(['0', '']), c] = pd.NA
    d.loc[d['Ingresos'] == 'NS / NR', 'Ingresos'] = pd.NA
    n_dup = int(d.duplicated().sum())
    d = d.drop_duplicates().dropna(subset=['Ocupacion', 'Agente', 'Sexo', 'Edad', 'Lesion']).reset_index(drop=True)
    vc = d['Lesion'].value_counts()
    d['Lesion_Modelo'] = d['Lesion'].where(~d['Lesion'].isin(vc[vc < 30].index), 'Otras lesiones / registros múltiples')
    for c in d.columns:
        if c not in ('Anio', 'Edad'): d[c] = d[c].astype(object).where(d[c].notna(), np.nan)
    return crudo, d, n_dup


@st.cache_resource
def entrenar(d):
    y = d['Lesion_Modelo']
    tr, tmp = train_test_split(d.index, test_size=0.30, random_state=42, stratify=y)
    va, te = train_test_split(tmp, test_size=0.50, random_state=42, stratify=y[tmp])
    pre = ColumnTransformer([
        ('onehot', Pipeline([('imputar', SimpleImputer(strategy='most_frequent')),
                             ('codificar', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10, sparse_output=False))]), NOM),
        ('ordinal', Pipeline([('imputar', SimpleImputer(strategy='most_frequent')), ('codificar', OrdinalEncoder(categories=[ESC, ING])),
                              ('escalar', MinMaxScaler())]), ['Escolaridad', 'Ingresos']),
        ('zscore', StandardScaler(), ['Edad']), ('minmax', MinMaxScaler(), ['Anio'])])
    pipe = Pipeline([('preprocesamiento', pre), ('seleccion', VarianceThreshold(0.01)),
                     ('modelo', LogisticRegression(max_iter=3000, C=0.3))]).fit(d.loc[tr, FEAT], y[tr])
    met = {}
    for nombre, idx in [('Entrenamiento', tr), ('Validación', va), ('Prueba', te)]:
        P = pipe.predict_proba(d.loc[idx, FEAT])
        met[nombre] = (accuracy_score(y[idx], pipe.classes_[P.argmax(1)]), top_k_accuracy_score(y[idx], P, k=3, labels=pipe.classes_))
    n_pre = pipe.named_steps['preprocesamiento'].transform(d.loc[tr[:1], FEAT]).shape[1]
    n_sel = int(pipe.named_steps['seleccion'].get_support().sum())
    return pipe, {'Entrenamiento': tr, 'Validación': va, 'Prueba': te}, met, n_pre, n_sel


crudo, df, n_dup = cargar()
with st.spinner('Entrenando el pipeline (solo la primera vez)…'):
    pipe, partes, met, n_pre, n_sel = entrenar(df)

st.title('🩺 Riesgo laboral en trabajadores informales de Bogotá')
st.caption('Datos: Secretaría Distrital de Salud · SIVISTRA · Datos Abiertos Bogotá. '
           'Proyecto SENA «Transformación de datos en modelos de IA» · Viviana Patiño Ballén')

k1, k2, k3, k4 = st.columns(4)
k1.metric('Registros originales', f'{len(crudo):,}'.replace(',', '.'))
k2.metric('Registros tras limpieza', f'{len(df):,}'.replace(',', '.'), f'{len(crudo) - len(df)} eliminados', delta_color='off')
k3.metric('Variables', f'{crudo.shape[1]}', '2 numéricas · 12 categóricas', delta_color='off')
k4.metric('Acierto en prueba (top 3)', f'{met["Prueba"][1]:.1%}'.replace('.', ','))

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['📋 Dataset', '📊 Exploración', '✂️ Partición', '⚙️ Pipeline', '🔮 Predicción', '🔒 Seguridad'])

# ---------------- 1. Dataset ----------------
with tab1:
    st.subheader('Muestra del dataset original')
    st.dataframe(crudo.head(20), width='stretch', height=300)
    st.subheader('Clasificación de las variables')
    clas = pd.DataFrame([
        ('Año', 'Numérica', 'Discreta', 'Contexto temporal'), ('Edad', 'Numérica', 'Discreta', 'Predictor'),
        ('Sexo', 'Categórica', 'Nominal', 'Predictor'), ('Localidad', 'Categórica', 'Nominal', 'Contexto'),
        ('Régimen de salud', 'Categórica', 'Nominal', 'Contexto'), ('Escolaridad', 'Categórica', 'Ordinal', 'Contexto'),
        ('Ocupación', 'Categórica', 'Nominal', 'Predictor'), ('Tipo UTI', 'Categórica', 'Nominal', 'Contexto'),
        ('Clase UTI', 'Categórica', 'Nominal', 'Contexto'), ('Nivel de ingresos', 'Categórica', 'Ordinal', 'Contexto'),
        ('Forma de pago', 'Categórica', 'Nominal', 'Contexto'), ('Síntoma', 'Categórica', 'Nominal', 'Excluida (fuga de información)'),
        ('Agente de riesgo', 'Categórica', 'Nominal', 'Predictor'), ('Tipo de lesión', 'Categórica', 'Nominal', 'Variable objetivo')],
        columns=['Variable', 'Tipo', 'Subtipo', 'Rol'])
    c1, c2 = st.columns([2, 1])
    c1.dataframe(clas, width='stretch', hide_index=True, height=530)
    c2.plotly_chart(px.pie(clas, names='Tipo', hole=.45, color='Tipo', color_discrete_map={'Numérica': '#2a78d6', 'Categórica': '#eb6834'},
                           title='Tipos de variables'), width='stretch')

# ---------------- 2. Exploración interactiva ----------------
with tab2:
    f1, f2, f3 = st.columns(3)
    anios = f1.slider('Años', int(df.Anio.min()), int(df.Anio.max()), (int(df.Anio.min()), int(df.Anio.max())))
    sexos = f2.multiselect('Sexo', sorted(df.Sexo.dropna().unique()), default=['Femenino', 'Masculino'])
    locs = f3.multiselect('Localidad (vacío = todas)', sorted(df.Localidad.dropna().unique()))
    v = df[df.Anio.between(*anios) & df.Sexo.isin(sexos)]
    if locs: v = v[v.Localidad.isin(locs)]
    st.caption(f'{len(v):,} registros con los filtros actuales'.replace(',', '.'))
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.histogram(v, x='Edad', nbins=40, color_discrete_sequence=['#2a78d6'], title='Distribución de la edad'), width='stretch')
    c2.plotly_chart(px.box(v, x='Sexo', y='Edad', color='Sexo', title='Edad por sexo (los puntos son valores atípicos)',
                           color_discrete_map={'Femenino': '#eb6834', 'Masculino': '#2a78d6', 'Indeterminado': '#9aa3a0'}), width='stretch')
    top_les = v.Lesion_Modelo.value_counts().head(10).sort_values()
    c1.plotly_chart(px.bar(x=top_les.values, y=top_les.index, orientation='h', title='10 lesiones más frecuentes',
                           labels={'x': 'Registros', 'y': ''}, color_discrete_sequence=['#2a78d6']), width='stretch')
    top_ag = v.Agente.value_counts().head(10).sort_values()
    c2.plotly_chart(px.bar(x=top_ag.values, y=top_ag.index, orientation='h', title='10 agentes de riesgo más frecuentes',
                           labels={'x': 'Registros', 'y': ''}, color_discrete_sequence=['#1baf7a']), width='stretch')
    num = pd.DataFrame({'Edad': v.Edad, 'Año': v.Anio, 'Escolaridad': pd.Series(pd.Categorical(v.Escolaridad, categories=ESC).codes, index=v.index).replace(-1, np.nan),
                        'Ingresos': pd.Series(pd.Categorical(v.Ingresos, categories=ING).codes, index=v.index).replace(-1, np.nan)})
    c1.plotly_chart(px.imshow(num.corr().round(2), text_auto=True, zmin=-1, zmax=1, color_continuous_scale='RdBu_r',
                              title='Mapa de calor de correlación'), width='stretch')
    muestra = v.sample(min(3000, len(v)), random_state=0).assign(Esc=lambda x: pd.Categorical(x.Escolaridad, categories=ESC).codes)
    c2.plotly_chart(px.scatter(muestra, x='Edad', y='Esc', color='Sexo', opacity=.4, title='Dispersión: edad vs. escolaridad (0 = ninguna, 11 = posgrado)',
                               labels={'Esc': 'Escolaridad'}, color_discrete_map={'Femenino': '#eb6834', 'Masculino': '#2a78d6', 'Indeterminado': '#9aa3a0'}),
                    width='stretch')

# ---------------- 3. Partición ----------------
with tab3:
    st.subheader('Partición 70 / 15 / 15 estratificada por tipo de lesión')
    st.markdown('Los duplicados se eliminan **antes** de partir, para que un mismo caso no quede en entrenamiento y prueba a la vez. '
                'La estratificación mantiene la misma proporción de cada lesión en los tres subconjuntos (semilla fija = 42).')
    tam = pd.DataFrame({'Subconjunto': list(partes), 'Registros': [len(i) for i in partes.values()]})
    tam['%'] = (100 * tam.Registros / tam.Registros.sum()).round(1)
    c1, c2 = st.columns([1, 2])
    c1.plotly_chart(px.pie(tam, names='Subconjunto', values='Registros', hole=.45, color='Subconjunto', color_discrete_map=COLORES,
                           title='Tamaño de cada subconjunto'), width='stretch')
    dist = pd.concat([df.loc[i, 'Lesion_Modelo'].value_counts(normalize=True).rename(k) for k, i in partes.items()], axis=1)
    dist = dist.loc[df.Lesion_Modelo.value_counts().index[:10]].mul(100).round(2).reset_index(names='Lesión').melt(id_vars='Lesión', var_name='Subconjunto', value_name='%')
    c2.plotly_chart(px.bar(dist, x='Lesión', y='%', color='Subconjunto', barmode='group', color_discrete_map=COLORES,
                           title='Las 10 lesiones más frecuentes tienen la misma proporción en cada subconjunto'), width='stretch')
    st.dataframe(tam, hide_index=True)

# ---------------- 4. Pipeline ----------------
with tab4:
    st.subheader('Etapas del pipeline de preprocesamiento')
    etapas = [('1. Limpieza', f'Normaliza texto, convierte «0» y «NS/NR» en faltantes, elimina {n_dup} duplicados y agrupa lesiones con menos de 30 casos'),
              ('2. Partición', '70/15/15 estratificada; todo lo siguiente se ajusta solo con entrenamiento'),
              ('3. Imputar y codificar', 'Moda para faltantes; One-Hot para 8 nominales; codificación ordinal para escolaridad e ingresos'),
              ('4. Escalar', 'Z-score para la edad; Min-Max para el año y los códigos ordinales'),
              ('5. Seleccionar', f'VarianceThreshold elimina columnas casi vacías: {n_pre} → {n_sel}'),
              ('6. Modelo', 'Regresión logística multinomial con regularización L2')]
    cols = st.columns(6)
    for c, (t, dsc) in zip(cols, etapas):
        c.markdown(f'**{t}**'); c.caption(dsc)
    fig = go.Figure(go.Funnel(y=['Columnas originales', 'Entradas del modelo', 'Tras codificación y escalado', 'Tras selección'],
                              x=[14, len(FEAT), n_pre, n_sel], marker_color=['#9aa3a0', '#9aa3a0', '#2a78d6', '#1F5F4F']))
    fig.update_layout(title='Número de columnas en cada etapa')
    c1, c2 = st.columns(2)
    c1.plotly_chart(fig, width='stretch')
    m = pd.DataFrame([(k, v[0] * 100, v[1] * 100) for k, v in met.items()], columns=['Subconjunto', 'Acierto 1.ª opción (%)', 'Acierto top 3 (%)'])
    c2.plotly_chart(px.bar(m.melt(id_vars='Subconjunto', var_name='Métrica', value_name='%'), x='Subconjunto', y='%', color='Métrica', barmode='group',
                           text_auto='.1f', title='Desempeño: estable en entrenamiento, validación y prueba',
                           color_discrete_sequence=['#2a78d6', '#1F5F4F']), width='stretch')

# ---------------- 5. Predicción ----------------
with tab5:
    st.subheader('¿Qué lesión es más probable para este perfil?')
    c1, c2 = st.columns([1, 1.3])
    with c1:
        edad = st.slider('Edad', 15, 90, 40)
        sexo = st.selectbox('Sexo', ['Femenino', 'Masculino'])
        ocup = st.selectbox('Ocupación', df.Ocupacion.value_counts().index.tolist())
        agente = st.selectbox('Agente de riesgo', df[df.Ocupacion == ocup].Agente.value_counts().index.tolist() + [a for a in df.Agente.value_counts().index if a not in set(df[df.Ocupacion == ocup].Agente)])
        loc = st.selectbox('Localidad', sorted(df.Localidad.dropna().unique()))
    moda = lambda c: df[c].mode().iloc[0]
    fila = pd.DataFrame([{'Sexo': sexo, 'Localidad': loc, 'Regimen': moda('Regimen'), 'Ocupacion': ocup, 'TipoUTI': moda('TipoUTI'),
                          'ClaseUTI': moda('ClaseUTI'), 'FormaPago': moda('FormaPago'), 'Agente': agente,
                          'Escolaridad': moda('Escolaridad'), 'Ingresos': moda('Ingresos'), 'Edad': edad, 'Anio': int(df.Anio.max())}])
    p = pd.Series(pipe.predict_proba(fila[FEAT])[0], index=pipe.classes_).sort_values(ascending=False).head(3)
    with c2:
        st.plotly_chart(px.bar(x=p.values * 100, y=p.index, orientation='h', text=[f'{v:.1%}' for v in p.values],
                               labels={'x': 'Probabilidad (%)', 'y': ''}, title='Top 3 de lesiones probables',
                               color_discrete_sequence=['#eb6834']).update_yaxes(autorange='reversed'), width='stretch')
        st.caption('Las variables no preguntadas (régimen, escolaridad, ingresos, tipo de UTI, forma de pago) se completan con el valor más frecuente. '
                   'El resultado apoya la prevención y no reemplaza una valoración clínica.')

# ---------------- 6. Seguridad ----------------
with tab6:
    st.subheader('Privacidad: riesgo de reidentificación (k-anonimato)')
    st.markdown('El dataset no tiene nombres ni documentos, pero **edad, sexo, localidad, ocupación y año** combinados pueden señalar a una persona. '
                'k = número de registros que comparten la misma combinación; si k = 1, el registro es único.')
    QI = ['Edad', 'Sexo', 'Localidad', 'Ocupacion', 'Anio']
    def pct_unicos(d):
        k = d.groupby(QI, dropna=False)[QI[0]].transform('size'); return 100 * (k == 1).mean(), 100 * (k < 5).mean()
    a = df.copy(); a['Edad'] = a.Edad // 10 * 10
    a['Ocupacion'] = a.Ocupacion.where(a.Ocupacion.map(a.Ocupacion.value_counts()) >= 100, 'Otra')
    a['Anio'] = np.where(a.Anio <= 2020, '2017–2020', '2021–2025'); a['Localidad'] = 'Bogotá'
    k_a = a.groupby(QI)[QI[0]].transform('size'); a_final = a[k_a >= 5]
    niveles = pd.DataFrame([('Original', *pct_unicos(df)), ('Generalizado', *pct_unicos(a)), ('Generalizado + supresión', *pct_unicos(a_final))],
                           columns=['Nivel', 'Registros únicos (%)', 'Registros con k < 5 (%)'])
    c1, c2 = st.columns([1.3, 1])
    c1.plotly_chart(px.bar(niveles.melt(id_vars='Nivel', var_name='Indicador', value_name='%'), x='Nivel', y='%', color='Indicador', barmode='group',
                           text_auto='.1f', color_discrete_sequence=['#eb6834', '#9aa3a0'], title='Riesgo antes y después de anonimizar'), width='stretch')
    c2.markdown(f"""**Medidas aplicadas**
- Anonimización: edad en grupos de 10 años, solo ocupaciones con ≥ 100 casos, año en dos periodos, sin localidad y supresión de grupos con k < 5 (se conserva el {100 * len(a_final) / len(df):.1f} % de los registros).
- Sin identificadores directos; la variable Síntoma no se usa.
- Integridad: huella SHA-256 del archivo original; se trabaja sobre copias.
- Este tablero no guarda los datos que se ingresan en la pestaña de predicción.

**Propuestas:** control de acceso por roles, cifrado en tránsito y en reposo, y autorización previa (Ley 1581 de 2012) si se usan datos propios de trabajadores.""")
    st.caption('Vista del archivo anonimizado (seguro para compartir):')
    st.dataframe(a_final[['Edad', 'Sexo', 'Ocupacion', 'Anio', 'Agente', 'Lesion']].rename(columns={'Edad': 'Edad (decenio)'}).head(10), hide_index=True, width='stretch')
