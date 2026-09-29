"""AA3-EV01 · Visualización, partición y pipeline de preprocesamiento.

Entrada : obs_salud1.csv (Datos Abiertos Bogotá · SIVISTRA)
Salidas : figuras_aa3_ev01/*.png, resultados_aa3_ev01.json, obs_salud_anonimizado.csv
Uso     : python aa3_ev01_visualizacion.py
"""
import hashlib, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler
warnings.filterwarnings('ignore')

RAIZ = Path(__file__).parent; FIG = RAIZ / 'figuras_aa3_ev01'; FIG.mkdir(exist_ok=True)
ARCHIVO = RAIZ / 'obs_salud1.csv'
COL = {'Año': 'Anio', 'Localidad de ocurrencia del caso': 'Localidad', 'Régimen aseguramiento en salud': 'Regimen',
       'Ocupación': 'Ocupacion', 'Tipo UTI': 'TipoUTI', 'Clase UTI': 'ClaseUTI', 'Nivel de ingresos': 'Ingresos',
       'Forma de Pago': 'FormaPago', 'Síntoma': 'Sintoma', 'Agente probablemente asociado': 'Agente',
       'Tipo de lesión o Sistema Comprometido': 'Lesion'}
ESC = ["No Fue A La Escuela", "Primaria Incompleta", "Primaria Completa", "Secundaria Incompleta", "Secundaria Completa",
       "Técnico Pos Secundaria Incompleto", "Técnico Pos Secundaria", "Técnico Pos Secundaria Completo",
       "Universidad Incompleta", "Universidad Completa", "Posgrado Incompleto", "Posgrado Completo"]
ING = ["Menos de 1 SMMLV", "1 SMMLV", "Entre 1 y 2 SMMLV", "2 y MÁS SMMLV"]
NOM = ["Sexo", "Localidad", "Regimen", "Ocupacion", "TipoUTI", "ClaseUTI", "FormaPago", "Agente"]
R = {}
AZ, NA, VE, GR, TI, T2 = '#2a78d6', '#eb6834', '#1baf7a', '#9aa3a0', '#15201b', '#4b5852'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.edgecolor': '#c9d1cd', 'axes.labelcolor': T2,
    'xtick.color': T2, 'ytick.color': T2, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
    'grid.color': '#e3e8e5', 'axes.axisbelow': True, 'axes.titleweight': 'bold', 'axes.titlesize': 10.5,
    'axes.titlecolor': TI, 'figure.dpi': 150, 'savefig.bbox': 'tight'})

# =============== 1. CARGA ===============
R['sha256_original'] = hashlib.sha256(ARCHIVO.read_bytes()).hexdigest()
crudo = pd.read_csv(ARCHIVO, sep=';', encoding='latin-1')
crudo.columns = [c.strip() for c in crudo.columns]; crudo = crudo.rename(columns=COL)
R['shape_crudo'] = list(crudo.shape)
R['tipos'] = {'numericas': [c for c in crudo.columns if pd.api.types.is_numeric_dtype(crudo[c])],
              'categoricas': [c for c in crudo.columns if not pd.api.types.is_numeric_dtype(crudo[c])]}

# =============== 2. EXPLORACIÓN VISUAL (datos crudos) ===============
txt = [c for c in crudo.columns if c not in ('Anio', 'Edad')]
vacios = {c: int(crudo[c].isna().sum() + crudo[c].astype(str).str.strip().isin(['0', 'NS / NR']).sum()) for c in txt}
vacios = {k: v for k, v in vacios.items() if v}
R['vacios'] = vacios; R['duplicados'] = int(crudo.duplicated().sum())
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.2), gridspec_kw={'width_ratios': [1.2, 1]})
ks = list(vacios); ax = axs[0]
ax.barh(ks, [vacios[k] for k in ks], color=NA, height=.6); ax.invert_yaxis()
for i, k in enumerate(ks): ax.text(vacios[k] + .3, i, str(vacios[k]), va='center', fontsize=8.5, color=TI)
ax.set_xlabel('Registros'); ax.set_title('Vacíos o sin sentido («0», «NS / NR») por variable', loc='left', fontsize=9.5); ax.grid(axis='y', visible=False)
ax = axs[1]
mal = crudo[txt].isna().any(axis=1) | crudo[txt].astype(str).apply(lambda c: c.str.strip().isin(['0', 'NS / NR'])).any(axis=1)
R['filas_con_vacio'] = int(mal.sum())
vals = [len(crudo), R['duplicados'], R['filas_con_vacio']]
ax.bar(['Total de\nregistros', 'Duplicados\nexactos', 'Filas con algún\nvacío o «0»'], vals, color=[AZ, NA, NA], width=.55)
ax.set_yscale('log'); ax.set_title('Escala de los problemas (eje log)', loc='left', fontsize=9.5)
for i, v in enumerate(vals): ax.text(i, v * 1.15, f'{v:,}'.replace(',', '.'), ha='center', fontsize=8.5, color=TI)
fig.suptitle('Figura 1. Valores vacíos y duplicados en los datos crudos', x=.01, ha='left', fontweight='bold', fontsize=10.5)
fig.tight_layout(); fig.savefig(FIG / '01_vacios.png'); plt.close(fig)

# Histograma y boxplot de Edad (outliers) + histograma de Año
e = crudo['Edad']; q1, q3 = e.quantile([.25, .75]); lim_s = q3 + 1.5 * (q3 - q1); lim_i = q1 - 1.5 * (q3 - q1)
R['edad'] = dict(media=round(e.mean(), 2), mediana=float(e.median()), min=int(e.min()), max=int(e.max()), q1=float(q1), q3=float(q3),
                 lim_sup=float(lim_s), atipicos=int((e > lim_s).sum() + (e < lim_i).sum()), menores18=int((e < 18).sum()), asim=round(float(e.skew()), 3))
fig, axs = plt.subplots(1, 3, figsize=(10, 3.1), gridspec_kw={'width_ratios': [1.5, .7, 1.2]})
axs[0].hist(e, bins=40, color=AZ, edgecolor='white', lw=.5); axs[0].axvline(e.mean(), color=NA, lw=1.6)
axs[0].set_title(f'Edad: media {e.mean():.1f}, asimetría {e.skew():.2f}'.replace('.', ','), loc='left', fontsize=9.5); axs[0].set_xlabel('Edad (años)'); axs[0].set_ylabel('Registros')
axs[1].boxplot(e, widths=.5, patch_artist=True, boxprops=dict(facecolor='#cde2fb', edgecolor=AZ), medianprops=dict(color=NA, lw=2),
               flierprops=dict(marker='o', markersize=4, markerfacecolor=NA, markeredgecolor='white'))
axs[1].set_xticks([1], ['Edad']); axs[1].set_title(f'{R["edad"]["atipicos"]} atípicos > {lim_s:.1f}'.replace('.', ','), loc='left', fontsize=9.5)
anio = crudo['Anio'].value_counts().sort_index()
axs[2].bar(anio.index, anio.values, color=AZ, width=.7); axs[2].set_title('Registros por año', loc='left', fontsize=9.5); axs[2].set_xticks(anio.index, [str(a)[2:] for a in anio.index])
R['por_anio'] = {int(k): int(v) for k, v in anio.items()}
fig.suptitle('Figura 2. Histogramas y diagrama de caja de las variables numéricas', x=.01, ha='left', fontweight='bold', fontsize=10.5)
fig.tight_layout(); fig.savefig(FIG / '02_histogramas_boxplot.png'); plt.close(fig)

# Distribución del objetivo (clases desbalanceadas)
lz = crudo['Lesion'].astype(str).str.strip(); vcl = lz.value_counts()
R['lesion_top'] = [[k, int(v)] for k, v in vcl.head(5).items()]; R['lesion_clases_crudo'] = int(vcl.size)
R['lesion_menos30'] = int((vcl < 30).sum())
fig, ax = plt.subplots(figsize=(9, 3))
ax.bar(range(len(vcl)), vcl.values, color=[AZ if v >= 30 else NA for v in vcl.values], width=.85); ax.set_yscale('log')
ax.axhline(30, color=T2, ls='--', lw=1); ax.text(len(vcl) - 1, 36, 'umbral: 30 casos', ha='right', fontsize=8, color=TI)
ax.set_xticks([]); ax.set_xlabel('Tipo de lesión (61 valores, ordenados por frecuencia)'); ax.set_ylabel('Registros (log)')
ax.set_title('Figura 3. Distribución irregular del objetivo: pocas clases concentran los casos (naranja < 30 casos)', loc='left')
fig.savefig(FIG / '03_objetivo.png'); plt.close(fig)

# =============== 3. LIMPIEZA (necesaria antes de la partición) ===============
def limpiar(d):
    d = d.copy()
    for c in d.columns:
        if c in ('Anio', 'Edad'): continue
        d[c] = d[c].astype('string').str.strip().str.replace(r'\s+', ' ', regex=True)
        d.loc[d[c].isin(['0', '']), c] = pd.NA
    d.loc[d['Ingresos'] == 'NS / NR', 'Ingresos'] = pd.NA
    d = d.drop_duplicates()                                       # antes de partir: evita el mismo caso en train y test
    d = d.dropna(subset=['Ocupacion', 'Agente', 'Sexo', 'Edad', 'Lesion']).reset_index(drop=True)
    vc = d['Lesion'].value_counts(); raras = vc[vc < 30].index
    d['Lesion_Modelo'] = d['Lesion'].where(~d['Lesion'].isin(raras), 'Otras lesiones / registros múltiples')
    for c in d.columns:                                           # faltantes restantes: se imputan DENTRO del pipeline
        if c not in ('Anio', 'Edad'): d[c] = d[c].astype(object).where(d[c].notna(), np.nan)
    return d
df = limpiar(crudo); R['shape_limpio'] = list(df.shape); R['faltantes_a_imputar'] = {c: int(v) for c, v in df.isna().sum().items() if v}; R['clases'] = int(df.Lesion_Modelo.nunique())

# Dispersión Edad vs Escolaridad (y año) — relaciones y outliers
esc = pd.Categorical(df['Escolaridad'], categories=ESC).codes
rng = np.random.default_rng(0); idx = rng.choice(len(df), 4000, replace=False)
R['r_edad_esc'] = round(float(np.corrcoef(df.Edad[esc >= 0], esc[esc >= 0])[0, 1]), 3)
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.3))
axs[0].scatter(df.Edad.to_numpy()[idx], esc[idx] + rng.uniform(-.3, .3, len(idx)), s=5, alpha=.35, color=AZ, edgecolors='none')
axs[0].set_xlabel('Edad (años)'); axs[0].set_ylabel('Escolaridad (0 = ninguna … 11 = posgrado)')
axs[0].set_title(f'Edad vs. escolaridad (r = {R["r_edad_esc"]:.2f})'.replace('.', ','), loc='left', fontsize=9.5)
sexo = df.Sexo.to_numpy()[idx]
for s, c in [('Femenino', NA), ('Masculino', AZ)]:
    m = sexo == s; axs[1].scatter(df.Anio.to_numpy()[idx][m] + rng.uniform(-.35, .35, m.sum()), df.Edad.to_numpy()[idx][m], s=5, alpha=.35, color=c, label=s, edgecolors='none')
axs[1].set_xlabel('Año'); axs[1].set_ylabel('Edad (años)'); axs[1].legend(frameon=False, markerscale=3, fontsize=8)
axs[1].set_title('Edad por año y sexo', loc='left', fontsize=9.5)
fig.suptitle('Figura 4. Gráficos de dispersión (muestra aleatoria de 4.000 registros)', x=.01, ha='left', fontweight='bold', fontsize=10.5)
fig.tight_layout(); fig.savefig(FIG / '04_dispersion.png'); plt.close(fig)

# Heatmap: V de Cramér entre variables (categóricas + numéricas discretizadas) y objetivo
def cramers_v(a, b):
    t = pd.crosstab(a, b); chi2 = chi2_contingency(t, correction=False)[0]; n = t.values.sum(); r, k = t.shape
    phi2 = max(0, chi2 / n - (k - 1) * (r - 1) / (n - 1)); r2 = r - (r - 1) ** 2 / (n - 1); k2 = k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(1e-9, min(k2 - 1, r2 - 1))))
H = df.copy(); H['Edad (grupos)'] = pd.cut(H.Edad, [14, 24, 34, 44, 54, 64, 100]).astype(str)
vars_h = ['Edad (grupos)', 'Anio', 'Sexo', 'Localidad', 'Regimen', 'Escolaridad', 'Ingresos', 'TipoUTI', 'ClaseUTI', 'FormaPago', 'Ocupacion', 'Agente', 'Lesion_Modelo']
lab = ['Edad', 'Año', 'Sexo', 'Localidad', 'Régimen', 'Escolaridad', 'Ingresos', 'Tipo UTI', 'Clase UTI', 'Forma pago', 'Ocupación', 'Agente', 'LESIÓN']
Mv = np.array([[1.0 if i == j else cramers_v(H[a].astype(str), H[b].astype(str)) for j, b in enumerate(vars_h)] for i, a in enumerate(vars_h)])
R['cramer_objetivo'] = {lab[i]: round(Mv[i, -1], 3) for i in range(len(lab) - 1)}
R['cramer_top_pares'] = sorted([(lab[i], lab[j], round(Mv[i, j], 3)) for i in range(len(lab)) for j in range(i + 1, len(lab))], key=lambda x: -x[2])[:6]
num = pd.DataFrame({'Edad': df.Edad, 'Año': df.Anio, 'Escolaridad': np.where(esc >= 0, esc, np.nan),
                    'Ingresos': pd.Series(pd.Categorical(df.Ingresos, categories=ING).codes).replace(-1, np.nan)})
Cp = num.corr(); R['pearson'] = Cp.round(3).to_dict()
fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.6), gridspec_kw={'width_ratios': [1, 2.1]})
im = axs[0].imshow(Cp.values, cmap='RdBu_r', vmin=-1, vmax=1); axs[0].grid(False)
axs[0].set_xticks(range(4), Cp.columns, rotation=40, ha='right'); axs[0].set_yticks(range(4), Cp.columns)
for i in range(4):
    for j in range(4): axs[0].text(j, i, f'{Cp.values[i, j]:.2f}'.replace('.', ','), ha='center', va='center', fontsize=8, color='white' if abs(Cp.values[i, j]) > .6 else TI)
axs[0].set_title('Pearson (numéricas y ordinales)', loc='left', fontsize=9.5)
im2 = axs[1].imshow(Mv, cmap='Blues', vmin=0, vmax=1); axs[1].grid(False)
axs[1].set_xticks(range(len(lab)), lab, rotation=55, ha='right', fontsize=8); axs[1].set_yticks(range(len(lab)), lab, fontsize=8)
for i in range(len(lab)):
    for j in range(len(lab)): axs[1].text(j, i, f'{Mv[i, j]:.2f}'.replace('.', ',')[1:] if Mv[i, j] < 1 else '1', ha='center', va='center', fontsize=6.5, color='white' if Mv[i, j] > .55 else TI)
axs[1].set_title('V de Cramér (asociación entre categóricas, 0 a 1)', loc='left', fontsize=9.5)
fig.colorbar(im2, ax=axs[1], shrink=.75)
fig.suptitle('Figura 5. Mapas de calor de correlación y asociación', x=.01, ha='left', fontweight='bold', fontsize=10.5)
fig.tight_layout(); fig.savefig(FIG / '05_heatmaps.png'); plt.close(fig)

# =============== 4. PARTICIÓN 70/15/15 ===============
y = df['Lesion_Modelo']
idx_tr, idx_tmp = train_test_split(df.index, test_size=0.30, random_state=42, stratify=y)
idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, random_state=42, stratify=y[idx_tmp])
parts = {'Entrenamiento': idx_tr, 'Validación': idx_va, 'Prueba': idx_te}
R['particion'] = {k: [int(len(v)), round(100 * len(v) / len(df), 1)] for k, v in parts.items()}
dist = pd.DataFrame({k: y[v].value_counts(normalize=True) for k, v in parts.items()}).fillna(0)
dist = dist.loc[y.value_counts().index]
R['max_dif_clase_pp'] = round(float((dist.max(axis=1) - dist.min(axis=1)).max() * 100), 2)
R['min_casos_clase'] = {k: int(y[v].value_counts().min()) for k, v in parts.items()}
R['edad_por_parte'] = {k: [round(float(df.Edad[v].mean()), 2), round(float(df.Edad[v].std()), 2)] for k, v in parts.items()}
R['fem_por_parte'] = {k: round(100 * float((df.Sexo[v] == 'Femenino').mean()), 1) for k, v in parts.items()}
fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.6), gridspec_kw={'width_ratios': [2.4, 1, 1]})
top = dist.head(10); x = np.arange(len(top)); w = .27
for i, (k, c) in enumerate(zip(parts, [AZ, NA, VE])):
    axs[0].bar(x + (i - 1) * w, top[k] * 100, w, color=c, label=k)
axs[0].set_xticks(x, [s[:22] + ('…' if len(s) > 22 else '') for s in top.index], rotation=55, ha='right', fontsize=7)
axs[0].set_ylabel('% de registros del subconjunto'); axs[0].legend(frameon=False, fontsize=8)
axs[0].set_title('10 clases más frecuentes del objetivo', loc='left', fontsize=9.5); axs[0].grid(axis='x', visible=False)
bp = axs[1].boxplot([df.Edad[v] for v in parts.values()], widths=.55, patch_artist=True, medianprops=dict(color=TI, lw=1.5), flierprops=dict(markersize=2))
for p, c in zip(bp['boxes'], [AZ, NA, VE]): p.set_facecolor(c); p.set_alpha(.55)
axs[1].set_xticks([1, 2, 3], ['Entren.', 'Valid.', 'Prueba']); axs[1].set_title('Edad', loc='left', fontsize=9.5)
fem = [R['fem_por_parte'][k] for k in parts]
axs[2].bar([0, 1, 2], fem, color=[AZ, NA, VE], width=.6); axs[2].set_ylim(0, 60)
for i, v in enumerate(fem): axs[2].text(i, v + 1, f'{v:.1f} %'.replace('.', ','), ha='center', fontsize=8.5, color=TI)
axs[2].set_xticks([0, 1, 2], ['Entren.', 'Valid.', 'Prueba']); axs[2].set_title('% mujeres', loc='left', fontsize=9.5); axs[2].grid(axis='x', visible=False)
fig.suptitle('Figura 6. Distribución de clases y variables en cada subconjunto (70/15/15 estratificado)', x=.01, ha='left', fontweight='bold', fontsize=10.5)
fig.tight_layout(); fig.savefig(FIG / '06_particion.png'); plt.close(fig)

# =============== 5. PIPELINE ===============
FEAT = NOM + ['Escolaridad', 'Ingresos', 'Edad', 'Anio']
pre = ColumnTransformer([
    ('onehot', Pipeline([('imputar', SimpleImputer(strategy='most_frequent')),
                         ('codificar', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10, sparse_output=False))]), NOM),
    ('ordinal', Pipeline([('imputar', SimpleImputer(strategy='most_frequent')), ('codificar', OrdinalEncoder(categories=[ESC, ING])),
                          ('escalar', MinMaxScaler())]), ['Escolaridad', 'Ingresos']),
    ('zscore', StandardScaler(), ['Edad']), ('minmax', MinMaxScaler(), ['Anio'])])
pipe = Pipeline([('preprocesamiento', pre), ('seleccion', VarianceThreshold(threshold=0.01)),
                 ('modelo', LogisticRegression(max_iter=3000, C=0.3))])
Xtr, Xva, Xte = df.loc[idx_tr, FEAT], df.loc[idx_va, FEAT], df.loc[idx_te, FEAT]
pipe.fit(Xtr, y[idx_tr])                                            # todo se ajusta SOLO con entrenamiento
n_pre = pipe.named_steps['preprocesamiento'].transform(Xtr.head(1)).shape[1]
n_sel = int(pipe.named_steps['seleccion'].get_support().sum())
R['dim_etapas'] = {'Crudo': 14, 'Tras limpieza': 14, 'Entradas\ndel modelo': len(FEAT), 'Tras codificación\ny escalado': int(n_pre), 'Tras\nselección': n_sel}
def met(Xs, ys):
    P = pipe.predict_proba(Xs); cl = pipe.classes_
    return [round(accuracy_score(ys, cl[P.argmax(1)]), 4), round(top_k_accuracy_score(ys, P, k=3, labels=cl), 4)]
R['metricas'] = {'Entrenamiento': met(Xtr, y[idx_tr]), 'Validación': met(Xva, y[idx_va]), 'Prueba': met(Xte, y[idx_te])}
prior = y[idx_tr].value_counts(normalize=True)
R['base_prueba'] = round(float((y[idx_te] == prior.index[0]).mean()), 4)

# Diagrama del pipeline
fig, ax = plt.subplots(figsize=(10.5, 5.2)); ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis('off')
def caja(x, yb, w, h, titulo, cuerpo, color, tcol='white'):
    ax.add_patch(FancyBboxPatch((x, yb), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=color, ec='none'))
    ax.text(x + w / 2, yb + h - 2.3, titulo, ha='center', va='top', fontsize=8.8, fontweight='bold', color=tcol)
    ax.text(x + w / 2, yb + h - 6.0, cuerpo, ha='center', va='top', fontsize=7.2, color=tcol, linespacing=1.35)
def flecha(x1, y1, x2, y2): ax.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='-|>', color=T2, lw=1.4))
C1, C2, C3 = '#1F5F4F', '#2a78d6', '#eb6834'
caja(1, 36, 17, 17, 'Datos crudos', f'obs_salud1.csv\n{R["shape_crudo"][0]:,} × 14\nLatin-1, separador «;»'.replace(',', '.'), '#4b5852')
caja(22, 36, 20, 17, '1. Limpieza', 'Espacios y «0» / «NS/NR»\n→ faltantes\nDuplicados (264)\nFilas sin ocupación u objetivo\nAgrupar clases < 30', C1)
caja(46, 36, 20, 17, '2. Partición 70/15/15', f'Estratificada por lesión\nsemilla 42\n{R["particion"]["Entrenamiento"][0]:,} / {R["particion"]["Validación"][0]:,} / {R["particion"]["Prueba"][0]:,}'.replace(',', '.'), '#4b5852')
caja(70, 36, 28, 17, 'Anonimización (para compartir)', 'Edad en decenios, ocupación ≥ 100\nAño en 2 periodos, sin localidad\nSupresión de grupos con k < 5\nHuella SHA-256 del original', '#8a5a00')
flecha(18.6, 44.5, 21.5, 44.5); flecha(42.6, 44.5, 45.5, 44.5)
ax.text(50, 26.6, 'Pipeline de scikit-learn: se ajusta solo con ENTRENAMIENTO y se aplica igual a validación y prueba', ha='center', fontsize=8.3, color=TI, fontweight='bold')
ax.add_patch(FancyBboxPatch((1, 1), 97, 28.2, boxstyle='round,pad=0.4,rounding_size=1.5', fc='none', ec=GR, ls='--'))
caja(3, 3, 22, 21, '3. Imputar y codificar', 'Moda para faltantes\nOne-Hot: 8 nominales\n(sexo, localidad, régimen,\nocupación, UTI, pago, agente)\nOrdinal: escolaridad, ingresos', C2)
caja(28, 3, 20, 21, '4. Escalado', 'Z-score: edad\nMin-Max: año y\ncódigos ordinales', C2)
caja(51, 3, 21, 21, '5. Selección', f'VarianceThreshold\n(varianza < 0,01)\n{n_pre} → {n_sel} columnas', C2)
caja(75, 3, 21, 21, '6. Modelo', 'Regresión logística\nmultinomial (L2)\n26 clases de lesión', C3)
flecha(25.6, 15, 27.5, 15); flecha(48.6, 15, 50.5, 15); flecha(72.6, 15, 74.5, 15); flecha(56, 35.5, 56, 29.6)
ax.set_title('Figura 7. Diagrama de flujo del pipeline de preprocesamiento', loc='left', fontsize=10.5)
fig.savefig(FIG / '07_pipeline.png'); plt.close(fig)

# Dimensión en cada etapa
fig, ax = plt.subplots(figsize=(8.4, 2.8))
et = list(R['dim_etapas'].items()); ax.bar(range(len(et)), [v for _, v in et], color=[GR, C1, GR, C2, C2], width=.6)
for i, (_, v) in enumerate(et): ax.text(i, v + 3, str(v), ha='center', fontsize=9, color=TI)
ax.set_xticks(range(len(et)), [k for k, _ in et], fontsize=8.3); ax.set_ylabel('Columnas'); ax.set_ylim(0, max(v for _, v in et) * 1.18)
ax.set_title('Figura 8. Número de columnas en cada etapa del pipeline', loc='left'); ax.grid(axis='x', visible=False)
fig.savefig(FIG / '08_dimension_etapas.png'); plt.close(fig)

# =============== 6. SEGURIDAD: riesgo de reidentificación (k-anonimato) ===============
QI = ['Edad', 'Sexo', 'Localidad', 'Ocupacion', 'Anio']
def k_stats(d, qi):
    g = d.groupby(qi, dropna=False).size(); k_reg = d.merge(g.rename('k').reset_index(), on=qi)['k']
    return {'unicos': int((k_reg == 1).sum()), 'pct_unicos': round(100 * float((k_reg == 1).mean()), 1),
            'k_menor_5': int((k_reg < 5).sum()), 'pct_k_menor_5': round(100 * float((k_reg < 5).mean()), 1)}
R['k_antes'] = k_stats(df, QI)
# Nivel 1 · generalización leve
leve = df.copy(); leve['Edad'] = (leve.Edad // 5 * 5).astype(int)
f_oc = leve.Ocupacion.value_counts(); leve['Ocupacion'] = leve.Ocupacion.where(leve.Ocupacion.map(f_oc) >= 10, 'Otra ocupación')
R['k_leve'] = k_stats(leve, QI)
# Nivel 2 · generalización fuerte para publicar
anon = df.drop(columns=['Lesion_Modelo']).copy()
anon['Edad'] = (anon.Edad // 10 * 10).astype(int).astype(str) + '–' + (anon.Edad // 10 * 10 + 9).astype(int).astype(str)
anon['Ocupacion'] = anon.Ocupacion.where(anon.Ocupacion.map(anon.Ocupacion.value_counts()) >= 100, 'Otra ocupación')
anon['Anio'] = np.where(anon.Anio <= 2020, '2017–2020', '2021–2025')
anon['Localidad'] = 'Bogotá D.C.'                     # se retira la localidad del archivo publicable
R['ocupaciones_publicables'] = int(anon.Ocupacion.nunique())
R['k_fuerte'] = k_stats(anon, QI)
# Nivel 3 · supresión de combinaciones con menos de 5 registros
g = anon.groupby(QI).size(); anon = anon.merge(g.rename('_k').reset_index(), on=QI)
R['suprimidos'] = int((anon._k < 5).sum())
anon = anon[anon._k >= 5].drop(columns=['_k', 'Localidad'])
R['k_final'] = k_stats(anon.assign(Localidad='-'), QI); R['filas_anonimizado'] = len(anon)
R['pct_conservado'] = round(100 * len(anon) / len(df), 1)
anon.to_csv(RAIZ / 'obs_salud_anonimizado.csv', index=False, encoding='utf-8-sig')
fig, ax = plt.subplots(figsize=(9, 3.1))
etq = ['Original\n(edad exacta)', 'Generalización leve\n(edad en 5 años,\nocupación ≥ 10)', 'Generalización fuerte\n(edad en 10 años, ocup. ≥ 100,\n2 periodos, sin localidad)', 'Fuerte + supresión\nde k < 5']
ks_ = [R['k_antes'], R['k_leve'], R['k_fuerte'], R['k_final']]
v1 = [k_['pct_unicos'] for k_ in ks_]; v5 = [k_['pct_k_menor_5'] for k_ in ks_]
xx = np.arange(4); ax.bar(xx - .18, v1, .34, color=NA, label='Registros únicos (k = 1)'); ax.bar(xx + .18, v5, .34, color=GR, label='Registros con k < 5')
for i in range(4):
    ax.text(i - .18, v1[i] + 1.5, f'{v1[i]:.1f} %'.replace('.', ','), ha='center', fontsize=8, color=TI)
    ax.text(i + .18, v5[i] + 1.5, f'{v5[i]:.1f} %'.replace('.', ','), ha='center', fontsize=8, color=TI)
ax.set_xticks(xx, etq, fontsize=7.8); ax.set_ylabel('% de registros'); ax.set_ylim(0, 108); ax.legend(frameon=False, fontsize=8)
ax.set_title('Figura 9. Riesgo de reidentificación (cuasi-identificadores: edad, sexo, localidad, ocupación, año)', loc='left', fontsize=10); ax.grid(axis='x', visible=False)
fig.savefig(FIG / '09_reidentificacion.png'); plt.close(fig)

json.dump(R, open(RAIZ / 'resultados_aa3_ev01.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
