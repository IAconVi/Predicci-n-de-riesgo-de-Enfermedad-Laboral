"""AA2-EV02 · Reducción de dimensionalidad sobre la matriz transformada de AA2-EV01.

Entrada : obs_salud1.csv (se repite la depuración y transformación de AA2-EV01 para partir
          exactamente de la misma matriz de 22.167 × 167).
Salidas : figuras_aa2_ev02/*.png, resultados_aa2_ev02.json, matriz_pca95.csv
"""
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler
warnings.filterwarnings('ignore')

RAIZ = Path(__file__).parent; FIG = RAIZ / 'figuras_aa2_ev02'; FIG.mkdir(exist_ok=True)
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

# ---------- 1. Reproducir la matriz de AA2-EV01 ----------
df = pd.read_csv(RAIZ / 'obs_salud1.csv', sep=';', encoding='latin-1')
df.columns = [c.strip() for c in df.columns]; df = df.rename(columns=COL)
for c in df.columns:
    if c in ('Anio', 'Edad'): continue
    df[c] = df[c].astype('string').str.strip().str.replace(r'\s+', ' ', regex=True)
    df.loc[df[c].isin(['0', '']), c] = pd.NA
df.loc[df.Ingresos == 'NS / NR', 'Ingresos'] = pd.NA
df = df.drop_duplicates().dropna(subset=['Ocupacion', 'Agente', 'Sexo', 'Edad', 'Lesion']).reset_index(drop=True)
for c in df.columns:
    if df[c].isna().any(): df[c] = df[c].fillna(df[c].mode().iloc[0])
vc = df.Lesion.value_counts(); raras = vc[vc < 30].index
df['Lesion_Modelo'] = df.Lesion.where(~df.Lesion.isin(raras), 'Otras lesiones / registros múltiples')
y = df.Lesion_Modelo.to_numpy()
ct = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10, sparse_output=False), NOM),
    ('ordinal', Pipeline([('c', OrdinalEncoder(categories=[ESC, ING])), ('s', MinMaxScaler())]), ['Escolaridad', 'Ingresos']),
    ('zscore', StandardScaler(), ['Edad']), ('minmax', MinMaxScaler(), ['Anio'])])
X = pd.DataFrame(ct.fit_transform(df[NOM + ['Escolaridad', 'Ingresos', 'Edad', 'Anio']]), columns=ct.get_feature_names_out())
X.columns = [c.split('__', 1)[1] for c in X.columns]
R['shape'] = list(X.shape); R['nulos'] = int(X.isna().sum().sum()); R['clases'] = int(len(np.unique(y)))
R['bloques'] = {c: int(sum(n.startswith(c + '_') for n in X.columns)) for c in NOM}
R['pct_ceros'] = round(100 * float((X.to_numpy() == 0).mean()), 1)
var = X.var(ddof=0); R['var_edad'] = round(float(var['Edad']), 3)
R['var_onehot_mediana'] = round(float(var[[c for c in X.columns if c.split('_')[0] in NOM]].median()), 4)
R['var_top5'] = var.sort_values(ascending=False).head(5).round(3).to_dict()

tr, te = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42, stratify=y)
Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
R['n_train'], R['n_test'] = len(tr), len(te)

# ---------- 2. PCA sobre la matriz de entrenamiento ----------
pca_full = PCA(random_state=42).fit(Xtr)
cum = np.cumsum(pca_full.explained_variance_ratio_)
k = {p: int(np.searchsorted(cum, p) + 1) for p in (0.80, 0.90, 0.95, 0.99)}
R['k_por_varianza'] = k
R['var_pc'] = [round(float(v) * 100, 2) for v in pca_full.explained_variance_ratio_[:10]]
R['cum_10'] = round(float(cum[9]) * 100, 1)
# Kaiser adaptado no aplica (no estandarizado); se usa varianza acumulada
comp = pd.DataFrame(pca_full.components_[:5], columns=X.columns, index=[f'CP{i+1}' for i in range(5)])
R['cargas'] = {pc: comp.loc[pc].reindex(comp.loc[pc].abs().sort_values(ascending=False).index).head(6).round(3).to_dict() for pc in comp.index}

# ---------- 3. Selección de características por varianza y correlación ----------
vt = VarianceThreshold(threshold=0.01).fit(Xtr)          # descarta indicadoras presentes en <~1 % de registros
keep_v = X.columns[vt.get_support()]
corr = Xtr[keep_v].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1))
pares = [(a, b, round(float(upper.loc[a, b]), 3)) for a in upper.index for b in upper.columns if upper.loc[a, b] > 0.9]
drop_c = sorted({b for a, b, _ in pares})
keep_sel = [c for c in keep_v if c not in drop_c]
R['sel'] = {'umbral_varianza': 0.01, 'tras_varianza': int(len(keep_v)), 'eliminadas_varianza': int(X.shape[1] - len(keep_v)),
            'pares_corr_090': pares, 'eliminadas_corr': drop_c, 'final': int(len(keep_sel))}
R['sel_eliminadas_por_bloque'] = {c: int(sum(n.startswith(c + '_') for n in X.columns if n not in keep_v)) for c in NOM}

# ---------- 4. Impacto en el modelo (misma partición, mismo clasificador) ----------
def evaluar(nombre, Xa, Xb, n_attr):
    t0 = time.time()
    m = LogisticRegression(max_iter=3000, C=0.3).fit(Xa, ytr)
    t = time.time() - t0
    P = m.predict_proba(Xb); cl = m.classes_
    return {'Representación': nombre, 'Atributos': n_attr, 'top1': round(accuracy_score(yte, cl[P.argmax(1)]), 4),
            'top3': round(top_k_accuracy_score(yte, P, k=3, labels=cl), 4), 'log_loss': round(log_loss(yte, P, labels=cl), 4),
            'seg_entrenamiento': round(t, 1)}
res = [evaluar('Matriz completa (AA2-EV01)', Xtr, Xte, X.shape[1])]
for p in (0.90, 0.95):
    pc = PCA(n_components=k[p], random_state=42).fit(Xtr)
    res.append(evaluar(f'PCA {int(p*100)} % de varianza', pc.transform(Xtr), pc.transform(Xte), k[p]))
res.append(evaluar('Selección varianza + correlación', Xtr[keep_sel], Xte[keep_sel], len(keep_sel)))
for kk in (10, 30):
    pc = PCA(n_components=kk, random_state=42).fit(Xtr)
    res.append(evaluar(f'PCA {kk} componentes', pc.transform(Xtr), pc.transform(Xte), kk))
R['impacto'] = res
curve = []
for kk in (2, 5, 10, 20, 30, 50, 75, 100, 125, 150):
    pc = PCA(n_components=kk, random_state=42).fit(Xtr)
    m = LogisticRegression(max_iter=3000, C=0.3).fit(pc.transform(Xtr), ytr)
    P = m.predict_proba(pc.transform(Xte)); curve.append((kk, round(float(cum[kk - 1]) * 100, 1), round(accuracy_score(yte, m.classes_[P.argmax(1)]), 4)))
R['curva'] = curve

# Guardar matriz reducida (PCA 95 %) para las siguientes etapas
pc95 = PCA(n_components=k[0.95], random_state=42).fit(Xtr)
Z = pd.DataFrame(pc95.transform(X), columns=[f'CP{i+1}' for i in range(k[0.95])]); Z['objetivo'] = y
Z.to_csv(RAIZ / 'matriz_pca95.csv', index=False, float_format='%.5g')

# ---------- 5. Gráficos ----------
AZ, NA, GR, TI, T2 = '#2a78d6', '#eb6834', '#9aa3a0', '#15201b', '#4b5852'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.edgecolor': '#c9d1cd', 'axes.labelcolor': T2,
    'xtick.color': T2, 'ytick.color': T2, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
    'grid.color': '#e3e8e5', 'axes.axisbelow': True, 'axes.titleweight': 'bold', 'axes.titlesize': 10.5,
    'axes.titlecolor': TI, 'figure.dpi': 150, 'savefig.bbox': 'tight'})
# Fig 1: varianza explicada
fig, ax = plt.subplots(figsize=(8.5, 3.4))
n = 120
ax.bar(range(1, n + 1), pca_full.explained_variance_ratio_[:n] * 100, color=AZ, width=0.9, label='Varianza de cada componente')
ax2 = ax.twinx(); ax2.plot(range(1, n + 1), cum[:n] * 100, color=NA, lw=2, label='Varianza acumulada'); ax2.set_ylim(0, 102)
ax2.spines['right'].set_visible(True); ax2.grid(False); ax2.set_ylabel('Acumulada (%)')
for p, ls in [(0.90, '--'), (0.95, ':')]:
    ax2.axhline(p * 100, color=T2, lw=1, ls=ls); ax2.axvline(k[p], color=T2, lw=1, ls=ls)
    ax2.text(k[p] + 1.5, 78 if p == 0.90 else 66, f'{int(p*100)} % → {k[p]} comp.', fontsize=8, color=TI, bbox=dict(fc='white', ec='none', pad=1))
ax.set_xlabel('Componente principal'); ax.set_ylabel('Individual (%)')
ax.set_title('Varianza explicada por los componentes principales (primeros 120 de 167)', loc='left')
fig.savefig(FIG / '01_varianza_explicada.png'); plt.close(fig)
# Fig 2: dispersión CP1 vs CP2 coloreada por las 3 lesiones más frecuentes
Ztr = pca_full.transform(Xtr)[:, :2]; top3 = pd.Series(ytr).value_counts().index[:3]
rng = np.random.default_rng(0); idx = rng.choice(len(Ztr), 4000, replace=False)
fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.scatter(Ztr[idx, 0], Ztr[idx, 1], s=5, color='#d5ddd8', alpha=.6, label='Otras lesiones', edgecolors='none')
for c, col in zip(top3, [AZ, NA, '#1baf7a']):
    msk = (ytr[idx] == c); ax.scatter(Ztr[idx][msk, 0], Ztr[idx][msk, 1], s=7, alpha=.65, color=col, label=c, edgecolors='none')
ax.set_xlabel(f'CP1 ({R["var_pc"][0]:.1f} % de la varianza)'.replace('.', ',')); ax.set_ylabel(f'CP2 ({R["var_pc"][1]:.1f} %)'.replace('.', ','))
ax.legend(frameon=False, fontsize=8, markerscale=2.5, loc='upper left', bbox_to_anchor=(1.01, 1))
ax.set_title('Registros proyectados en los dos primeros componentes (muestra de 4.000)', loc='left')
fig.savefig(FIG / '02_dispersion_cp1_cp2.png'); plt.close(fig)
# Fig 3: mapa de calor de cargas
top_vars = comp.abs().max().sort_values(ascending=False).head(15).index
fig, ax = plt.subplots(figsize=(7.8, 5.2))
M = comp[top_vars].T
im = ax.imshow(M.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(5), M.columns); ax.set_yticks(range(len(top_vars)), [v[:45] for v in top_vars], fontsize=7.8); ax.grid(False)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M.values[i, j]; ax.text(j, i, f'{v:.2f}'.replace('.', ','), ha='center', va='center', fontsize=7, color='white' if abs(v) > .55 else TI)
fig.colorbar(im, ax=ax, shrink=.8, label='Carga')
ax.set_title('Mapa de calor de cargas: 15 variables con más peso en CP1–CP5', loc='left')
fig.savefig(FIG / '03_mapa_calor_cargas.png'); plt.close(fig)
# Fig 4: mapa de calor de correlación antes (variables originales no one-hot + CP) -> correlación de las 20 variables de mayor varianza vs CP
sub = X[var.sort_values(ascending=False).head(12).index]
fig, axs = plt.subplots(1, 2, figsize=(10, 4.3))
c1 = sub.corr().values; im = axs[0].imshow(c1, cmap='RdBu_r', vmin=-1, vmax=1)
axs[0].set_xticks(range(12), [c[:18] for c in sub.columns], rotation=70, fontsize=6.5, ha='right'); axs[0].set_yticks(range(12), [c[:18] for c in sub.columns], fontsize=6.5)
axs[0].set_title('Antes: 12 variables de mayor varianza', loc='left', fontsize=9.5); axs[0].grid(False)
Zc = pd.DataFrame(pca_full.transform(Xtr)[:, :12]).corr().values
axs[1].imshow(Zc, cmap='RdBu_r', vmin=-1, vmax=1); axs[1].set_xticks(range(12), [f'CP{i+1}' for i in range(12)], rotation=70, fontsize=7)
axs[1].set_yticks(range(12), [f'CP{i+1}' for i in range(12)], fontsize=7); axs[1].grid(False)
axs[1].set_title('Después: 12 primeros componentes (sin correlación)', loc='left', fontsize=9.5)
fig.colorbar(im, ax=axs, shrink=.8, label='Correlación de Pearson')
fig.savefig(FIG / '04_correlacion_antes_despues.png'); plt.close(fig)
R['max_corr_antes'] = round(float(np.abs(c1 - np.eye(12)).max()), 3); R['max_corr_despues'] = round(float(np.abs(Zc - np.eye(12)).max()), 6)
# Fig 5: acierto vs componentes
fig, ax = plt.subplots(figsize=(7.8, 3.3))
ks = [c[0] for c in curve]; acc = [c[2] * 100 for c in curve]
ax.plot(ks, acc, color=AZ, lw=2, marker='o'); ax.axhline(res[0]['top1'] * 100, color=NA, ls='--', lw=1.5, label=f'Matriz completa ({res[0]["top1"]*100:.1f} %)'.replace('.', ','))
ax.axvline(k[0.95], color=T2, ls=':', lw=1); ax.text(k[0.95] + 2, min(acc) + 1, f'95 % varianza\n({k[0.95]} comp.)', fontsize=8, color=TI)
ax.set_xlabel('Número de componentes principales'); ax.set_ylabel('Acierto en la 1.ª opción (%)'); ax.legend(frameon=False, loc='lower right')
ax.set_title('Acierto del modelo según el número de componentes', loc='left')
fig.savefig(FIG / '05_acierto_vs_componentes.png'); plt.close(fig)

json.dump(R, open(RAIZ / 'resultados_aa2_ev02.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
