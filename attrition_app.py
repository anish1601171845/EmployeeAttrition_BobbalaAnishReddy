import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score, precision_recall_fscore_support
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AttritionIQ — HR Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stFileUploader label { color: #94a3b8 !important; font-size: 0.78rem !important; letter-spacing: 0.05em; text-transform: uppercase; }

/* Main background */
.main { background: #0d1117; }
.block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

/* Page header */
.page-header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 60%, #1a1040 100%);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
}
.page-header h1 {
    font-size: 2rem; font-weight: 700; color: #f1f5f9;
    margin: 0 0 0.25rem 0; letter-spacing: -0.02em;
}
.page-header p { color: #64748b; font-size: 0.9rem; margin: 0; }
.page-header .badge {
    display: inline-block; background: #7c3aed22; border: 1px solid #7c3aed55;
    color: #a78bfa; font-size: 0.72rem; padding: 2px 10px; border-radius: 20px;
    font-weight: 500; letter-spacing: 0.04em; margin-bottom: 0.6rem;
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.2rem; }
.metric-card {
    flex: 1; background: #161b26; border: 1px solid #1e2a3a;
    border-radius: 12px; padding: 1.2rem 1.4rem;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #334155; }
.metric-card .label { color: #64748b; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.35rem; }
.metric-card .value { color: #f1f5f9; font-size: 1.8rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; line-height: 1; }
.metric-card .sub { color: #475569; font-size: 0.8rem; margin-top: 0.25rem; }
.metric-card.accent { border-color: #7c3aed44; background: #1a1040; }
.metric-card.accent .value { color: #a78bfa; }
.metric-card.green { border-color: #06402b; background: #052018; }
.metric-card.green .value { color: #34d399; }
.metric-card.red { border-color: #4c1414; background: #1f0a0a; }
.metric-card.red .value { color: #f87171; }

/* Section headers */
.section-title {
    font-size: 1rem; font-weight: 600; color: #cbd5e1;
    padding: 0.5rem 0 0.75rem 0;
    border-bottom: 1px solid #1e2130;
    margin-bottom: 1.2rem;
    letter-spacing: -0.01em;
}
.section-title span { color: #7c3aed; margin-right: 0.4rem; }

/* Insight cards */
.insight-card {
    background: #161b26; border-left: 3px solid #7c3aed;
    border-radius: 0 10px 10px 0; padding: 0.9rem 1.2rem;
    margin-bottom: 0.75rem; font-size: 0.85rem; color: #94a3b8;
    line-height: 1.6;
}
.insight-card strong { color: #e2e8f0; }
.insight-card.amber { border-color: #f59e0b; }
.insight-card.red { border-color: #ef4444; }
.insight-card.green { border-color: #10b981; }

/* Nav tabs */
div[data-testid="stHorizontalBlock"] { gap: 0.75rem; }

/* Chart container */
.chart-wrap {
    background: #161b26; border: 1px solid #1e2130;
    border-radius: 12px; padding: 1rem;
}

/* Prediction result */
.pred-high {
    background: #1f0a0a; border: 1px solid #ef4444;
    border-radius: 12px; padding: 1.5rem; text-align: center;
}
.pred-low {
    background: #052018; border: 1px solid #10b981;
    border-radius: 12px; padding: 1.5rem; text-align: center;
}
.pred-score {
    font-size: 3.5rem; font-weight: 800;
    font-family: 'JetBrains Mono', monospace; line-height: 1;
}
.pred-label { font-size: 1rem; margin-top: 0.4rem; }

/* Table styling */
table { width: 100%; border-collapse: collapse; }
th { background: #161b26; color: #64748b; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; padding: 0.6rem 1rem; text-align: left; }
td { padding: 0.55rem 1rem; border-bottom: 1px solid #1e2130; color: #cbd5e1; font-size: 0.84rem; }
tr:last-child td { border-bottom: none; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Colour palette for charts ──────────────────────────────────────────────────
PALETTE = {
    'bg':      '#161b26',
    'border':  '#1e2130',
    'text':    '#e2e8f0',
    'muted':   '#64748b',
    'purple':  '#7c3aed',
    'green':   '#34d399',
    'red':     '#f87171',
    'amber':   '#f59e0b',
    'blue':    '#60a5fa',
    'models':  ['#60a5fa','#34d399','#f59e0b','#f87171','#a78bfa'],
}

def style_axes(ax):
    ax.set_facecolor(PALETTE['bg'])
    ax.figure.patch.set_facecolor(PALETTE['bg'])
    ax.tick_params(colors=PALETTE['muted'], labelsize=8)
    ax.xaxis.label.set_color(PALETTE['muted'])
    ax.yaxis.label.set_color(PALETTE['muted'])
    ax.title.set_color(PALETTE['text'])
    for spine in ax.spines.values():
        spine.set_color(PALETTE['border'])
    ax.grid(color=PALETTE['border'], linewidth=0.5, alpha=0.5)

# ════════════════════════════════════════════════════════════════════════════════
# DATA & MODEL (cached)
# ════════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_and_prepare(uploaded_bytes):
    import io
    df = pd.read_csv(io.BytesIO(uploaded_bytes))
    df_eda = df.copy()
    df_eda['AttritionNum'] = df_eda['Attrition'].map({'Yes': 1, 'No': 0})

    cols_to_drop = ['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours']
    df2 = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    df2['Attrition'] = df2['Attrition'].map({'Yes': 1, 'No': 0})
    cat_cols = df2.select_dtypes(include='object').columns.tolist()
    df_enc = pd.get_dummies(df2, columns=cat_cols, drop_first=True)
    X = df_enc.drop('Attrition', axis=1)
    y = df_enc['Attrition']
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return df_eda, X, y, X_scaled, scaler

@st.cache_resource(show_spinner=False)
def train_models(data_hash, _X_scaled, _y):
    X_scaled, y = _X_scaled, _y
    neg, pos = (y == 0).sum(), (y == 1).sum()
    spw = round(neg / pos, 2)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
        'XGBoost':             XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                                              colsample_bytree=0.8, scale_pos_weight=spw, eval_metric='logloss',
                                              random_state=42, n_jobs=-1),
        'LightGBM':            LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, num_leaves=31,
                                               subsample=0.8, colsample_bytree=0.8, is_unbalance=True,
                                               random_state=42, n_jobs=-1, verbose=-1),
    }

    preds, probs = {}, {}
    for name, m in models.items():
        if name == 'XGBoost':
            m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        else:
            m.fit(X_train, y_train)
        preds[name] = m.predict(X_test)
        probs[name] = m.predict_proba(X_test)[:, 1]

    # threshold tuning on XGBoost
    xgb_prob = probs['XGBoost']
    thresholds = np.arange(0.20, 0.65, 0.05)
    rows = []
    for t in thresholds:
        p_arr = (xgb_prob >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_test, p_arr, pos_label=1, average='binary', zero_division=0)
        rows.append({'Threshold': round(t, 2), 'Precision': round(p, 4), 'Recall': round(r, 4), 'F1': round(f, 4)})
    thresh_df = pd.DataFrame(rows)
    best_t = thresh_df.loc[thresh_df['F1'].idxmax(), 'Threshold']

    # CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = {}
    for name, m in models.items():
        cv_scores[name] = cross_val_score(m, X_scaled, y, cv=cv, scoring='roc_auc', n_jobs=-1)

    # results table
    results = []
    for name in models:
        results.append({
            'Model':     name,
            'Precision': round(precision_score(y_test, preds[name], zero_division=0), 4),
            'Recall':    round(recall_score(y_test, preds[name]), 4),
            'F1':        round(f1_score(y_test, preds[name]), 4),
            'ROC-AUC':   round(roc_auc_score(y_test, probs[name]), 4),
        })
    results_df = pd.DataFrame(results).set_index('Model')

    feat_xgb  = pd.Series(models['XGBoost'].feature_importances_, index=X_scaled.columns)
    feat_lgbm = pd.Series(models['LightGBM'].feature_importances_, index=X_scaled.columns)

    return {
        'models': models, 'preds': preds, 'probs': probs,
        'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
        'results_df': results_df, 'cv_scores': cv_scores,
        'thresh_df': thresh_df, 'best_t': best_t,
        'feat_xgb': feat_xgb, 'feat_lgbm': feat_lgbm, 'spw': spw,
    }

# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 AttritionIQ")
    st.markdown("---")
    uploaded = st.file_uploader("Upload CSV Dataset", type=["csv"],
                                 help="IBM HR Analytics Employee Attrition dataset")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Overview",
        "🔍 Exploratory Analysis",
        "🤖 Model Performance",
        "📈 Feature Importance",
        "🎯 Predict Employee",
        "💡 HR Insights",
    ])
    st.markdown("---")
    st.markdown("<div style='color:#475569;font-size:0.75rem'>IBM HR Analytics · 1,470 Employees · 35 Features</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════════════════════
if uploaded is None:
    st.markdown("""
    <div class="page-header">
        <div class="badge">INTERNSHIP PROJECT — WEEK 2</div>
        <h1>Employee Attrition Prediction</h1>
        <p>Upload the IBM HR Analytics CSV to launch the dashboard.</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("👈  Upload **WA_Fn-UseC_-HR-Employee-Attrition.csv** in the sidebar to get started.")
    st.stop()

with st.spinner("Loading & training models…"):
    raw_bytes = uploaded.getvalue()
    df_eda, X, y, X_scaled, scaler = load_and_prepare(raw_bytes)
    data_hash = hash(raw_bytes)
    out = train_models(data_hash, X_scaled, y)

models    = out['models']
preds     = out['preds']
probs     = out['probs']
X_test    = out['X_test']
y_test    = out['y_test']
results_df = out['results_df']
cv_scores  = out['cv_scores']
thresh_df  = out['thresh_df']
best_t     = out['best_t']
feat_xgb   = out['feat_xgb']
feat_lgbm  = out['feat_lgbm']

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("""
    <div class="page-header">
        <div class="badge">IBM HR ANALYTICS · 5-MODEL PIPELINE</div>
        <h1>Employee Attrition Dashboard</h1>
        <p>Predicting who leaves — and why — using XGBoost, LightGBM, and 3 baselines.</p>
    </div>
    """, unsafe_allow_html=True)

    total      = len(df_eda)
    left       = df_eda['AttritionNum'].sum()
    stayed     = total - left
    att_rate   = left / total * 100
    best_auc   = results_df['ROC-AUC'].max()
    best_model = results_df['ROC-AUC'].idxmax()

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card"><div class="label">Employees</div><div class="value">{total:,}</div><div class="sub">in dataset</div></div>
      <div class="metric-card red"><div class="label">Attrition Count</div><div class="value">{int(left):,}</div><div class="sub">{att_rate:.1f}% of workforce</div></div>
      <div class="metric-card green"><div class="label">Retention Count</div><div class="value">{int(stayed):,}</div><div class="sub">{100-att_rate:.1f}% stayed</div></div>
      <div class="metric-card accent"><div class="label">Best ROC-AUC</div><div class="value">{best_auc:.3f}</div><div class="sub">{best_model}</div></div>
      <div class="metric-card"><div class="label">Features</div><div class="value">35</div><div class="sub">→ {X_scaled.shape[1]} after encoding</div></div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title"><span>◈</span> Attrition Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor(PALETTE['bg'])
        ax.set_facecolor(PALETTE['bg'])
        labels, sizes = ['Stayed', 'Left'], [stayed, left]
        colors = [PALETTE['green'], PALETTE['red']]
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.55, edgecolor=PALETTE['bg'], linewidth=2))
        for t in texts: t.set_color(PALETTE['muted']); t.set_fontsize(9)
        for a in autotexts: a.set_color(PALETTE['text']); a.set_fontsize(10); a.set_fontweight('bold')
        ax.set_title('Class Balance', color=PALETTE['text'], fontsize=11, pad=10)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown('<div class="section-title"><span>◈</span> Model Comparison (ROC-AUC)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        style_axes(ax)
        names  = list(results_df.index)
        aucs   = results_df['ROC-AUC'].values
        colors = [PALETTE['purple'] if n in ('XGBoost','LightGBM') else PALETTE['muted'] for n in names]
        bars   = ax.bar(range(len(names)), aucs, color=colors, edgecolor=PALETTE['bg'], linewidth=1.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(' ','\n') for n in names], fontsize=7)
        ax.set_ylim(0.7, 0.97)
        ax.set_ylabel('ROC-AUC', fontsize=9)
        ax.set_title('Test Set ROC-AUC', fontsize=10)
        for bar, v in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.003, f'{v:.3f}',
                    ha='center', va='bottom', color=PALETTE['text'], fontsize=8, fontweight='bold')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown('<div class="section-title"><span>◈</span> Dataset Snapshot</div>', unsafe_allow_html=True)
    st.dataframe(df_eda.drop('AttritionNum', axis=1).head(10), use_container_width=True,
                 hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Exploratory Analysis":
    st.markdown("""
    <div class="page-header">
        <div class="badge">EXPLORATORY DATA ANALYSIS</div>
        <h1>Who Is Leaving — and Why?</h1>
        <p>Attrition patterns across departments, roles, income, and tenure.</p>
    </div>
    """, unsafe_allow_html=True)

    # Chart 1: Dept & Role
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title"><span>◈</span> Attrition Rate by Department</div>', unsafe_allow_html=True)
        dept = df_eda.groupby('Department')['AttritionNum'].mean().mul(100).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        style_axes(ax)
        colors = [PALETTE['red'], PALETTE['amber'], PALETTE['green']][:len(dept)]
        bars = ax.bar(dept.index, dept.values, color=colors, edgecolor=PALETTE['bg'], linewidth=1.5)
        ax.set_ylabel('Attrition Rate (%)', fontsize=9)
        ax.set_title('By Department', fontsize=10)
        for bar, v in zip(bars, dept.values):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.5, f'{v:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color=PALETTE['text'])
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.markdown('<div class="section-title"><span>◈</span> Attrition Rate by Job Role</div>', unsafe_allow_html=True)
        role = df_eda.groupby('JobRole')['AttritionNum'].mean().mul(100).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        style_axes(ax)
        palette_r = sns.color_palette('RdYlGn', len(role))
        ax.barh(role.index, role.values, color=palette_r, edgecolor=PALETTE['bg'], linewidth=1)
        ax.set_xlabel('Attrition Rate (%)', fontsize=9)
        ax.set_title('By Job Role', fontsize=10)
        for i, v in enumerate(role.values):
            ax.text(v+0.3, i, f'{v:.1f}%', va='center', fontsize=8, color=PALETTE['text'])
        st.pyplot(fig, use_container_width=True); plt.close()

    # Chart 2: Income & WLB
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title"><span>◈</span> Monthly Income vs Attrition</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        style_axes(ax)
        stayed_inc = df_eda[df_eda['Attrition']=='No']['MonthlyIncome']
        left_inc   = df_eda[df_eda['Attrition']=='Yes']['MonthlyIncome']
        bp = ax.boxplot([stayed_inc, left_inc], patch_artist=True,
                        boxprops=dict(facecolor=PALETTE['bg']),
                        medianprops=dict(color=PALETTE['purple'], linewidth=2),
                        whiskerprops=dict(color=PALETTE['muted']),
                        capprops=dict(color=PALETTE['muted']),
                        flierprops=dict(marker='o', color=PALETTE['muted'], alpha=0.3, markersize=3))
        bp['boxes'][0].set_facecolor('#052018'); bp['boxes'][1].set_facecolor('#1f0a0a')
        ax.set_xticklabels(['Stayed', 'Left'], fontsize=9)
        ax.set_ylabel('Monthly Income ($)', fontsize=9)
        ax.set_title('Income Distribution', fontsize=10)
        for i, (vals, col) in enumerate([(stayed_inc, PALETTE['green']), (left_inc, PALETTE['red'])], 1):
            ax.text(i, vals.median()+100, f'Median\n${vals.median():,.0f}',
                    ha='center', va='bottom', fontsize=8, color=col, fontweight='bold')
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.markdown('<div class="section-title"><span>◈</span> Work-Life Balance vs Attrition</div>', unsafe_allow_html=True)
        wlb = df_eda.groupby('WorkLifeBalance')['AttritionNum'].mean().mul(100)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        style_axes(ax)
        colors = [PALETTE['red'], PALETTE['amber'], PALETTE['green'], PALETTE['blue']]
        bars = ax.bar(wlb.index, wlb.values, color=colors, edgecolor=PALETTE['bg'], linewidth=1.5, width=0.6)
        ax.set_xlabel('Rating (1=Bad  →  4=Best)', fontsize=9)
        ax.set_ylabel('Attrition Rate (%)', fontsize=9)
        ax.set_title('Work-Life Balance Rating', fontsize=10)
        for bar, v in zip(bars, wlb.values):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.4, f'{v:.1f}%',
                    ha='center', fontsize=9, fontweight='bold', color=PALETTE['text'])
        st.pyplot(fig, use_container_width=True); plt.close()

    # Overtime
    st.markdown('<div class="section-title"><span>◈</span> Overtime & Business Travel Impact</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        ot = df_eda.groupby('OverTime')['AttritionNum'].mean().mul(100).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(4, 3))
        style_axes(ax)
        colors = [PALETTE['red'] if v > 20 else PALETTE['green'] for v in ot.values]
        ax.bar(ot.index, ot.values, color=colors, edgecolor=PALETTE['bg'], linewidth=1.5, width=0.5)
        ax.set_ylabel('Attrition Rate (%)', fontsize=9); ax.set_title('OverTime', fontsize=10)
        for x, v in enumerate(ot.values):
            ax.text(x, v+0.5, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold', color=PALETTE['text'])
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        bt = df_eda.groupby('BusinessTravel')['AttritionNum'].mean().mul(100).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(4, 3))
        style_axes(ax)
        colors = [PALETTE['red'], PALETTE['amber'], PALETTE['green']][:len(bt)]
        ax.bar(bt.index, bt.values, color=colors, edgecolor=PALETTE['bg'], linewidth=1.5, width=0.5)
        ax.set_ylabel('Attrition Rate (%)', fontsize=9); ax.set_title('Business Travel', fontsize=10)
        for x, v in enumerate(bt.values):
            ax.text(x, v+0.5, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold', color=PALETTE['text'])
        st.pyplot(fig, use_container_width=True); plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown("""
    <div class="page-header">
        <div class="badge">5-MODEL COMPARISON · STRATIFIED CV · THRESHOLD TUNING</div>
        <h1>Model Performance</h1>
        <p>Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM.</p>
    </div>
    """, unsafe_allow_html=True)

    # Model comparison table
    st.markdown('<div class="section-title"><span>◈</span> Test Set Metrics</div>', unsafe_allow_html=True)
    best_model_name = results_df['ROC-AUC'].idxmax()
    def highlight_best(row):
        if row.name == best_model_name:
            return ['background-color: #1a1040; color: #a78bfa'] * len(row)
        return [''] * len(row)
    st.dataframe(results_df.style.apply(highlight_best, axis=1).format('{:.4f}'),
                 use_container_width=True)

    # ROC Curves
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title"><span>◈</span> ROC Curves — All 5 Models</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        style_axes(ax)
        model_names = list(results_df.index)
        for name, color in zip(model_names, PALETTE['models']):
            fpr, tpr, _ = roc_curve(y_test, probs[name])
            auc = roc_auc_score(y_test, probs[name])
            lw = 2.5 if name in ('XGBoost','LightGBM') else 1.5
            ls = '-' if name in ('XGBoost','LightGBM') else '--'
            ax.plot(fpr, tpr, lw=lw, ls=ls, color=color, label=f'{name} ({auc:.3f})')
        ax.plot([0,1],[0,1],':', color=PALETTE['muted'], lw=1.2, label='Random (0.500)')
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve Comparison'); ax.legend(fontsize=8, loc='lower right', facecolor=PALETTE['bg'], edgecolor=PALETTE['border'])
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.markdown('<div class="section-title"><span>◈</span> 5-Fold CV ROC-AUC with Error Bars</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        style_axes(ax)
        names_cv = list(cv_scores.keys())
        means_cv = [cv_scores[n].mean() for n in names_cv]
        stds_cv  = [cv_scores[n].std() for n in names_cv]
        bars = ax.bar(range(len(names_cv)), means_cv, yerr=stds_cv,
                      color=PALETTE['models'], edgecolor=PALETTE['bg'], capsize=6, linewidth=1.2, alpha=0.9)
        ax.set_ylim(0.70, 0.97); ax.set_ylabel('ROC-AUC')
        ax.set_title('Stratified 5-Fold CV')
        ax.set_xticks(range(len(names_cv)))
        ax.set_xticklabels([n.replace(' ','\n') for n in names_cv], fontsize=7)
        for bar, m, s in zip(bars, means_cv, stds_cv):
            ax.text(bar.get_x()+bar.get_width()/2, m+s+0.004, f'{m:.4f}',
                    ha='center', fontsize=8, fontweight='bold', color=PALETTE['text'])
        st.pyplot(fig, use_container_width=True); plt.close()

    # Confusion matrix + Threshold tuning
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="section-title"><span>◈</span> Confusion Matrix — XGBoost (threshold={best_t})</div>', unsafe_allow_html=True)
        xgb_pred_tuned = (probs['XGBoost'] >= best_t).astype(int)
        cm = confusion_matrix(y_test, xgb_pred_tuned)
        fig, ax = plt.subplots(figsize=(5, 4))
        style_axes(ax)
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn',
                    xticklabels=['Predicted: Stayed','Predicted: Left'],
                    yticklabels=['Actual: Stayed','Actual: Left'],
                    linewidths=2, linecolor=PALETTE['bg'],
                    annot_kws={'size':14,'weight':'bold'}, ax=ax)
        ax.set_title(f'Confusion Matrix (threshold={best_t})', fontsize=10)
        tn, fp, fn, tp = cm.ravel()
        ax.text(0.5, -0.18, f'TN={tn}  FP={fp}  FN={fn}  TP={tp}',
                ha='center', transform=ax.transAxes, fontsize=9, color=PALETTE['muted'])
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.markdown('<div class="section-title"><span>◈</span> Threshold Sensitivity (XGBoost)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        style_axes(ax)
        ax.plot(thresh_df['Threshold'], thresh_df['Precision'], color=PALETTE['blue'], lw=2, label='Precision', marker='o', markersize=4)
        ax.plot(thresh_df['Threshold'], thresh_df['Recall'],    color=PALETTE['red'],  lw=2, label='Recall',    marker='s', markersize=4)
        ax.plot(thresh_df['Threshold'], thresh_df['F1'],        color=PALETTE['green'],lw=2.5, label='F1',      marker='^', markersize=5)
        ax.axvline(best_t, color=PALETTE['purple'], linestyle='--', lw=1.5, label=f'Best t={best_t}')
        ax.set_xlabel('Threshold'); ax.set_ylabel('Score')
        ax.set_title('Precision / Recall / F1 vs Threshold')
        ax.legend(fontsize=8, facecolor=PALETTE['bg'], edgecolor=PALETTE['border'])
        st.pyplot(fig, use_container_width=True); plt.close()


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📈 Feature Importance":
    st.markdown("""
    <div class="page-header">
        <div class="badge">WHAT DRIVES ATTRITION</div>
        <h1>Feature Importance</h1>
        <p>XGBoost and LightGBM agree on the key predictors of employee exit.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    for col, (name, feat) in zip([col1, col2], [('XGBoost', feat_xgb), ('LightGBM', feat_lgbm)]):
        with col:
            st.markdown(f'<div class="section-title"><span>◈</span> Top 12 Features — {name}</div>', unsafe_allow_html=True)
            top12 = feat.nlargest(12).sort_values()
            fig, ax = plt.subplots(figsize=(6, 5))
            style_axes(ax)
            median_v = top12.median()
            colors = [PALETTE['purple'] if v > median_v else PALETTE['blue'] for v in top12.values]
            ax.barh(top12.index, top12.values, color=colors, edgecolor=PALETTE['bg'], height=0.6)
            ax.axvline(median_v, color=PALETTE['muted'], linestyle='--', lw=1, alpha=0.6)
            ax.set_xlabel('Importance Score', fontsize=9)
            ax.set_title(f'{name} Feature Importances', fontsize=10)
            for i, v in enumerate(top12.values):
                ax.text(v + top12.max()*0.01, i, f'{v:.4f}', va='center', fontsize=8, color=PALETTE['text'])
            st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown('<div class="section-title"><span>◈</span> Top 5 Shared Predictors</div>', unsafe_allow_html=True)
    top5_xgb  = set(feat_xgb.nlargest(10).index)
    top5_lgbm = set(feat_lgbm.nlargest(10).index)
    shared    = top5_xgb & top5_lgbm
    cols = st.columns(min(5, len(shared)))
    for col, feat_name in zip(cols, list(shared)[:5]):
        with col:
            xgb_rank  = list(feat_xgb.sort_values(ascending=False).index).index(feat_name) + 1
            lgbm_rank = list(feat_lgbm.sort_values(ascending=False).index).index(feat_name) + 1
            st.markdown(f"""
            <div class="metric-card accent">
              <div class="label">Shared Top Feature</div>
              <div class="value" style="font-size:1rem">{feat_name}</div>
              <div class="sub">XGB #{xgb_rank} · LGBM #{lgbm_rank}</div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT EMPLOYEE
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Predict Employee":
    st.markdown("""
    <div class="page-header">
        <div class="badge">LIVE PREDICTION · XGBOOST</div>
        <h1>Predict Attrition Risk</h1>
        <p>Adjust employee attributes to get an attrition probability score.</p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**Work Conditions**")
        overtime      = st.selectbox("OverTime", ['Yes', 'No'])
        business_travel = st.selectbox("Business Travel", ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
        job_satisfaction = st.slider("Job Satisfaction (1–4)", 1, 4, 3)
        wlb           = st.slider("Work-Life Balance (1–4)", 1, 4, 3)
    with col_b:
        st.markdown("**Employee Profile**")
        age            = st.slider("Age", 18, 60, 32)
        monthly_income = st.slider("Monthly Income ($)", 1000, 20000, 5000, 100)
        years_company  = st.slider("Years at Company", 0, 40, 3)
        years_manager  = st.slider("Years With Manager", 0, 20, 2)
    with col_c:
        st.markdown("**Role Details**")
        job_level      = st.slider("Job Level (1–5)", 1, 5, 2)
        job_role       = st.selectbox("Job Role", sorted(df_eda['JobRole'].unique()))
        department     = st.selectbox("Department", sorted(df_eda['Department'].unique()))
        education      = st.slider("Education (1–5)", 1, 5, 3)

    if st.button("🎯  Predict Attrition Risk", use_container_width=True):
        # Build a row matching the training data
        sample = df_eda.drop('AttritionNum', axis=1).iloc[0:1].copy()
        # Overwrite fields we care about
        sample['Age']                 = age
        sample['MonthlyIncome']       = monthly_income
        sample['YearsAtCompany']      = years_company
        sample['YearsWithCurrManager']= years_manager
        sample['JobLevel']            = job_level
        sample['JobSatisfaction']     = job_satisfaction
        sample['WorkLifeBalance']     = wlb
        sample['Education']           = education
        sample['OverTime']            = overtime
        sample['BusinessTravel']      = business_travel
        sample['JobRole']             = job_role
        sample['Department']          = department
        sample['Attrition']           = 'No'  # placeholder

        # Preprocess same way as training
        cols_to_drop = ['EmployeeNumber','EmployeeCount','Over18','StandardHours']
        sample.drop(columns=[c for c in cols_to_drop if c in sample.columns], inplace=True)
        sample['Attrition'] = sample['Attrition'].map({'Yes':1,'No':0})
        cat_cols = sample.select_dtypes(include='object').columns.tolist()
        sample_enc = pd.get_dummies(sample, columns=cat_cols, drop_first=True)

        # Align columns with training X
        for c in X.columns:
            if c not in sample_enc.columns:
                sample_enc[c] = 0
        sample_enc = sample_enc[X.columns]
        sample_enc.drop(columns=['Attrition'], errors='ignore', inplace=True)
        sample_scaled = scaler.transform(sample_enc)

        xgb_m  = models['XGBoost']
        prob   = xgb_m.predict_proba(sample_scaled)[0][1]
        risk   = "HIGH RISK" if prob >= best_t else "LOW RISK"
        cls    = "pred-high" if prob >= best_t else "pred-low"
        emoji  = "⚠️" if prob >= best_t else "✅"

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"""
            <div class="{cls}">
              <div class="pred-score">{prob*100:.1f}%</div>
              <div class="pred-label">{emoji} {risk}</div>
              <div style="color:#64748b;font-size:0.75rem;margin-top:0.5rem">threshold = {best_t}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("**Key Risk Factors for this profile:**")
            if overtime == 'Yes':
                st.markdown('<div class="insight-card red"><strong>OverTime: Yes</strong> — strongest single predictor of attrition. Burnout risk is elevated.</div>', unsafe_allow_html=True)
            if years_company <= 2:
                st.markdown('<div class="insight-card amber"><strong>Early Tenure (≤2 yrs)</strong> — highest-risk window. Onboarding quality matters here.</div>', unsafe_allow_html=True)
            if monthly_income < 4000:
                st.markdown('<div class="insight-card amber"><strong>Low Monthly Income</strong> — below median for leavers. Financial dissatisfaction is a driver.</div>', unsafe_allow_html=True)
            if wlb == 1:
                st.markdown('<div class="insight-card red"><strong>Poor Work-Life Balance (1)</strong> — 31% attrition rate at this rating.</div>', unsafe_allow_html=True)
            if job_satisfaction <= 2:
                st.markdown('<div class="insight-card amber"><strong>Low Job Satisfaction</strong> — below median. Role engagement may need attention.</div>', unsafe_allow_html=True)
            if prob < best_t:
                st.markdown('<div class="insight-card green"><strong>Profile looks stable</strong> — this employee shares few characteristics with historical leavers.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: HR INSIGHTS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "💡 HR Insights":
    st.markdown("""
    <div class="page-header">
        <div class="badge">BUSINESS RECOMMENDATIONS</div>
        <h1>HR Insights & Actions</h1>
        <p>Translating model findings into concrete retention strategies.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title"><span>◈</span> Top Attrition Drivers</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-card red"><strong>1. OverTime</strong> — Employees working overtime are the single strongest predictor of leaving. Burnout from overwork is a primary driver, independent of salary.</div>
    <div class="insight-card amber"><strong>2. Monthly Income / Job Level</strong> — Lower salary bands and junior job levels show dramatically higher exit rates. Financial dissatisfaction is a core lever, but it interacts with role type.</div>
    <div class="insight-card amber"><strong>3. Early Tenure (0–2 years)</strong> — The first 1–2 years are the most critical attrition window. Employees who survive past year 3 are significantly more likely to stay long-term.</div>
    <div class="insight-card"><strong>4. Work-Life Balance Rating</strong> — Employees rating WLB as "1 (Bad)" show ≈31% attrition vs under 15% for rating 3–4. One of the clearest non-salary drivers.</div>
    <div class="insight-card"><strong>5. Years with Current Manager</strong> — Short tenure under a manager spikes attrition. Manager relationship quality is a key retention lever.</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title"><span>◈</span> Highest-Risk Groups</div>', unsafe_allow_html=True)
    role_data = df_eda.groupby('JobRole')['AttritionNum'].agg(['mean','sum','count'])
    role_data['Rate%'] = (role_data['mean']*100).round(1)
    role_data = role_data.sort_values('Rate%', ascending=False)[['count','sum','Rate%']]
    role_data.columns = ['Total', 'Left', 'Attrition Rate (%)']
    st.dataframe(role_data.head(5), use_container_width=True)

    st.markdown('<div class="section-title"><span>◈</span> Concrete Recommendations</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="insight-card green">
        <strong>✅ Recommendation 1 — Overtime Audit + Compensation Trigger</strong><br><br>
        Flag any employee who has worked overtime for 3+ consecutive months. Offer either a workload review meeting, bonus/compensation adjustment, or flexible scheduling. This targets the <em>single strongest predictor</em> of attrition.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="insight-card green">
        <strong>✅ Recommendation 2 — Early Tenure Check-In Program (0–18 months)</strong><br><br>
        Implement a structured 30-60-90 day check-in system for all new hires, followed by quarterly satisfaction surveys through year 1.5. Pair new hires with a senior mentor. Prioritise Sales Reps and Lab Technicians.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title"><span>◈</span> Model Limitations</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-card amber">
    <strong>⚠️ Important Caveats</strong><br>
    This model reflects historical patterns — it cannot account for future events (recession, restructuring, new competitors). It also cannot explain <em>why</em> a specific employee is at risk, only that they share characteristics with past leavers. HR teams should use predictions as a <strong>starting point for a conversation</strong>, not a final verdict. The 16% class imbalance means even a well-tuned model will occasionally miss high-risk employees. Never make compensation or HR decisions based on the model output alone.
    </div>
    """, unsafe_allow_html=True)
