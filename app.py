# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import pearsonr

st.set_page_config(page_title="ETA & Couverture RT – Analyses", layout="wide", page_icon="📡")

# =============================
# Helpers
# =============================

def _to_datetime(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, dayfirst=True, errors='coerce')
    except Exception:
        return pd.to_datetime(series, errors='coerce')

@st.cache_data(show_spinner=False)
def load_accuracy(file) -> pd.DataFrame:
    if file is None:
        return None
    df = pd.read_csv(file)
    df.columns = [c.strip().replace(' ', '_').replace('-', '_') for c in df.columns]
    # Late harmonisation
    if 'Late' in df.columns and 'late' not in df.columns:
        df = df.rename(columns={'Late':'late'})
    if 'late' not in df.columns:
        df['late'] = np.nan
    # Dates
    if 'startDate' in df.columns:
        df['startDate'] = _to_datetime(df['startDate'])
    if 'endDate' in df.columns:
        df['endDate'] = _to_datetime(df['endDate'])
    # Types
    for col in ['routeID','direction_id','totalPredictions','accurate','early','late']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Percent recalculation
    if {'accurate','totalPredictions'}.issubset(df.columns) and 'accurate_pct' not in df.columns:
        df['accurate_pct'] = df['accurate']/df['totalPredictions']
    if {'early','totalPredictions'}.issubset(df.columns) and 'early_pct' not in df.columns:
        df['early_pct'] = df['early']/df['totalPredictions']
    if {'late','totalPredictions'}.issubset(df.columns) and 'late_pct' not in df.columns:
        df['late_pct'] = df['late']/df['totalPredictions']
    # Friendly categories
    if 'direction_id' in df.columns:
        df['direction'] = df['direction_id'].map({0:'Aller (0)',1:'Retour (1)'}).fillna(df['direction_id'].astype(str))
    if 'timePeriod' in df.columns:
        df['periode'] = df['timePeriod']
    if 'Time_Bucket' in df.columns:
        df['fenetre'] = df['Time_Bucket']
    return df

@st.cache_data(show_spinner=False)
def load_coverage(file) -> pd.DataFrame:
    if file is None:
        return None
    df = pd.read_csv(file)
    df.columns = [c.strip().replace(' ', '_').replace('-', '_') for c in df.columns]
    # Dates
    if 'startDate' in df.columns:
        df['startDate'] = _to_datetime(df['startDate'])
    if 'endDate' in df.columns:
        df['endDate'] = _to_datetime(df['endDate'])
    # Types
    for col in ['route','scheduledTripStops','countTrackedExplained','countOnFullyMissingTrips','countMissingOther']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Fractions recalculées si absentes
    if 'fractionTrackedExplained' not in df.columns and {'countTrackedExplained','scheduledTripStops'}.issubset(df.columns):
        df['fractionTrackedExplained'] = df['countTrackedExplained']/df['scheduledTripStops']
    if 'fractionOnFullyMissingTrips' not in df.columns and {'countOnFullyMissingTrips','scheduledTripStops'}.issubset(df.columns):
        df['fractionOnFullyMissingTrips'] = df['countOnFullyMissingTrips']/df['scheduledTripStops']
    if 'fractionMissingOther' not in df.columns and {'countMissingOther','scheduledTripStops'}.issubset(df.columns):
        df['fractionMissingOther'] = df['countMissingOther']/df['scheduledTripStops']
    # période
    if 'timePeriod' in df.columns:
        df['periode'] = df['timePeriod']
    return df

# UI components ---------------------------------------------------------------

def kpi_cards_accuracy(df_f: pd.DataFrame):
    total = float(df_f['totalPredictions'].sum())
    acc = df_f['accurate'].sum()/total if total>0 else np.nan
    early = df_f['early'].sum()/total if total>0 else np.nan
    late = df_f['late'].sum()/total if total>0 else np.nan
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Prédictions (nb)", f"{int(total):,}".replace(',', ' '))
    c2.metric("Exactitude", f"{acc*100:0.1f}%" if pd.notna(acc) else 'NA')
    c3.metric("En avance", f"{early*100:0.1f}%" if pd.notna(early) else 'NA')
    c4.metric("En retard", f"{late*100:0.1f}%" if pd.notna(late) else 'NA')


def kpi_cards_coverage(df_f: pd.DataFrame):
    total = float(df_f['scheduledTripStops'].sum())
    tracked = df_f['countTrackedExplained'].sum()/total if total>0 else np.nan
    miss_full = df_f['countOnFullyMissingTrips'].sum()/total if total>0 else np.nan
    miss_other = df_f['countMissingOther'].sum()/total if total>0 else np.nan
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Arrêts planifiés (nb)", f"{int(total):,}".replace(',', ' '))
    c2.metric("Suivi expliqué", f"{tracked*100:0.1f}%" if pd.notna(tracked) else 'NA')
    c3.metric("Trajets entièrement manquants", f"{miss_full*100:0.1f}%" if pd.notna(miss_full) else 'NA')
    c4.metric("Autres manquements", f"{miss_other*100:0.1f}%" if pd.notna(miss_other) else 'NA')


def add_wilson_ci(df, success_col, total_col):
    ci_low, ci_high = proportion_confint(df[success_col], df[total_col], alpha=0.05, method='wilson')
    out = df.copy()
    out['ci_low'] = ci_low
    out['ci_high'] = ci_high
    return out


def stacked_parts(df, x, counts=True, parts_cols=None, order=None, labels_map=None):
    g = df.groupby(x, dropna=False).agg({
        'scheduledTripStops':'sum' if 'scheduledTripStops' in df else 'sum',
        'countTrackedExplained': 'sum' if 'countTrackedExplained' in df else 'sum',
        'countOnFullyMissingTrips': 'sum' if 'countOnFullyMissingTrips' in df else 'sum',
        'countMissingOther': 'sum' if 'countMissingOther' in df else 'sum'
    }).reset_index()
    # parts
    for col in ['countTrackedExplained','countOnFullyMissingTrips','countMissingOther']:
        if col in g.columns and 'scheduledTripStops' in g.columns:
            g[col] = g[col]/g['scheduledTripStops']
    melted = g.melt(id_vars=[x], value_vars=['countTrackedExplained','countOnFullyMissingTrips','countMissingOther'], var_name='type', value_name='part')
    mapping = labels_map or {
        'countTrackedExplained':'Suivi expliqué',
        'countOnFullyMissingTrips':'Trajets entièrement manquants',
        'countMissingOther':'Autres manquements'
    }
    melted['type'] = melted['type'].map(mapping)
    if order:
        melted[x] = pd.Categorical(melted[x], categories=order, ordered=True)
    fig = px.bar(melted, x=x, y='part', color='type', barmode='stack',
                 color_discrete_map={'Suivi expliqué':'#2ca02c','Trajets entièrement manquants':'#ff7f0e','Autres manquements':'#d62728'},
                 labels={'part':'Part (%)','type':'Catégorie'})
    fig.update_yaxes(tickformat='.0%', range=[0,1])
    return fig


def coverage_bar(df, group_col):
    g = df.groupby(group_col, dropna=False).agg({
        'countTrackedExplained':'sum', 'scheduledTripStops':'sum'
    }).reset_index()
    g['tracked_ratio'] = g['countTrackedExplained']/g['scheduledTripStops']
    g = add_wilson_ci(g, 'countTrackedExplained','scheduledTripStops')
    chart = alt.Chart(g).mark_bar(color='#2ca02c').encode(
        x=alt.X(f'{group_col}:N', sort='-y'),
        y=alt.Y('tracked_ratio:Q', title='Suivi expliqué', axis=alt.Axis(format='%')),
        tooltip=[group_col, alt.Tooltip('scheduledTripStops:Q','Arrêts planifiés', format=',d'), alt.Tooltip('tracked_ratio:Q','Couverture', format='.1%')]
    )
    err = alt.Chart(g).mark_errorbar().encode(x=alt.X(f'{group_col}:N'), y='ci_low:Q', y2='ci_high:Q')
    return (chart+err).properties(height=380)


def heatmap_cov(df, row='route', col='periode'):
    g = df.groupby([row,col], dropna=False).agg({'countTrackedExplained':'sum','scheduledTripStops':'sum'}).reset_index()
    g['tracked_ratio'] = g['countTrackedExplained']/g['scheduledTripStops']
    pivot = g.pivot(index=row, columns=col, values='tracked_ratio')
    fig = px.imshow(pivot, color_continuous_scale='Greens', aspect='auto', labels=dict(color='Suivi expliqué'))
    fig.update_coloraxes(cmin=0, cmax=1, colorbar_title='Suivi expliqué')
    return fig


def pareto_missing(df, kind='countMissingOther'):
    assert kind in ('countMissingOther','countOnFullyMissingTrips')
    g = df.groupby('route', dropna=False).agg({kind:'sum'}).reset_index().sort_values(kind, ascending=False)
    g['cum'] = g[kind].cumsum()/g[kind].sum()
    colors = {'countMissingOther':'#d62728','countOnFullyMissingTrips':'#ff7f0e'}
    fig = px.bar(g, x='route', y=kind, color_discrete_sequence=[colors[kind]], labels={kind:'Volume', 'route':'Route'})
    line = px.line(g, x='route', y='cum')
    line.update_traces(yaxis='y2', line_color='#1f77b4')
    fig.update_layout(yaxis2=dict(overlaying='y', side='right', tickformat='.0%', range=[0,1]))
    for tr in line.data:
        fig.add_trace(tr)
    fig.update_layout(title=('Pareto – Autres manquements' if kind=='countMissingOther' else 'Pareto – Trajets entièrement manquants'), showlegend=False)
    return fig


def download_df_button(df, filename):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button('⬇️ Télécharger les données filtrées (CSV)', data=csv, file_name=filename, mime='text/csv')

# =============================
# UI
# =============================
st.title("📡 Analyse ETA & Couverture de l'information voyageur en temps réel")
st.caption("Téléversez vos fichiers puis explorez : exactitude des ETA, couverture RT et leur corrélation.")

col_u1, col_u2 = st.columns(2)
with col_u1:
    acc_file = st.file_uploader("Fichier CSV – **ETA accuracy**", type=['csv'], key='acc')
with col_u2:
    cov_file = st.file_uploader("Fichier CSV – **Couverture temps réel**", type=['csv'], key='cov')

# Chargement
acc_df = load_accuracy(acc_file) if acc_file is not None else None
cov_df = load_coverage(cov_file) if cov_file is not None else None

# Tabs globales
tab_acc, tab_cov, tab_corr = st.tabs(["ETA Accuracy", "Couverture RT", "Corrélation ETA ↔ Couverture"])

# ------------------------------------------------------------------ ETA ACC
with tab_acc:
    if acc_df is None:
        st.info("➡️ Téléversez un fichier *ETA accuracy* pour activer cette section.")
    else:
        st.subheader("Filtres (ETA)")
        # Filtres
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1:
            routes = sorted(acc_df['routeID'].dropna().unique().tolist()) if 'routeID' in acc_df else []
            f_routes = st.multiselect("Routes", routes)
        with c2:
            per = sorted(acc_df['periode'].dropna().unique().tolist()) if 'periode' in acc_df else []
            f_per = st.multiselect("Périodes", per)
        with c3:
            fen = sorted(acc_df['Time_Bucket'].dropna().unique().tolist()) if 'Time_Bucket' in acc_df else []
            f_fen = st.multiselect("Fenêtres", fen)
        with c4:
            dirs = sorted(acc_df['direction'].dropna().unique().tolist()) if 'direction' in acc_df else []
            f_dir = st.multiselect("Directions", dirs)
        with c5:
            providers = sorted(acc_df['provider'].dropna().unique().tolist()) if 'provider' in acc_df else []
            f_prov = st.multiselect("Providers", providers)
        mask = pd.Series(True, index=acc_df.index)
        if f_routes: mask &= acc_df['routeID'].isin(f_routes)
        if f_per:    mask &= acc_df['periode'].isin(f_per)
        if f_fen:    mask &= acc_df['Time_Bucket'].isin(f_fen)
        if f_dir:    mask &= acc_df['direction'].isin(f_dir)
        if f_prov:   mask &= acc_df['provider'].isin(f_prov)
        fdf = acc_df[mask].copy()

        # KPI
        kpi_cards_accuracy(fdf)

        # Viz
        st.subheader("Distribution 'Exact / En avance / En retard' par fenêtre")
        order = ["0 - 3 minutes","3 - 6 minutes","6 - 10 minutes","10 - 15 minutes"]
        if 'Time_Bucket' in fdf.columns:
            data = fdf.groupby('Time_Bucket', dropna=False).agg({'accurate':'sum','early':'sum','late':'sum','totalPredictions':'sum'}).reset_index()
            for col in ['accurate','early','late']:
                data[col] = data[col]/data['totalPredictions']
            melted = data.melt(id_vars=['Time_Bucket'], value_vars=['accurate','early','late'], var_name='type', value_name='part')
            labels = {'accurate':'Exact','early':'En avance','late':'En retard'}
            melted['type'] = melted['type'].map(labels)
            if set(order).issubset(set(melted['Time_Bucket'].unique())):
                melted['Time_Bucket'] = pd.Categorical(melted['Time_Bucket'], categories=order, ordered=True)
            fig = px.bar(melted, x='Time_Bucket', y='part', color='type', barmode='stack', color_discrete_map={'Exact':'#2ca02c','En avance':'#1f77b4','En retard':'#d62728'}, labels={'part':'Part (%)','type':'Catégorie'})
            fig.update_yaxes(tickformat='.0%', range=[0,1])
            st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Exactitude par période")
            if 'periode' in fdf.columns:
                st.altair_chart(_acc_bar := (lambda df: (
                    (lambda g: (alt.Chart(g).mark_bar(color='#2ca02c').encode(
                        x=alt.X('periode:N', sort='-y'), y=alt.Y('accuracy:Q', axis=alt.Axis(format='%'), title='Exactitude'),
                        tooltip=['periode', alt.Tooltip('totalPredictions:Q', title='Nb prédictions', format=',d'), alt.Tooltip('accuracy:Q', title='Exactitude', format='.1%')]
                    ) + alt.Chart(g).mark_errorbar().encode(x='periode:N', y='acc_ci_low:Q', y2='acc_ci_high:Q')).properties(height=340))
                    (lambda g: g.assign(acc_ci_low=proportion_confint(g['accurate'], g['totalPredictions'], method='wilson')[0], acc_ci_high=proportion_confint(g['accurate'], g['totalPredictions'], method='wilson')[1]))
                    (fdf.groupby('periode', dropna=False).agg({'accurate':'sum','totalPredictions':'sum'}).reset_index().assign(accuracy=lambda d: d['accurate']/d['totalPredictions']))
                ))(fdf), use_container_width=True)
        with col2:
            st.subheader("Exactitude par direction")
            if 'direction' in fdf.columns:
                g = fdf.groupby('direction', dropna=False).agg({'accurate':'sum','totalPredictions':'sum'}).reset_index()
                g['accuracy'] = g['accurate']/g['totalPredictions']
                ci = proportion_confint(g['accurate'], g['totalPredictions'], method='wilson')
                g['acc_ci_low'], g['acc_ci_high'] = ci
                chart = alt.Chart(g).mark_bar(color='#2ca02c').encode(x=alt.X('direction:N', sort='-y'), y=alt.Y('accuracy:Q', axis=alt.Axis(format='%')))
                err = alt.Chart(g).mark_errorbar().encode(x='direction:N', y='acc_ci_low:Q', y2='acc_ci_high:Q')
                st.altair_chart((chart+err).properties(height=340), use_container_width=True)

        st.subheader("Table filtrée (ETA)")
        st.dataframe(fdf)

# ---------------------------------------------------------------- Couverture
with tab_cov:
    if cov_df is None:
        st.info("➡️ Téléversez un fichier *Couverture temps réel* pour activer cette section.")
    else:
        st.subheader("Filtres (Couverture RT)")
        c1,c2 = st.columns(2)
        with c1:
            routes = sorted(cov_df['route'].dropna().unique().tolist()) if 'route' in cov_df else []
            f_routes = st.multiselect("Routes", routes)
        with c2:
            per = sorted(cov_df['periode'].dropna().unique().tolist()) if 'periode' in cov_df else []
            f_per = st.multiselect("Périodes", per)
        mask = pd.Series(True, index=cov_df.index)
        if f_routes: mask &= cov_df['route'].isin(f_routes)
        if f_per:    mask &= cov_df['periode'].isin(f_per)
        cdf = cov_df[mask].copy()

        # KPI
        kpi_cards_coverage(cdf)

        st.subheader("Couverture par période (barres empilées)")
        order = ["All","Rush AM","Rush PM","Off-Peak"]
        fig = stacked_parts(cdf, x='periode', order=order)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Suivi expliqué par route (IC 95%)")
            st.altair_chart(coverage_bar(cdf, 'route'), use_container_width=True)
        with col2:
            st.subheader("Heatmap Couverture – route × période")
            st.plotly_chart(heatmap_cov(cdf, 'route','periode'), use_container_width=True)

        st.subheader("Pareto des manquements")
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(pareto_missing(cdf, 'countMissingOther'), use_container_width=True)
        with c4:
            st.plotly_chart(pareto_missing(cdf, 'countOnFullyMissingTrips'), use_container_width=True)

        st.subheader("Table filtrée (Couverture)")
        st.dataframe(cdf)

# -------------------------------------------------------------- Corrélation
with tab_corr:
    if (acc_df is None) or (cov_df is None):
        st.info("➡️ Téléversez les **deux** fichiers pour activer la corrélation.")
    else:
        st.subheader("Corrélation par route × période")
        # Agrégations
        a = (acc_df.groupby(['routeID','periode'], dropna=False)
                .agg(accurate=('accurate','sum'), totalPredictions=('totalPredictions','sum'))
                .reset_index())
        a = a[a['totalPredictions']>0].assign(accuracy=lambda d: d['accurate']/d['totalPredictions'])
        c = (cov_df.groupby(['route','periode'], dropna=False)
                .agg(countTrackedExplained=('countTrackedExplained','sum'), scheduledTripStops=('scheduledTripStops','sum'))
                .reset_index())
        c = c[c['scheduledTripStops']>0].assign(tracked_ratio=lambda d: d['countTrackedExplained']/d['scheduledTripStops'])
        # Jointure
        m = a.merge(c, left_on=['routeID','periode'], right_on=['route','periode'], how='inner')
        if len(m)==0:
            st.warning("Aucune clé route × période commune. Vérifiez la cohérence des identifiants (routeID vs route).")
        else:
            st.caption(f"Paires route×période jointes: {len(m):,}".replace(',', ' '))
            fig = px.scatter(m, x='tracked_ratio', y='accuracy', color='periode', size='totalPredictions', hover_data=['routeID'], labels={'tracked_ratio':'Couverture – suivi expliqué', 'accuracy':'Exactitude ETA'})
            fig.update_xaxes(tickformat='.0%'); fig.update_yaxes(tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)
            # Corrélation globale
            if m['tracked_ratio'].nunique()>1 and m['accuracy'].nunique()>1:
                r, p = pearsonr(m['tracked_ratio'], m['accuracy'])
                st.metric("Corrélation de Pearson (non pondérée)", f"r = {r:0.3f} (p={p:0.3g})")
            # Cas notables
            st.subheader("Cas notables")
            # Faible couverture mais bonne exactitude / bonne couverture mais faible exactitude
            low_cov = m.sort_values('tracked_ratio').head(10)[['routeID','periode','tracked_ratio','accuracy','totalPredictions']]
            low_acc = m.sort_values('accuracy').head(10)[['routeID','periode','tracked_ratio','accuracy','totalPredictions']]
            c1,c2 = st.columns(2)
            with c1:
                st.markdown("**Faible couverture, exactitude correcte** (à investiguer)")
                st.dataframe(low_cov)
            with c2:
                st.markdown("**Bonne couverture, exactitude faible** (vérifier algorithmes ETA)")
                st.dataframe(low_acc)

st.divider()
st.caption("© 2025 – Tableau de bord Streamlit – ETA & Couverture RT")
