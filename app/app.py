"""
app.py — TTC Bus Delay Analytics Dashboard
Matches Figma design: TTC Analytics Dashboard (Sleek Light)
Run with: streamlit run app.py  (from app/ folder)
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, date, time

# ── Paths ─────────────────────────────────────────────────────────────────────
APP_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT      = os.path.dirname(APP_DIR)
DATA_PATH = os.path.join(ROOT, "data", "processed", "master_ttc_eda_master.csv")
MODEL_PATH = os.path.join(ROOT, "models", "lgbm_ttc_regressor_bundle.pkl")

# ── Design tokens from Figma ──────────────────────────────────────────────────
RED       = "#b50303"
RED_DARK  = "#93000a"
RED_LIGHT = "#fef2f2"
RED_MID   = "#dc2626"
NAVY      = "#0f172a"
TEXT_PRI  = "#191c1d"
TEXT_SEC  = "#64748b"
TEXT_MUT  = "#94a3b8"
SURFACE   = "#f8f9fa"
WHITE     = "#ffffff"
BORDER    = "#e2e8f0"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TTC Analytics",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS — load from style.css and inject via st.html ───────────────────
_css_path = os.path.join(APP_DIR, "style.css")
with open(_css_path) as _f:
    st.html(f"<style>{_f.read()}</style>")



# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df['Service_Date'] = pd.to_datetime(df['Service_Date'], errors='coerce')
    return df

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">Transit Control</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Editorial Analytics</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        label_visibility="collapsed",
        options=[
            "🏠  Dashboard",
            "🚌  Route Analysis",
            "🌦  Weather Impact",
            "⚠️  Incident Analysis",
            "🔮  Predictive Routing",
            "🔧  Vehicle Intel",
        ],
    )
    st.divider()
    st.markdown(
        '<div style="font-size:11px;color:#94a3b8;font-family:Inter,sans-serif;">'
        'Data: 2015–2025 · 674k records<br>Model: LightGBM v4'
        '</div>',
        unsafe_allow_html=True
    )

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_data()
df_delay = df[df['Min_Delay'] > 0].copy()
bundle = load_model()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD (System Overview)
# ════════════════════════════════════════════════════════════════════════════
if "Dashboard" in page:

    # Header
    col_title, col_badge = st.columns([3, 1])
    with col_title:
        st.markdown('<div class="section-eyebrow">Real-Time Performance</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">System Overview</div>', unsafe_allow_html=True)
    with col_badge:
        st.markdown('<br><br><span class="badge-live">● SYSTEM LIVE</span>', unsafe_allow_html=True)

    # ── KPI Cards ────────────────────────────────────────────────────────────
    total_incidents = len(df_delay)
    avg_delay       = df_delay['Min_Delay'].mean()
    severe_rate     = df_delay['Is_Severe'].mean() * 100
    severe_30_rate  = df_delay['Is_Severe_30'].mean() * 100

    # YoY change
    last_year = df_delay[df_delay['Year'] == df_delay['Year'].max()]
    prev_year = df_delay[df_delay['Year'] == df_delay['Year'].max() - 1]
    yoy_incident_change = ((len(last_year) - len(prev_year)) / len(prev_year) * 100) if len(prev_year) > 0 else 0
    yoy_delay_change    = last_year['Min_Delay'].mean() - prev_year['Min_Delay'].mean()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="ttc-card">
          <div class="kpi-label">Total Incidents (All Years)</div>
          <div class="kpi-value-red">{total_incidents:,}</div>
          <div class="kpi-change-{'up' if yoy_incident_change < 0 else 'down'}">
            {'↓' if yoy_incident_change < 0 else '↑'} {abs(yoy_incident_change):.1f}% vs prior year
          </div>
        </div>""", unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="ttc-card">
          <div class="kpi-label">Avg Delay Duration</div>
          <div style="display:flex;align-items:baseline;gap:4px;">
            <span class="kpi-value-dark">{avg_delay:.0f}</span>
            <span class="kpi-unit">min</span>
          </div>
          <div class="kpi-change-{'down'}">
            ↑ {abs(yoy_delay_change):.1f} min vs prior year
          </div>
        </div>""", unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="ttc-card">
          <div class="kpi-label">Severe Rate (≥15 min)</div>
          <div style="display:flex;align-items:baseline;gap:4px;">
            <span class="kpi-value-dark">{severe_rate:.0f}</span>
            <span class="kpi-unit">%</span>
          </div>
          <div class="kpi-change-down">of all delays are severe</div>
        </div>""", unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="ttc-card">
          <div class="kpi-label">Critical Rate (≥30 min)</div>
          <div style="display:flex;align-items:baseline;gap:4px;">
            <span class="kpi-value-dark">{severe_30_rate:.0f}</span>
            <span class="kpi-unit">%</span>
          </div>
          <div class="kpi-change-down">require urgent response</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── YoY Trend Chart ───────────────────────────────────────────────────────
    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown('<div class="ttc-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Delay Trends — Year over Year</div>', unsafe_allow_html=True)

        yoy = df_delay.groupby('Year').agg(
            Total_Incidents=('Min_Delay', 'count'),
            Avg_Delay=('Min_Delay', 'mean'),
            Severe_Rate=('Is_Severe', 'mean'),
        ).reset_index()

        fig = go.Figure()
        fig.add_bar(
            x=yoy['Year'],
            y=yoy['Total_Incidents'],
            name='Incidents',
            marker_color=[f'rgba(181,3,3,{0.1 + 0.8 * i/len(yoy)})' for i in range(len(yoy))],
        )
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_family='Inter',
            showlegend=False,
            margin=dict(l=0, r=0, t=8, b=0),
            height=280,
            xaxis=dict(showgrid=False, tickfont=dict(color='#94a3b8', size=11)),
            yaxis=dict(showgrid=True, gridcolor='#f1f5f9', tickfont=dict(color='#94a3b8', size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="ttc-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Severity Trend</div>', unsafe_allow_html=True)

        fig2 = go.Figure()
        fig2.add_scatter(
            x=yoy['Year'],
            y=(yoy['Severe_Rate'] * 100).round(1),
            mode='lines+markers',
            line=dict(color=RED, width=2.5),
            marker=dict(color=RED, size=6),
            name='Severe Rate %',
        )
        fig2.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_family='Inter',
            showlegend=False,
            margin=dict(l=0, r=0, t=8, b=0),
            height=280,
            xaxis=dict(showgrid=False, tickfont=dict(color='#94a3b8', size=11)),
            yaxis=dict(
                showgrid=True,
                gridcolor='#f1f5f9',
                tickfont=dict(color='#94a3b8', size=11),
                ticksuffix='%',
            ),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Monthly pattern ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="ttc-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-heading">Monthly Delay Patterns (avg across all years)</div>', unsafe_allow_html=True)

    monthly = df_delay.groupby('Month').agg(
        Avg_Incidents=('Min_Delay', 'count'),
        Avg_Delay=('Min_Delay', 'mean'),
        Severe_Rate=('Is_Severe', 'mean'),
    ).reset_index()
    monthly['Month_Name'] = monthly['Month'].map({
        1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
        7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'
    })

    fig3 = go.Figure()
    fig3.add_bar(
        x=monthly['Month_Name'],
        y=monthly['Avg_Delay'].round(1),
        name='Avg Delay (min)',
        marker_color=RED,
        opacity=0.85,
    )
    fig3.add_scatter(
        x=monthly['Month_Name'],
        y=(monthly['Severe_Rate'] * 100).round(1),
        name='Severe Rate %',
        mode='lines+markers',
        line=dict(color=NAVY, width=2),
        marker=dict(size=5),
        yaxis='y2',
    )
    fig3.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_family='Inter',
        height=280,
        margin=dict(l=0, r=0, t=8, b=0),
        legend=dict(orientation='h', y=1.1, x=0, font=dict(size=11, color=TEXT_SEC)),
        xaxis=dict(showgrid=False, tickfont=dict(color='#94a3b8', size=11)),
        yaxis=dict(showgrid=True, gridcolor='#f1f5f9', tickfont=dict(color='#94a3b8', size=11), title='Avg Delay (min)'),
        yaxis2=dict(overlaying='y', side='right', tickfont=dict(color='#94a3b8', size=11), ticksuffix='%', showgrid=False),
        barmode='overlay',
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ROUTE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif "Route Analysis" in page:

    st.markdown('<div class="section-eyebrow">Operational Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Route Performance</div>', unsafe_allow_html=True)

    # Top route summary
    route_summary = (
        df_delay.groupby('Route_Number')
        .agg(
            Total_Incidents=('Min_Delay', 'count'),
            Avg_Delay=('Min_Delay', 'mean'),
            Median_Delay=('Min_Delay', 'median'),
            Severe_Rate=('Is_Severe', 'mean'),
            Severe_30_Rate=('Is_Severe_30', 'mean'),
            Avg_Headway=('Headway_min', 'mean'),
        )
        .round(2)
        .query('Total_Incidents >= 100')
        .reset_index()
    )
    route_summary['Risk_Score'] = (
        route_summary['Avg_Delay'] *
        route_summary['Avg_Headway'] *
        route_summary['Severe_Rate']
    ).round(2)

    c1, c2 = st.columns([1, 2])

    with c1:
        # Featured worst route
        worst = route_summary.sort_values('Total_Incidents', ascending=False).iloc[0]
        st.markdown(f"""
        <div class="ttc-card-surface">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;">
            <div style="background:white;border-radius:12px;padding:12px;box-shadow:0 1px 2px rgba(0,0,0,0.05);">
              🚌
            </div>
            <span class="badge-critical">CRITICAL</span>
          </div>
          <div style="font-family:'Public Sans',sans-serif;font-weight:700;font-size:20px;color:#191c1d;margin-bottom:4px;">
            Route {int(worst['Route_Number'])}
          </div>
          <div style="font-family:Inter,sans-serif;font-size:14px;color:#64748b;margin-bottom:20px;">
            Highest incident frequency. Avg delay: {worst['Avg_Delay']:.1f} min
          </div>
          <div style="border-top:1px solid rgba(226,232,240,0.5);padding-top:16px;display:flex;justify-content:space-between;">
            <div style="font-family:Inter,sans-serif;font-weight:700;font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">
              Total incidents
            </div>
            <div style="font-family:Inter,sans-serif;font-weight:500;font-size:12px;color:#0f172a;">
              {int(worst['Total_Incidents']):,} recorded
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="ttc-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Route Efficiency Leaderboard — Top 15 by Incidents</div>', unsafe_allow_html=True)

        top15 = route_summary.sort_values('Total_Incidents', ascending=False).head(15)
        max_delay = top15['Avg_Delay'].max()

        for i, (_, row) in enumerate(top15.iterrows(), 1):
            bar_pct = int(row['Avg_Delay'] / max_delay * 100)
            bar_color = RED if i == 1 else (RED_MID if i <= 3 else "#f87171")
            st.markdown(f"""
            <div class="lb-row">
              <div style="display:flex;align-items:center;gap:16px;flex:1;">
                <span class="lb-rank">{i:02d}</span>
                <div>
                  <div class="lb-route">Route {int(row['Route_Number'])}</div>
                  <div class="lb-sub">{int(row['Total_Incidents']):,} incidents</div>
                </div>
              </div>
              <div style="display:flex;align-items:center;gap:24px;">
                <div style="text-align:right;">
                  <div class="lb-delay">{row['Avg_Delay']:.1f}m</div>
                  <div class="lb-sub">avg delay</div>
                </div>
                <div style="background:#f1f5f9;border-radius:9999px;width:120px;height:6px;overflow:hidden;">
                  <div style="background:{bar_color};width:{bar_pct}%;height:100%;border-radius:9999px;"></div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Route Risk Profile ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="ttc-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-heading">Route Risk Profile — High Delay × Low Frequency</div>', unsafe_allow_html=True)
    st.caption("Risk Score = Avg Delay × Headway × Severe Rate. Higher = more operationally risky for passengers.")

    top_risk = route_summary.sort_values('Risk_Score', ascending=False).head(20)
    fig_risk = px.scatter(
        top_risk,
        x='Avg_Headway',
        y='Avg_Delay',
        size='Total_Incidents',
        color='Severe_Rate',
        color_continuous_scale=[[0, '#fef2f2'], [0.5, '#dc2626'], [1, '#7f1d1d']],
        hover_name='Route_Number',
        hover_data={'Total_Incidents': True, 'Risk_Score': True},
        labels={
            'Avg_Headway': 'Avg Headway (min between buses)',
            'Avg_Delay': 'Avg Delay (min)',
            'Severe_Rate': 'Severe Rate',
        },
        size_max=40,
    )
    fig_risk.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_family='Inter',
        height=380,
        margin=dict(l=0, r=0, t=8, b=0),
        coloraxis_colorbar=dict(title='Severe Rate', tickformat='.0%'),
    )
    fig_risk.update_traces(marker=dict(line=dict(width=1, color='white')))
    st.plotly_chart(fig_risk, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — WEATHER IMPACT
# ════════════════════════════════════════════════════════════════════════════
elif "Weather Impact" in page:

    st.markdown('<div class="section-eyebrow">Environmental Correlation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Weather Impact</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1])

    with c1:
        st.markdown('<div class="ttc-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Temperature vs Avg Delay Duration</div>', unsafe_allow_html=True)

        df_w = df_delay.dropna(subset=['Temp_C']).copy()
        df_w['Temp_Band'] = pd.cut(
            df_w['Temp_C'],
            bins=[-35, -15, 0, 5, 15, 25, 45],
            labels=['Extreme Cold\n(<-15°C)', 'Cold\n(-15–0°C)', 'Cool\n(0–5°C)',
                    'Mild\n(5–15°C)', 'Warm\n(15–25°C)', 'Hot\n(25°C+)']
        )
        temp_agg = df_w.groupby('Temp_Band', observed=True).agg(
            Avg_Delay=('Min_Delay', 'mean'),
            Severe_Rate=('Is_Severe', 'mean'),
            Count=('Min_Delay', 'count'),
        ).reset_index()

        fig_temp = go.Figure()
        fig_temp.add_bar(
            x=temp_agg['Temp_Band'].astype(str),
            y=temp_agg['Avg_Delay'].round(1),
            marker_color=[RED if v > temp_agg['Avg_Delay'].mean() else '#fca5a5' for v in temp_agg['Avg_Delay']],
            name='Avg Delay',
        )
        fig_temp.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            font_family='Inter', height=300,
            margin=dict(l=0, r=0, t=8, b=0),
            showlegend=False,
            xaxis=dict(showgrid=False, tickfont=dict(color='#94a3b8', size=11)),
            yaxis=dict(showgrid=True, gridcolor='#f1f5f9', tickfont=dict(color='#94a3b8', size=11), title='Avg Delay (min)'),
        )
        st.plotly_chart(fig_temp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="ttc-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-family:\'Public Sans\',sans-serif;font-weight:700;font-size:14px;color:#191c1d;margin-bottom:16px;">Visibility Impact</div>', unsafe_allow_html=True)

        df_v = df_delay.dropna(subset=['Visibility_km']).copy()
        df_v['Vis_Band'] = pd.cut(
            df_v['Visibility_km'],
            bins=[0, 1, 5, 15, 100],
            labels=['<1km', '1–5km', '5–15km', '15km+']
        )
        vis_agg = df_v.groupby('Vis_Band', observed=True)['Min_Delay'].mean().reset_index()

        for _, row in vis_agg.iterrows():
            pct = int(row['Min_Delay'] / vis_agg['Min_Delay'].max() * 100)
            color = RED if row['Min_Delay'] > vis_agg['Min_Delay'].mean() else '#fca5a5'
            st.markdown(f"""
            <div style="margin-bottom:16px;">
              <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <span style="font-family:Inter,sans-serif;font-size:12px;color:#64748b;">{row['Vis_Band']}</span>
                <span style="font-family:Inter,sans-serif;font-weight:700;font-size:12px;color:#0f172a;">{row['Min_Delay']:.1f} min</span>
              </div>
              <div style="background:#f1f5f9;border-radius:9999px;height:6px;overflow:hidden;">
                <div style="background:{color};width:{pct}%;height:100%;border-radius:9999px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Seasonal ──────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<div class="ttc-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Seasonal Delay Patterns</div>', unsafe_allow_html=True)
        seasonal = df_delay.groupby('Season').agg(
            Avg_Delay=('Min_Delay', 'mean'),
            Severe_Rate=('Is_Severe', 'mean'),
        ).reset_index()
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']
        seasonal['Season'] = pd.Categorical(seasonal['Season'], categories=season_order, ordered=True)
        seasonal = seasonal.sort_values('Season')

        fig_s = px.bar(
            seasonal, x='Season', y='Avg_Delay',
            color='Avg_Delay',
            color_continuous_scale=[[0, '#fef2f2'], [1, RED]],
            labels={'Avg_Delay': 'Avg Delay (min)'},
        )
        fig_s.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            font_family='Inter', height=260,
            margin=dict(l=0, r=0, t=8, b=0),
            showlegend=False,
            coloraxis_showscale=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
        )
        st.plotly_chart(fig_s, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="ttc-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Wind Speed vs Severe Rate</div>', unsafe_allow_html=True)
        df_w2 = df_delay.dropna(subset=['Wind_Spd_kmh']).copy()
        df_w2['Wind_Band'] = pd.cut(df_w2['Wind_Spd_kmh'], bins=[0,10,20,30,50,150],
                                     labels=['0–10','10–20','20–30','30–50','50+'])
        wind_agg = df_w2.groupby('Wind_Band', observed=True).agg(
            Severe_Rate=('Is_Severe', 'mean'),
            Count=('Min_Delay', 'count'),
        ).reset_index()
        fig_w = px.line(
            wind_agg, x='Wind_Band', y=(wind_agg['Severe_Rate']*100).round(1),
            markers=True,
            labels={'y': 'Severe Rate %', 'Wind_Band': 'Wind Speed (km/h)'},
            color_discrete_sequence=[RED],
        )
        fig_w.update_traces(line=dict(width=2.5), marker=dict(size=7))
        fig_w.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            font_family='Inter', height=260,
            margin=dict(l=0, r=0, t=8, b=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#f1f5f9', ticksuffix='%'),
        )
        st.plotly_chart(fig_w, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — INCIDENT ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif "Incident Analysis" in page:

    st.markdown('<div class="section-eyebrow">Situational Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Incident Analysis</div>', unsafe_allow_html=True)

    incident_agg = (
        df_delay.dropna(subset=['Incident_Category'])
        .groupby('Incident_Category')
        .agg(
            Total=('Min_Delay', 'count'),
            Avg_Delay=('Min_Delay', 'mean'),
            Severe_Rate=('Is_Severe', 'mean'),
            Total_Hrs=('Min_Delay', lambda x: x.sum()/60),
        )
        .round(2)
        .sort_values('Total', ascending=False)
        .reset_index()
    )
    incident_agg['Share'] = (incident_agg['Total'] / incident_agg['Total'].sum() * 100).round(1)

    c1, c2 = st.columns(2)

    with c1:
        # Critical callout — top incident type
        top_inc = incident_agg.iloc[0]
        mech_rows = incident_agg[incident_agg['Incident_Category'] == 'Mechanical']
        mech_share = mech_rows['Share'].values[0] if len(mech_rows) > 0 else 0

        st.markdown(f"""
        <div class="callout-critical">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
            <span style="font-size:24px;">⚠️</span>
            <div class="callout-title">Top Incident Type:<br>{top_inc['Incident_Category']}</div>
          </div>
          <div class="callout-body">
            <b>{top_inc['Incident_Category']}</b> accounts for <b>{mech_share:.0f}%</b> of all delay incidents —
            {int(top_inc['Total']):,} total incidents producing {int(top_inc['Total_Hrs']):,} hours of cumulative delay.
            Average delay duration: <b>{top_inc['Avg_Delay']:.1f} minutes</b>.
            Severe rate: <b>{top_inc['Severe_Rate']*100:.1f}%</b>.
          </div>
          <br>
          <div style="font-family:Inter,sans-serif;font-size:13px;color:rgba(147,0,10,0.7);">
            ⚡ Diversion incidents (6.5% of volume) account for <b>27.8%</b> of all delay hours
            — the highest impact-to-frequency ratio of any category.
          </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="ttc-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Root Cause Distribution</div>', unsafe_allow_html=True)

        # Treemap
        fig_tree = px.treemap(
            incident_agg,
            path=['Incident_Category'],
            values='Total',
            color='Avg_Delay',
            color_continuous_scale=[[0,'#fef2f2'],[0.3,'#fca5a5'],[0.7,RED],[1,RED_DARK]],
            custom_data=['Avg_Delay', 'Share', 'Severe_Rate'],
        )
        fig_tree.update_traces(
            texttemplate='<b>%{label}</b><br>%{customdata[1]:.1f}%',
            hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Avg Delay: %{customdata[0]:.1f} min<br>Severe Rate: %{customdata[2]:.1%}<extra></extra>',
            textfont=dict(family='Inter', size=12),
        )
        fig_tree.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=320,
            coloraxis_colorbar=dict(title='Avg Delay'),
        )
        st.plotly_chart(fig_tree, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Incident breakdown table ───────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="ttc-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-heading">Incident Category Breakdown</div>', unsafe_allow_html=True)

    display_df = incident_agg.rename(columns={
        'Incident_Category': 'Category',
        'Total': 'Incidents',
        'Avg_Delay': 'Avg Delay (min)',
        'Severe_Rate': 'Severe Rate',
        'Total_Hrs': 'Total Hours',
        'Share': 'Share %',
    })
    display_df['Severe Rate'] = (display_df['Severe Rate'] * 100).round(1).astype(str) + '%'
    display_df['Total Hours'] = display_df['Total Hours'].round(0).astype(int)
    st.dataframe(
        display_df[['Category', 'Incidents', 'Avg Delay (min)', 'Severe Rate', 'Total Hours', 'Share %']],
        use_container_width=True,
        hide_index=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PREDICTIVE ROUTING (ML Model)
# ════════════════════════════════════════════════════════════════════════════
elif "Predictive Routing" in page:

    st.markdown('<div class="section-eyebrow">Future Operations</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Predictive Routing</div>', unsafe_allow_html=True)

    if bundle is None:
        st.error(f"Model not found at `{MODEL_PATH}`. Run `train_model_v4.py` first.")
        st.stop()

    clf             = bundle["model"]
    route_encoding  = bundle["route_encoding"]
    global_avg      = bundle["global_avg"]
    incident_avg    = bundle["incident_avg"]
    div_route       = bundle["div_route"]
    global_div      = bundle["global_div"]
    feature_columns = bundle["feature_columns"]

    # GTFS headway lookup
    headway_path = os.path.join(ROOT, "data", "processed", "gtfs_headway_lookup.csv")
    headway_lookup = pd.read_csv(headway_path) if os.path.exists(headway_path) else None

    def get_season(month):
        if month in [12,1,2]: return "Winter"
        elif month in [3,4,5]: return "Spring"
        elif month in [6,7,8]: return "Summer"
        return "Fall"

    def get_headway(route, hour):
        if headway_lookup is None: return 3.9
        match = headway_lookup[(headway_lookup['Route_Number']==route) & (headway_lookup['Hour']==hour)]
        return float(match['Headway_min'].values[0]) if len(match) > 0 else 3.9

    c_input, c_result = st.columns([1, 2])

    with c_input:
        st.markdown('<div class="ttc-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Simulation Engine</div>', unsafe_allow_html=True)

        route_num   = st.number_input("Route Number", min_value=1, max_value=999, value=29, step=1)
        sel_date    = st.date_input("Date", value=date.today())
        sel_time    = st.time_input("Time", value=time(8, 0), step=1800)
        incident_type = st.selectbox("Incident Type", sorted(incident_avg.keys()))

        st.markdown("**Weather Conditions**")
        temp       = st.slider("Temperature (°C)", -30, 40, 5)
        visibility = st.slider("Visibility (km)", 0, 50, 15)
        wind       = st.slider("Wind Speed (km/h)", 0, 100, 20)
        humidity   = st.slider("Relative Humidity (%)", 0, 100, 70)

        predict_btn = st.button("RUN PREDICTION")
        st.markdown('</div>', unsafe_allow_html=True)

    with c_result:
        st.markdown('<div class="ttc-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Prediction Analysis</div>', unsafe_allow_html=True)

        if predict_btn:
            dt      = datetime.combine(sel_date, sel_time)
            hour    = dt.hour
            month   = dt.month
            season  = get_season(month)
            day     = dt.strftime("%A")
            rush    = 1 if hour in [7,8,9,16,17,18] else 0
            weekend = 1 if day in ["Saturday","Sunday"] else 0
            headway = get_headway(float(route_num), hour)

            route_avg_val    = route_encoding.get(route_num, global_avg)
            inc_avg_val      = incident_avg.get(incident_type, global_avg)
            div_route_val    = div_route.get(route_num, global_div)
            inc_route_inter  = inc_avg_val * route_avg_val
            headway_x_route  = headway * route_avg_val
            rush_x_route     = rush * route_avg_val
            hour_x_month     = hour * month

            row = {col: 0 for col in feature_columns}
            row['Hour']                    = hour
            row['Month']                   = month
            row['Is_Rush_Hour']            = rush
            row['Is_Weekend']              = weekend
            row['Temp_C']                  = temp
            row['Visibility_km']           = visibility
            row['Wind_Spd_kmh']            = wind
            row['Rel_Humidity_pct']        = humidity
            row['Headway_min']             = headway
            row['Route_Delay_Avg']         = route_avg_val
            row['Incident_Delay_Avg']      = inc_avg_val
            row['Diversion_Route_Avg']     = div_route_val
            row['Incident_Route_Interaction'] = inc_route_inter
            row['Headway_x_Route_Avg']     = headway_x_route
            row['Rush_x_Route_Avg']        = rush_x_route
            row['Hour_x_Month']            = hour_x_month

            for col in [f'Day_{day}', f'Season_{season}', f'Incident_Category_{incident_type}']:
                if col in row:
                    row[col] = 1

            X    = np.array([row[col] for col in feature_columns]).reshape(1, -1)
            pred = np.expm1(clf.predict(X)[0])
            pred = float(np.clip(pred, 1, 300))

            # Severity tier
            if pred < 15:
                tier, tier_color, tier_emoji = "Minor", "#16a34a", "✅"
            elif pred < 30:
                tier, tier_color, tier_emoji = "Moderate", "#d97706", "⚠️"
            else:
                tier, tier_color, tier_emoji = "Severe", RED, "🔴"

            # Historical context for this route
            route_hist = df_delay[df_delay['Route_Number'] == float(route_num)]
            route_hist_avg = route_hist['Min_Delay'].mean() if len(route_hist) > 0 else global_avg

            c_r1, c_r2 = st.columns(2)
            with c_r1:
                st.markdown(f"""
                <div style="background:#f8f9fa;border-radius:12px;padding:16px;margin-bottom:12px;">
                  <div style="font-family:Inter,sans-serif;font-size:14px;color:#475569;margin-bottom:4px;">Predicted Delay</div>
                  <div style="font-family:'Public Sans',sans-serif;font-weight:800;font-size:32px;color:{tier_color};">
                    {pred:.0f} min
                  </div>
                </div>
                <div style="background:#f8f9fa;border-radius:12px;padding:16px;margin-bottom:12px;">
                  <div style="font-family:Inter,sans-serif;font-size:14px;color:#475569;margin-bottom:4px;">Severity Tier</div>
                  <div style="font-family:'Public Sans',sans-serif;font-weight:700;font-size:20px;color:{tier_color};">
                    {tier_emoji} {tier}
                  </div>
                </div>
                <div style="background:#f8f9fa;border-radius:12px;padding:16px;">
                  <div style="font-family:Inter,sans-serif;font-size:14px;color:#475569;margin-bottom:4px;">Route {route_num} Historical Avg</div>
                  <div style="font-family:'Public Sans',sans-serif;font-weight:700;font-size:20px;color:#0f172a;">
                    {route_hist_avg:.1f} min
                  </div>
                </div>""", unsafe_allow_html=True)

            with c_r2:
                # Scenario summary gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=min(pred, 60),
                    number={'suffix': ' min', 'font': {'family': 'Public Sans', 'size': 28, 'color': tier_color}},
                    gauge={
                        'axis': {'range': [0, 60], 'tickfont': {'size': 10, 'color': '#94a3b8'}},
                        'bar': {'color': tier_color},
                        'bgcolor': '#f8f9fa',
                        'steps': [
                            {'range': [0, 15], 'color': '#dcfce7'},
                            {'range': [15, 30], 'color': '#fef3c7'},
                            {'range': [30, 60], 'color': '#fef2f2'},
                        ],
                        'threshold': {'line': {'color': RED, 'width': 2}, 'thickness': 0.75, 'value': pred},
                    }
                ))
                fig_gauge.update_layout(
                    height=220,
                    margin=dict(l=10, r=10, t=10, b=0),
                    paper_bgcolor='white',
                    font_family='Inter',
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            # Recommended strategy
            is_rush_str = "rush hour" if rush else "off-peak"
            st.markdown(f"""
            <div class="pred-result">
              <div class="pred-label">Recommended Strategy</div>
              <div style="font-family:Inter,sans-serif;font-size:14px;color:#475569;line-height:1.6;">
                Route <b style="color:#0f172a">{route_num}</b> is predicted to run
                <b style="color:{tier_color}">{pred:.0f} minutes</b> behind schedule
                ({tier.lower()} delay) during <b style="color:#0f172a">{is_rush_str}</b>
                on a {day} in {season}.
                {'Consider deploying a replacement vehicle or notifying passengers via the TTC app.' if pred >= 15 else 'No immediate action required — monitor for deteriorating conditions.'}
                Incident type: <b style="color:#0f172a">{incident_type}</b>.
                Current headway on this route: <b style="color:#0f172a">{headway:.1f} min</b>.
              </div>
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div style="background:#f8fafc;border:2px dashed #e2e8f0;border-radius:16px;
                        padding:60px 40px;text-align:center;">
              <div style="font-size:40px;margin-bottom:16px;">🔮</div>
              <div style="font-family:Inter,sans-serif;font-weight:700;font-size:12px;
                          letter-spacing:1px;text-transform:uppercase;color:#94a3b8;">
                Configure parameters and run prediction
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 — VEHICLE INTEL
# ════════════════════════════════════════════════════════════════════════════
elif "Vehicle Intel" in page:

    st.markdown('<div class="section-eyebrow">Fleet Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Vehicle Intel</div>', unsafe_allow_html=True)

    mech_df = df_delay[
        (df_delay['Incident_Category'] == 'Mechanical') &
        (df_delay['Vehicle'].notna()) &
        (df_delay['Vehicle'].astype(str) != '0.0')
    ].copy()

    vehicle_intel = (
        mech_df.groupby('Vehicle')
        .agg(
            Mechanical_Incidents=('Min_Delay', 'count'),
            Avg_Delay=('Min_Delay', 'mean'),
            Total_Delay_Hrs=('Min_Delay', lambda x: x.sum()/60),
            Severe_Rate=('Is_Severe', 'mean'),
        )
        .round(2)
        .sort_values('Mechanical_Incidents', ascending=False)
        .reset_index()
        .head(30)
    )

    # ── KPIs ──────────────────────────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"""<div class="ttc-card">
          <div class="kpi-label">Vehicles Tracked</div>
          <div class="kpi-value-dark">{mech_df['Vehicle'].nunique():,}</div>
          <div class="kpi-change-down">with mechanical incidents</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        worst_v = vehicle_intel.iloc[0]
        st.markdown(f"""<div class="ttc-card">
          <div class="kpi-label">Worst Vehicle (most incidents)</div>
          <div class="kpi-value-red">{int(worst_v['Vehicle'])}</div>
          <div class="kpi-change-down">{int(worst_v['Mechanical_Incidents'])} incidents · {worst_v['Avg_Delay']:.1f} min avg</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        high_severe = vehicle_intel[vehicle_intel['Severe_Rate'] >= 0.35]
        st.markdown(f"""<div class="ttc-card">
          <div class="kpi-label">Replacement Candidates (≥35% severe)</div>
          <div class="kpi-value-red">{len(high_severe)}</div>
          <div class="kpi-change-down">vehicles flagged for review</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown('<div class="ttc-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Top 30 Vehicles by Mechanical Incidents</div>', unsafe_allow_html=True)

        fig_v = px.bar(
            vehicle_intel.sort_values('Mechanical_Incidents'),
            x='Mechanical_Incidents',
            y=vehicle_intel.sort_values('Mechanical_Incidents')['Vehicle'].astype(int).astype(str),
            orientation='h',
            color='Severe_Rate',
            color_continuous_scale=[[0,'#fef2f2'],[0.3,'#fca5a5'],[0.7,RED],[1,RED_DARK]],
            custom_data=['Avg_Delay', 'Total_Delay_Hrs'],
            labels={'Mechanical_Incidents': 'Mechanical Incidents', 'y': 'Vehicle ID'},
        )
        fig_v.update_layout(
            plot_bgcolor='white', paper_bgcolor='white',
            font_family='Inter', height=600,
            margin=dict(l=0, r=0, t=8, b=0),
            coloraxis_colorbar=dict(title='Severe Rate', tickformat='.0%'),
            xaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
            yaxis=dict(showgrid=False, tickfont=dict(size=10)),
        )
        fig_v.update_traces(
            hovertemplate='<b>Vehicle %{y}</b><br>Incidents: %{x}<br>Avg Delay: %{customdata[0]:.1f} min<br>Total Hours: %{customdata[1]:.0f}<extra></extra>'
        )
        st.plotly_chart(fig_v, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="ttc-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">Fleet Replacement Priority</div>', unsafe_allow_html=True)
        st.caption("Ranked by Severe Rate × Incident Count — highest operational risk.")

        priority = vehicle_intel.sort_values(
            by=['Severe_Rate', 'Mechanical_Incidents'], ascending=False
        ).head(15)

        for i, (_, row) in enumerate(priority.iterrows(), 1):
            sev_pct = int(row['Severe_Rate'] * 100)
            color   = RED if sev_pct >= 35 else ("#d97706" if sev_pct >= 20 else "#94a3b8")
            st.markdown(f"""
            <div class="lb-row">
              <div style="display:flex;align-items:center;gap:12px;flex:1;">
                <span class="lb-rank">{i:02d}</span>
                <div>
                  <div class="lb-route">Bus {int(row['Vehicle'])}</div>
                  <div class="lb-sub">{int(row['Mechanical_Incidents'])} incidents</div>
                </div>
              </div>
              <div style="text-align:right;">
                <div style="font-family:Inter,sans-serif;font-weight:700;font-size:14px;color:{color};">
                  {sev_pct}% severe
                </div>
                <div class="lb-sub">{row['Avg_Delay']:.1f} min avg</div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)