"""
deploy/app.py — TTC Bus Delay Analytics Dashboard (Deployment Version)
Reads pre-aggregated CSVs from deploy/data/
Run with: python -m streamlit run app.py
"""
import os
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime, date, time

# ── Paths ─────────────────────────────────────────────────────────────────────
APP_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(APP_DIR, "data")
MODEL_PATH = os.path.join(APP_DIR, "models", "lgbm_ttc_regressor_bundle.pkl")

# ── Design tokens ──────────────────────────────────────────────────────────────
RED      = "#b50303"
RED_DARK = "#93000a"
RED_MID  = "#dc2626"
NAVY     = "#0f172a"

POWERBI_URL = "https://app.powerbi.com/view?r=eyJrIjoiOTYyMmU5MGMtYjdkNC00Nzk3LTlkYzUtMGYwZTg5YjAxZGM5IiwidCI6ImE2YTU1ODM4LTk1ZWEtNGQ2Zi1iMTc4LWRmOTljZGRiODc4NiJ9"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TTC Analytics",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS ────────────────────────────────────────────────────────────────
_css_path = os.path.join(APP_DIR, "style.css")
with open(_css_path) as _f:
    st.html(f"<style>{_f.read()}</style>")

# ── Load pre-aggregated data ──────────────────────────────────────────────────
@st.cache_data
def load_csv(name):
    return pd.read_csv(os.path.join(DATA_DIR, name))

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

kpi_df     = load_csv("kpi.csv")
kpi        = kpi_df.iloc[0]
yoy        = load_csv("yoy.csv")
monthly    = load_csv("monthly.csv")
route_sum  = load_csv("route_summary.csv")
inc        = load_csv("incident_summary.csv")
vi         = load_csv("vehicle_intel.csv")
temp_agg   = load_csv("weather_temp.csv")
vis_agg    = load_csv("weather_vis.csv")
wind_agg   = load_csv("weather_wind.csv")
seasonal   = load_csv("seasonal.csv")
bundle     = load_model()

hw_path = os.path.join(DATA_DIR, "gtfs_headway_lookup.csv")
hw_df   = pd.read_csv(hw_path) if os.path.exists(hw_path) else None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-brand">Transit Control</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-sub">Editorial Analytics</p>', unsafe_allow_html=True)
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
            "📊  Power BI Report",
        ],
    )
    st.divider()
    st.caption("Data: 2015–2025 · 644k incidents\nModel: LightGBM v4")

# ── Helpers ───────────────────────────────────────────────────────────────────
def section_header(eyebrow, heading):
    st.markdown(f'<p class="section-eyebrow">{eyebrow}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="section-heading">{heading}</p>', unsafe_allow_html=True)

def card_title(text):
    st.markdown(f'<p class="card-heading">{text}</p>', unsafe_allow_html=True)

def plotly_defaults(fig, height=300):
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
        font_family="Inter", margin=dict(l=0,r=0,t=8,b=0), height=height)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if "Dashboard" in page:
    col_h, col_b = st.columns([3, 1])
    with col_h:
        section_header("Real-Time Performance", "System Overview")
    with col_b:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<span class="badge-live">● SYSTEM LIVE</span>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="ttc-card">
          <div class="kpi-label">Total Incidents</div>
          <div class="kpi-value-red">{int(kpi['total_incidents']):,}</div>
          <div class="kpi-change-{'up' if kpi['yoy_incident_pct'] < 0 else 'down'}">
            {'↓' if kpi['yoy_incident_pct'] < 0 else '↑'} {abs(kpi['yoy_incident_pct']):.1f}% vs prior year
          </div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="ttc-card">
          <div class="kpi-label">Avg Delay Duration</div>
          <span class="kpi-value-dark">{kpi['avg_delay']:.0f}</span>
          <span class="kpi-unit"> min</span>
          <div class="kpi-change-down">↑ {abs(kpi['yoy_delay_min']):.1f} min vs prior year</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="ttc-card">
          <div class="kpi-label">Severe Rate (≥15 min)</div>
          <span class="kpi-value-dark">{kpi['severe_rate']:.0f}</span>
          <span class="kpi-unit"> %</span>
          <div class="kpi-change-down">of all delays are severe</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="ttc-card">
          <div class="kpi-label">Critical Rate (≥30 min)</div>
          <span class="kpi-value-dark">{kpi['severe_30_rate']:.0f}</span>
          <span class="kpi-unit"> %</span>
          <div class="kpi-change-down">require urgent response</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    with c1:
        with st.container(border=True):
            card_title("Delay Trends — Year over Year")
            fig = go.Figure()
            fig.add_bar(x=yoy["Year"], y=yoy["Incidents"],
                marker_color=[f"rgba(181,3,3,{0.15+0.75*i/len(yoy)})" for i in range(len(yoy))])
            fig = plotly_defaults(fig, 270)
            fig.update_layout(showlegend=False,
                xaxis=dict(showgrid=False, tickfont=dict(color="#94a3b8",size=11)),
                yaxis=dict(showgrid=True, gridcolor="#f1f5f9", tickfont=dict(color="#94a3b8",size=11)))
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        with st.container(border=True):
            card_title("Severity Trend")
            fig2 = go.Figure()
            fig2.add_scatter(x=yoy["Year"], y=(yoy["Severe_Rate"]*100).round(1),
                mode="lines+markers", line=dict(color=RED,width=2.5), marker=dict(color=RED,size=6))
            fig2 = plotly_defaults(fig2, 270)
            fig2.update_layout(showlegend=False,
                xaxis=dict(showgrid=False, tickfont=dict(color="#94a3b8",size=11)),
                yaxis=dict(showgrid=True, gridcolor="#f1f5f9", tickfont=dict(color="#94a3b8",size=11), ticksuffix="%"))
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        card_title("Monthly Delay Patterns (avg across all years)")
        fig3 = go.Figure()
        fig3.add_bar(x=monthly["Month_Name"], y=monthly["Avg_Delay"].round(1),
                     name="Avg Delay (min)", marker_color=RED, opacity=0.85)
        fig3.add_scatter(x=monthly["Month_Name"], y=(monthly["Severe_Rate"]*100).round(1),
                         name="Severe Rate %", mode="lines+markers",
                         line=dict(color=NAVY,width=2), marker=dict(size=5), yaxis="y2")
        fig3 = plotly_defaults(fig3, 270)
        fig3.update_layout(
            legend=dict(orientation="h",y=1.1,font=dict(size=11,color="#64748b")),
            xaxis=dict(showgrid=False, tickfont=dict(color="#94a3b8",size=11)),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", tickfont=dict(color="#94a3b8",size=11)),
            yaxis2=dict(overlaying="y", side="right", tickfont=dict(color="#94a3b8",size=11),
                        ticksuffix="%", showgrid=False))
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ROUTE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Route Analysis" in page:
    section_header("Operational Analytics", "Route Performance")
    c1, c2 = st.columns([1, 2])
    with c1:
        worst = route_sum.sort_values("Incidents", ascending=False).iloc[0]
        st.markdown(f"""<div class="ttc-card">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;">
            <div style="background:#f8f9fa;border-radius:12px;padding:12px;">🚌</div>
            <span class="badge-critical">CRITICAL</span>
          </div>
          <div style="font-family:'Public Sans',sans-serif;font-weight:700;font-size:20px;color:#191c1d;margin-bottom:4px;">
            Route {int(worst['Route_Number'])}</div>
          <div style="font-family:Inter,sans-serif;font-size:14px;color:#64748b;margin-bottom:20px;">
            Highest frequency · avg {worst['Avg_Delay']:.1f} min delay</div>
          <div style="border-top:1px solid #e2e8f0;padding-top:16px;display:flex;justify-content:space-between;">
            <span style="font-size:10px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;">Total incidents</span>
            <span style="font-weight:500;font-size:12px;color:#0f172a;">{int(worst['Incidents']):,}</span>
          </div></div>""", unsafe_allow_html=True)
    with c2:
        with st.container(border=True):
            card_title("Route Efficiency Leaderboard — Top 15 by Incidents")
            top15  = route_sum.sort_values("Incidents", ascending=False).head(15)
            max_d  = top15["Avg_Delay"].max()
            colors = [RED, RED, RED_MID, RED_MID] + ["#f87171"]*11
            for i, (_, row) in enumerate(top15.iterrows(), 1):
                pct = int(row["Avg_Delay"]/max_d*100)
                st.markdown(f"""<div class="lb-row">
                  <div style="display:flex;align-items:center;gap:16px;flex:1;">
                    <span class="lb-rank">{i:02d}</span>
                    <div><div class="lb-route">Route {int(row['Route_Number'])}</div>
                    <div class="lb-sub">{int(row['Incidents']):,} incidents</div></div>
                  </div>
                  <div style="display:flex;align-items:center;gap:24px;">
                    <div style="text-align:right;">
                      <div class="lb-delay">{row['Avg_Delay']:.1f}m</div>
                      <div class="lb-sub">avg delay</div></div>
                    <div style="background:#f1f5f9;border-radius:9999px;width:120px;height:6px;overflow:hidden;">
                      <div style="background:{colors[i-1]};width:{pct}%;height:100%;border-radius:9999px;"></div>
                    </div></div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        card_title("Route Risk Profile — High Delay × Low Frequency")
        st.caption("Risk Score = Avg Delay × Headway × Severe Rate.")
        top_risk = route_sum.sort_values("Risk_Score", ascending=False).head(20)
        fig_r = px.scatter(top_risk, x="Avg_Headway", y="Avg_Delay", size="Incidents",
            color="Severe_Rate", color_continuous_scale=[[0,"#fef2f2"],[0.5,RED],[1,RED_DARK]],
            hover_name="Route_Number",
            labels={"Avg_Headway":"Headway (min)","Avg_Delay":"Avg Delay (min)","Severe_Rate":"Severe Rate"})
        fig_r = plotly_defaults(fig_r, 380)
        fig_r.update_traces(marker=dict(line=dict(width=1,color="white")))
        st.plotly_chart(fig_r, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — WEATHER IMPACT
# ═══════════════════════════════════════════════════════════════════════════════
elif "Weather Impact" in page:
    section_header("Environmental Correlation", "Weather Impact")
    c1, c2 = st.columns([3, 1])
    with c1:
        with st.container(border=True):
            card_title("Temperature vs Avg Delay Duration")
            fig_t = go.Figure()
            fig_t.add_bar(x=temp_agg["Temp_Band"].astype(str), y=temp_agg["Avg_Delay"].round(1),
                marker_color=[RED if v > temp_agg["Avg_Delay"].mean() else "#fca5a5" for v in temp_agg["Avg_Delay"]])
            fig_t = plotly_defaults(fig_t, 290)
            fig_t.update_layout(showlegend=False,
                xaxis=dict(showgrid=False, tickfont=dict(color="#94a3b8",size=11)),
                yaxis=dict(showgrid=True, gridcolor="#f1f5f9", tickfont=dict(color="#94a3b8",size=11)))
            st.plotly_chart(fig_t, use_container_width=True)
    with c2:
        with st.container(border=True):
            card_title("Visibility Impact")
            for _, row in vis_agg.iterrows():
                pct   = int(row["Avg_Delay"]/vis_agg["Avg_Delay"].max()*100)
                color = RED if row["Avg_Delay"] > vis_agg["Avg_Delay"].mean() else "#fca5a5"
                st.markdown(f"""<div style="margin-bottom:16px;">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <span style="font-size:12px;color:#64748b;">{row['Vis_Band']}</span>
                    <span style="font-weight:700;font-size:12px;color:#0f172a;">{row['Avg_Delay']:.1f} min</span>
                  </div>
                  <div style="background:#f1f5f9;border-radius:9999px;height:6px;overflow:hidden;">
                    <div style="background:{color};width:{pct}%;height:100%;border-radius:9999px;"></div>
                  </div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        with st.container(border=True):
            card_title("Seasonal Delay Patterns")
            seas = seasonal.copy()
            seas["Season"] = pd.Categorical(seas["Season"],
                categories=["Winter","Spring","Summer","Fall"], ordered=True)
            seas = seas.sort_values("Season")
            fig_s = px.bar(seas, x="Season", y="Avg_Delay",
                color="Avg_Delay", color_continuous_scale=[[0,"#fef2f2"],[1,RED]],
                labels={"Avg_Delay":"Avg Delay (min)"})
            fig_s = plotly_defaults(fig_s, 260)
            fig_s.update_layout(showlegend=False, coloraxis_showscale=False,
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#f1f5f9"))
            st.plotly_chart(fig_s, use_container_width=True)
    with c4:
        with st.container(border=True):
            card_title("Wind Speed vs Severe Rate")
            fig_w = px.line(wind_agg, x="Wind_Band", y=(wind_agg["Severe_Rate"]*100).round(1),
                markers=True, color_discrete_sequence=[RED],
                labels={"y":"Severe Rate %","Wind_Band":"Wind Speed (km/h)"})
            fig_w.update_traces(line=dict(width=2.5), marker=dict(size=7))
            fig_w = plotly_defaults(fig_w, 260)
            fig_w.update_layout(showlegend=False,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#f1f5f9", ticksuffix="%"))
            st.plotly_chart(fig_w, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — INCIDENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Incident Analysis" in page:
    section_header("Situational Intelligence", "Incident Analysis")
    top_inc = inc.iloc[0]
    c1, c2  = st.columns(2)
    with c1:
        st.markdown(f"""<div class="callout-critical">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
            <span style="font-size:24px;">⚠️</span>
            <div class="callout-title">Top Incident:<br>{top_inc['Incident_Category']}</div>
          </div>
          <div class="callout-body">
            <b>{top_inc['Incident_Category']}</b> is <b>{top_inc['Share']:.0f}%</b> of incidents —
            {int(top_inc['Total']):,} total producing <b>{int(top_inc['Total_Hrs']):,} hours</b> of delay.
            Avg: <b>{top_inc['Avg_Delay']:.1f} min</b> · Severe rate: <b>{top_inc['Severe_Rate']*100:.0f}%</b>.
          </div><br>
          <div style="font-size:13px;color:rgba(147,0,10,0.7);">
            ⚡ Diversion (6.5% of volume) = <b>27.8%</b> of all delay hours.
          </div></div>""", unsafe_allow_html=True)
    with c2:
        with st.container(border=True):
            card_title("Root Cause Distribution")
            fig_tree = px.treemap(inc, path=["Incident_Category"], values="Total",
                color="Avg_Delay",
                color_continuous_scale=[[0,"#fef2f2"],[0.3,"#fca5a5"],[0.7,RED],[1,RED_DARK]],
                custom_data=["Avg_Delay","Share","Severe_Rate"])
            fig_tree.update_traces(
                texttemplate="<b>%{label}</b><br>%{customdata[1]:.1f}%",
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Avg Delay: %{customdata[0]:.1f} min<br>Severe: %{customdata[2]:.1%}<extra></extra>",
                textfont=dict(family="Inter",size=12))
            fig_tree.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=300)
            st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        card_title("Incident Category Breakdown")
        disp = inc.rename(columns={"Incident_Category":"Category","Total":"Incidents",
            "Avg_Delay":"Avg Delay (min)","Severe_Rate":"Severe Rate",
            "Total_Hrs":"Total Hours","Share":"Share %"})
        disp["Severe Rate"] = (disp["Severe Rate"]*100).round(1).astype(str)+"%"
        disp["Total Hours"] = disp["Total Hours"].round(0).astype(int)
        st.dataframe(disp[["Category","Incidents","Avg Delay (min)","Severe Rate","Total Hours","Share %"]],
                     use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PREDICTIVE ROUTING
# ═══════════════════════════════════════════════════════════════════════════════
elif "Predictive Routing" in page:
    section_header("Future Operations", "Predictive Routing")
    if bundle is None:
        st.error("Model bundle not found. Ensure `models/lgbm_ttc_regressor_bundle.pkl` is present.")
        st.stop()

    clf             = bundle["model"]
    route_encoding  = bundle["route_encoding"]
    global_avg      = bundle["global_avg"]
    incident_avg    = bundle["incident_avg"]
    div_route       = bundle["div_route"]
    global_div      = bundle["global_div"]
    feature_columns = bundle["feature_columns"]

    def get_season(m):
        return "Winter" if m in [12,1,2] else "Spring" if m in [3,4,5] else "Summer" if m in [6,7,8] else "Fall"

    def get_headway(route, hour):
        if hw_df is None: return 3.9
        m = hw_df[(hw_df["Route_Number"]==route) & (hw_df["Hour"]==hour)]
        return float(m["Headway_min"].values[0]) if len(m) else 3.9

    c_in, c_out = st.columns([1, 2])
    with c_in:
        with st.container(border=True):
            card_title("Simulation Engine")
            route_num     = st.number_input("Route Number", 1, 999, 29, 1)
            sel_date      = st.date_input("Date", value=date.today())
            sel_time      = st.time_input("Time", value=time(8,0), step=1800)
            incident_type = st.selectbox("Incident Type", sorted(incident_avg.keys()))
            st.markdown("**Weather**")
            temp       = st.slider("Temperature (°C)", -30, 40, 5)
            visibility = st.slider("Visibility (km)", 0, 50, 15)
            wind       = st.slider("Wind (km/h)", 0, 100, 20)
            humidity   = st.slider("Humidity (%)", 0, 100, 70)
            run        = st.button("RUN PREDICTION")

    with c_out:
        with st.container(border=True):
            card_title("Prediction Analysis")
            if run:
                dt      = datetime.combine(sel_date, sel_time)
                hour    = dt.hour
                month   = dt.month
                season  = get_season(month)
                day     = dt.strftime("%A")
                rush    = 1 if hour in [7,8,9,16,17,18] else 0
                weekend = 1 if day in ["Saturday","Sunday"] else 0
                hw      = get_headway(float(route_num), hour)
                r_avg   = route_encoding.get(route_num, global_avg)
                i_avg   = incident_avg.get(incident_type, global_avg)
                dv_avg  = div_route.get(route_num, global_div)
                row = {col: 0 for col in feature_columns}
                row.update({"Hour":hour,"Month":month,"Is_Rush_Hour":rush,"Is_Weekend":weekend,
                    "Temp_C":temp,"Visibility_km":visibility,"Wind_Spd_kmh":wind,"Rel_Humidity_pct":humidity,
                    "Headway_min":hw,"Route_Delay_Avg":r_avg,"Incident_Delay_Avg":i_avg,
                    "Diversion_Route_Avg":dv_avg,"Incident_Route_Interaction":i_avg*r_avg,
                    "Headway_x_Route_Avg":hw*r_avg,"Rush_x_Route_Avg":rush*r_avg,"Hour_x_Month":hour*month})
                for col in [f"Day_{day}",f"Season_{season}",f"Incident_Category_{incident_type}"]:
                    if col in row: row[col] = 1
                X    = np.array([row[col] for col in feature_columns]).reshape(1,-1)
                pred = float(np.clip(np.expm1(clf.predict(X)[0]), 1, 300))
                tier   = "Minor" if pred < 15 else "Moderate" if pred < 30 else "Severe"
                tcolor = "#16a34a" if pred < 15 else "#d97706" if pred < 30 else RED
                r_hist = route_sum[route_sum["Route_Number"]==float(route_num)]["Avg_Delay"]
                r_hist_val = float(r_hist.values[0]) if len(r_hist) else global_avg

                cr1, cr2 = st.columns(2)
                with cr1:
                    st.markdown(f"""
                    <div style="background:#f8f9fa;border-radius:12px;padding:16px;margin-bottom:12px;">
                      <div style="font-size:13px;color:#475569;">Predicted Delay</div>
                      <div style="font-family:'Public Sans',sans-serif;font-weight:800;font-size:36px;color:{tcolor};">{pred:.0f} min</div>
                    </div>
                    <div style="background:#f8f9fa;border-radius:12px;padding:16px;margin-bottom:12px;">
                      <div style="font-size:13px;color:#475569;">Severity</div>
                      <div style="font-family:'Public Sans',sans-serif;font-weight:700;font-size:22px;color:{tcolor};">
                        {'✅' if tier=='Minor' else '⚠️' if tier=='Moderate' else '🔴'} {tier}</div>
                    </div>
                    <div style="background:#f8f9fa;border-radius:12px;padding:16px;">
                      <div style="font-size:13px;color:#475569;">Route {route_num} Hist. Avg</div>
                      <div style="font-family:'Public Sans',sans-serif;font-weight:700;font-size:22px;color:#0f172a;">{r_hist_val:.1f} min</div>
                    </div>""", unsafe_allow_html=True)
                with cr2:
                    fig_g = go.Figure(go.Indicator(mode="gauge+number", value=min(pred,60),
                        number={"suffix":" min","font":{"family":"Public Sans","size":26,"color":tcolor}},
                        gauge={"axis":{"range":[0,60],"tickfont":{"size":10,"color":"#94a3b8"}},
                               "bar":{"color":tcolor},"bgcolor":"#f8f9fa",
                               "steps":[{"range":[0,15],"color":"#dcfce7"},
                                        {"range":[15,30],"color":"#fef3c7"},
                                        {"range":[30,60],"color":"#fef2f2"}]}))
                    fig_g.update_layout(height=200, margin=dict(l=10,r=10,t=10,b=0),
                                        paper_bgcolor="white", font_family="Inter")
                    st.plotly_chart(fig_g, use_container_width=True)

                rush_str = "rush hour" if rush else "off-peak"
                st.markdown(f"""<div class="pred-result">
                  <div class="pred-label">Recommended Strategy</div>
                  <div style="font-size:14px;color:#475569;line-height:1.6;">
                    Route <b style="color:#0f172a">{route_num}</b> predicted
                    <b style="color:{tcolor}">{pred:.0f} min</b> behind schedule ({tier.lower()})
                    during <b style="color:#0f172a">{rush_str}</b>, {day}, {season}.
                    Incident: <b style="color:#0f172a">{incident_type}</b>.
                    Headway: <b style="color:#0f172a">{hw:.1f} min</b>.
                    {'⚡ Deploy replacement vehicle or notify passengers.' if pred >= 15 else '✅ No immediate action required.'}
                  </div></div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div style="background:#f8fafc;border:2px dashed #e2e8f0;border-radius:16px;
                            padding:80px 40px;text-align:center;">
                  <div style="font-size:40px;margin-bottom:16px;">🔮</div>
                  <div style="font-weight:700;font-size:11px;letter-spacing:1px;text-transform:uppercase;color:#94a3b8;">
                    Configure inputs and click Run Prediction</div></div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — VEHICLE INTEL
# ═══════════════════════════════════════════════════════════════════════════════
elif "Vehicle Intel" in page:
    section_header("Fleet Analytics", "Vehicle Intel")
    total_v = int(vi["Total_Vehicles"].iloc[0]) if "Total_Vehicles" in vi.columns else len(vi)
    worst_v = vi.iloc[0]
    high_sv = vi[vi["Severe_Rate"] >= 0.35]

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"""<div class="ttc-card">
          <div class="kpi-label">Vehicles Tracked</div>
          <div class="kpi-value-dark">{total_v:,}</div>
          <div class="kpi-change-down">with mechanical incidents</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="ttc-card">
          <div class="kpi-label">Worst Vehicle</div>
          <div class="kpi-value-red">{int(worst_v['Vehicle'])}</div>
          <div class="kpi-change-down">{int(worst_v['Incidents'])} incidents · {worst_v['Avg_Delay']:.1f} min avg</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="ttc-card">
          <div class="kpi-label">Replacement Candidates (≥35% severe)</div>
          <div class="kpi-value-red">{len(high_sv)}</div>
          <div class="kpi-change-down">vehicles flagged for review</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    with c1:
        with st.container(border=True):
            card_title("Top 30 Vehicles by Mechanical Incidents")
            vi_s = vi.sort_values("Incidents")
            vi_s["Vehicle_Label"] = "Bus " + vi_s["Vehicle"].astype(int).astype(str)
            fig_v = px.bar(vi_s, x="Incidents", y="Vehicle_Label", orientation="h",
                color="Severe_Rate",
                color_continuous_scale=[[0,"#fef2f2"],[0.3,"#fca5a5"],[0.7,RED],[1,RED_DARK]],
                custom_data=["Avg_Delay","Total_Hrs"],
                labels={"Incidents":"Mechanical Incidents","Vehicle_Label":"Vehicle"})
            fig_v = plotly_defaults(fig_v, 600)
            fig_v.update_layout(
                coloraxis_colorbar=dict(title="Severe Rate", tickformat=".0%"),
                xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
                yaxis=dict(showgrid=False, tickfont=dict(size=10), categoryorder="total ascending"))
            fig_v.update_traces(hovertemplate="<b>%{y}</b><br>Incidents: %{x}<br>Avg Delay: %{customdata[0]:.1f} min<extra></extra>")
            st.plotly_chart(fig_v, use_container_width=True)
    with c2:
        with st.container(border=True):
            card_title("Fleet Replacement Priority")
            st.caption("Ranked by Severe Rate × Incidents")
            priority = vi.sort_values(by=["Severe_Rate","Incidents"], ascending=False).head(15)
            for i, (_, row) in enumerate(priority.iterrows(), 1):
                sev_pct = int(row["Severe_Rate"]*100)
                color   = RED if sev_pct >= 35 else ("#d97706" if sev_pct >= 20 else "#94a3b8")
                st.markdown(f"""<div class="lb-row">
                  <div style="display:flex;align-items:center;gap:12px;flex:1;">
                    <span class="lb-rank">{i:02d}</span>
                    <div><div class="lb-route">Bus {int(row['Vehicle'])}</div>
                    <div class="lb-sub">{int(row['Incidents'])} incidents</div></div>
                  </div>
                  <div style="text-align:right;">
                    <div style="font-weight:700;font-size:14px;color:{color};">{sev_pct}% severe</div>
                    <div class="lb-sub">{row['Avg_Delay']:.1f} min avg</div>
                  </div></div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — POWER BI REPORT
# ═══════════════════════════════════════════════════════════════════════════════
elif "Power BI" in page:
    section_header("Business Intelligence", "EDA Dashboard")
    st.caption("Interactive Power BI report — explore delay patterns, route performance, and neighbourhood analysis.")
    components.iframe(
        src=POWERBI_URL,
        width=None,
        height=760,
        scrolling=True,
    )