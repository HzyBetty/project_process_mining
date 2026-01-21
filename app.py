import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import networkx as nx

# ============================================================================
# 1. DATA ENGINE
# ============================================================================

@st.cache_data
def load_and_preprocess_data(file):
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.drop_duplicates(subset=['patient_id', 'timestamp', 'event'])
    
    triage_map = {1: "1 - Immediate", 2: "2 - Emergent", 3: "3 - Urgent", 4: "4 - Less Urgent", 5: "5 - Non-Urgent"}
    df['triage_name'] = df['triage_code'].map(triage_map).fillna("Unknown")
    df = df.sort_values(['patient_id', 'timestamp']).reset_index(drop=True)
    df['is_admitted'] = df['disposition_desc'].str.contains('Admit|admit', na=False).astype(int)
    
    df['gap_hours'] = df.groupby('patient_id')['timestamp'].diff().dt.total_seconds() / 3600
    df['is_new_visit'] = (df['gap_hours'].isna()) | (df['gap_hours'] > 24)
    start_events = ['Ambulance Arrival', 'Triage', 'Registration', 'Ambulance Transfer']
    df.loc[df['event'].isin(start_events), 'is_new_visit'] = True
    
    df['visit_id_num'] = df.groupby('patient_id')['is_new_visit'].cumsum()
    df['visit_key'] = df['patient_id'].astype(str) + "-V" + df['visit_id_num'].astype(str)
    
    df['step_duration'] = df.groupby('visit_key')['timestamp'].diff().dt.total_seconds() / 60
    df.loc[df['is_new_visit'] == True, 'step_duration'] = np.nan
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['visit_step_count'] = df.groupby('visit_key').cumcount() + 1
    return df

# ============================================================================
# 2. UI SETTINGS
# ============================================================================
st.set_page_config(page_title="ED Triage Lead Pro", layout="wide", page_icon="🏥")
st.title("🏥 ED Triage Lead Decision Support Tool")

uploaded_file = st.sidebar.file_uploader("Upload ED Event Log (CSV)", type=['csv'])

if uploaded_file:
    df = load_and_preprocess_data(uploaded_file)
    selected_triage = st.sidebar.multiselect("Filter Triage Levels", sorted(df['triage_name'].unique()), default=df['triage_name'].unique())
    filtered_df = df[df['triage_name'].isin(selected_triage)].copy()
    
    filtered_df['prev_event'] = filtered_df.groupby('visit_key')['event'].shift(1)
    clean_intra_df = filtered_df.dropna(subset=['prev_event', 'step_duration']).copy()
    clean_intra_df = clean_intra_df[clean_intra_df['event'] != clean_intra_df['prev_event']]
    clean_intra_df['transition'] = clean_intra_df['prev_event'] + " → " + clean_intra_df['event']

    # COMMAND CENTER
    st.subheader("🚀 Triage Lead Command Center")
    visit_agg = filtered_df.groupby('visit_key')['timestamp'].agg(lambda x: (x.max() - x.min()).total_seconds()/3600)
    avg_visit_los = visit_agg.mean()
    total_active_visits = filtered_df['visit_key'].nunique()
    
    if not clean_intra_df.empty:
        bottleneck_stats = clean_intra_df.groupby('transition')['step_duration'].mean()
        top_delay_val = bottleneck_stats.max()
        top_delay_name = bottleneck_stats.idxmax()
    else:
        top_delay_val, top_delay_name = 0, "N/A"

    if avg_visit_los > 4:
        staff_status, staff_action, staff_icon, status_func = "🔴 CRITICAL", "Deploy Surge Staffing", "🚨", st.error
    elif avg_visit_los > 2.5:
        staff_status, staff_action, staff_icon, status_func = "🟡 ELEVATED", "Monitor Bed-Ahead Strategy", "⚠️", st.warning
    else:
        staff_status, staff_action, staff_icon, status_func = "🟢 OPTIMAL", "Standard Staffing Levels", "✅", st.success

    c1, c2, c3 = st.columns(3)
    c1.metric("📊 Total Visit Volume", f"{total_active_visits:,}")
    c2.metric("⏳ Avg. Length of Stay", f"{avg_visit_los:.1f} Hrs")
    c3.metric("⚠️ Peak Step Delay", f"{top_delay_val:.0f} Mins")

    status_func(f"### SYSTEM STATUS: {staff_status} | {staff_icon} ACTION: {staff_action}")
    st.markdown("---")

    # SECTION 1: FLOW MAP
    st.header("1️⃣ Patient Flow & Bottleneck Discovery")
    st.info("**Lead Brief:** The axes represent the logical distance between clinical steps. Nodes closer together indicate high-frequency transitions.")
    
    
    
    min_vol = st.slider("Min Patient Volume", 1, 100, 10)
    dfg_counts = clean_intra_df['transition'].value_counts()
    dfg_counts = dfg_counts[dfg_counts >= min_vol]
    G = nx.DiGraph()
    for trans, count in dfg_counts.items():
        u, v = trans.split(" → ")
        G.add_edge(u, v, weight=count)
    
    if G.nodes:
        pos = nx.spring_layout(G, k=2.5, seed=42)
        max_weight = max([G[u][v]['weight'] for u, v in G.edges()]) if G.edges() else 1
        edge_traces = []
        for u, v in G.edges():
            x0, y0 = pos[u]; x1, y1 = pos[v]
            w = (G[u][v]['weight'] / max_weight) * 15 
            edge_traces.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], line=dict(width=max(w, 1.5), color='rgba(50, 171, 96, 0.45)'), mode='lines'))
        node_trace = go.Scatter(x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()], mode='markers+text', text=list(G.nodes()), textposition="top center", marker=dict(size=14, color='DarkSlateGrey', line=dict(width=2, color='white')))
        
        fig_dfg = go.Figure(data=edge_traces + [node_trace])
        fig_dfg.update_layout(showlegend=False, height=550, margin=dict(t=0,b=0,l=0,r=0), 
                             xaxis=dict(title="Relative Connection Proximity (X)", showgrid=False, zeroline=False, showticklabels=False),
                             yaxis=dict(title="Relative Connection Proximity (Y)", showgrid=False, zeroline=False, showticklabels=False))
        st.plotly_chart(fig_dfg, use_container_width=True)
    
    st.markdown("---")

    # SECTION 2: PROTOCOL CONFORMANCE
    st.header("2️⃣ Protocol Safety & The 'Golden Rule'")
    golden_path = ['Triage', 'Registration', 'Assessment', 'Discharge']
    g_cols = st.columns(len(golden_path))
    for i, step in enumerate(golden_path):
        g_cols[i].markdown(f"""<div style="background-color:#F0F2F6; padding:10px; border-radius:5px; text-align:center; border:1px solid #FFD700;"><b>{i+1}. {step}</b></div>""", unsafe_allow_html=True)
    
    v_groups = filtered_df.groupby('visit_key')
    compliant = sum(1 for _, v in v_groups if [e for e in v.sort_values('timestamp')['event'].unique() if e in golden_path] == [e for e in golden_path if e in [x for x in v['event'].unique() if x in golden_path]] and len([x for x in v['event'].unique() if x in golden_path]) >= 2)
    st.metric("Golden Rule Compliance Rate", f"{(compliant/len(v_groups)):.1%}")
    st.table(clean_intra_df.groupby('transition')['step_duration'].mean().sort_values(ascending=False).head(5).rename("Avg Mins"))

    st.markdown("---")

    # SECTION 3: ZONE LOAD
    st.header("3️⃣ Initial Zone Distribution")
    zone_counts = filtered_df.drop_duplicates('visit_key')['initial_zone'].value_counts()
    st.plotly_chart(px.bar(x=zone_counts.index, y=zone_counts.values, labels={'x':'Physical Zone', 'y':'Volume'}, color=zone_counts.values, color_continuous_scale='Blues'), use_container_width=True)

    st.markdown("---")

    # SECTION 4: ADMISSION PREDICTION
    st.header("4️⃣ Multi-Factor Admission Drivers (AI)")
    ml_df = filtered_df.drop_duplicates('visit_key').copy()
    le = LabelEncoder()
    for col in ['gender', 'triage_name', 'initial_zone']:
        ml_df[f'{col}_enc'] = le.fit_transform(ml_df[col].fillna('Unknown'))
    X = ml_df[['age', 'triage_name_enc', 'hour_of_day', 'gender_enc', 'initial_zone_enc', 'visit_step_count']].fillna(0)
    y = ml_df['is_admitted']
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
    importances = pd.DataFrame({'Feature': ['Age', 'Triage', 'Hour', 'Gender', 'Zone', 'Complexity'], 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=True)
    st.plotly_chart(px.bar(importances, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Magma'), use_container_width=True)

    st.markdown("---")

    # SECTION 5: SIMULATION & HEATMAP
    st.header("5️⃣ Capacity Simulation & Hourly Trends")
    st.info("**Lead Brief:** Observe how the green distribution shifts away from the blue baseline as efficiency increases.")
    
    

    h1, h2 = st.columns(2)
    with h1:
        st.subheader("Comparative Efficiency Simulator")
        imp = st.select_slider("Target Efficiency Gain (%)", options=[0, 10, 20, 30, 40, 50], value=20)
        std_dev = visit_agg.std() if not pd.isna(visit_agg.std()) else 0.5
        baseline = np.random.normal(avg_visit_los, std_dev, 1000)
        improved = np.random.normal(avg_visit_los * (1 - (imp/100)), std_dev, 1000)
        sim_df = pd.DataFrame({'LOS': np.concatenate([baseline, improved]), 'Scenario': ['Baseline']*1000 + [f'Target']*1000})
        
        fig_sim = px.histogram(sim_df, x='LOS', color='Scenario', barmode='overlay', nbins=50, color_discrete_map={'Baseline': '#636EFA', 'Target': '#00CC96'})
        
        # Fixed Overlapping Annotations
        fig_sim.add_vline(x=avg_visit_los, line_dash="dash", line_color="blue")
        fig_sim.add_vline(x=avg_visit_los * (1 - (imp/100)), line_dash="dash", line_color="green")
        
        fig_sim.add_annotation(x=avg_visit_los, y=1, yref="paper", text="Current Avg", showarrow=False, font=dict(color="blue"), bgcolor="white", xanchor="left")
        fig_sim.add_annotation(x=avg_visit_los * (1 - (imp/100)), y=0.9, yref="paper", text="Target Avg", showarrow=False, font=dict(color="green"), bgcolor="white", xanchor="right")
        
        st.plotly_chart(fig_sim, use_container_width=True)
        st.success(f"💡 Potential to save **{avg_visit_los * (imp/100):.1f} hours** per patient.")

    with h2:
        st.subheader("Hourly Transition Heatmap")
        heatmap_data = clean_intra_df.groupby(['hour_of_day', 'transition'])['step_duration'].mean().unstack().fillna(0)
        st.plotly_chart(px.imshow(heatmap_data.T, color_continuous_scale='Reds'), use_container_width=True)

    st.markdown("---")

    # SECTION 6: RED FLAGS
    st.header("6️⃣ Red Flag Alerts")
    limit = clean_intra_df['step_duration'].mean() + (2 * clean_intra_df['step_duration'].std())
    anoms = clean_intra_df[clean_intra_df['step_duration'] > limit]
    if not anoms.empty:
        target_v = st.selectbox("Investigate Delay:", sorted(anoms['visit_key'].unique()))
        st.table(filtered_df[filtered_df['visit_key'] == target_v][['timestamp', 'event', 'step_duration']])
    else:
        st.success("Flow within statistical control.")

else:
    st.info("Please upload an ED Event Log CSV via the sidebar to begin.")