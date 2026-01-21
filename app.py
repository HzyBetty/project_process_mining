import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import networkx as nx

# ============================================================================
# 1. DATA ENGINE (Strict 24-Hour Isolation)
# ============================================================================

@st.cache_data
def load_and_preprocess_data(file):
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    triage_map = {1: "1 - Immediate", 2: "2 - Emergent", 3: "3 - Urgent", 4: "4 - Less Urgent", 5: "5 - Non-Urgent"}
    df['triage_name'] = df['triage_code'].map(triage_map).fillna("Unknown")
    df = df.sort_values(['patient_id', 'timestamp']).reset_index(drop=True)
    df['is_admitted'] = df['disposition_desc'].str.contains('Admit|admit', na=False).astype(int)
    
    # Isolation Rules
    df['gap_hours'] = df.groupby('patient_id')['timestamp'].diff().dt.total_seconds() / 3600
    df['is_new_visit'] = (df['gap_hours'].isna()) | (df['gap_hours'] > 24)
    start_events = ['Ambulance Arrival', 'Triage', 'Registration', 'Ambulance Transfer']
    df.loc[df['event'].isin(start_events), 'is_new_visit'] = True
    
    df['visit_id_num'] = df.groupby('patient_id')['is_new_visit'].cumsum()
    df['visit_key'] = df['patient_id'].astype(str) + "-V" + df['visit_id_num'].astype(str)
    
    df['step_duration'] = df.groupby('visit_key')['timestamp'].diff().dt.total_seconds() / 60
    df.loc[df['is_new_visit'] == True, 'step_duration'] = np.nan
    df['hour_of_day'] = df['timestamp'].dt.hour
    return df

# ============================================================================
# 2. UI LAYOUT & SETTINGS
# ============================================================================
st.set_page_config(page_title="ED Triage Lead Pro", layout="wide", page_icon="🏥")

st.title("🏥 ED Triage Lead Decision Support Tool")

uploaded_file = st.sidebar.file_uploader("Upload ED Event Log (CSV)", type=['csv'])

if uploaded_file:
    df = load_and_preprocess_data(uploaded_file)
    selected_triage = st.sidebar.multiselect("Filter Triage Levels", sorted(df['triage_name'].unique()), default=df['triage_name'].unique())
    filtered_df = df[df['triage_name'].isin(selected_triage)].copy()
    
    # Pre-calculate Transitions
    filtered_df['prev_event'] = filtered_df.groupby('visit_key')['event'].shift(1)
    clean_intra_df = filtered_df.dropna(subset=['prev_event', 'step_duration']).copy()
    clean_intra_df['transition'] = clean_intra_df['prev_event'] + " → " + clean_intra_df['event']

    # ============================================================================
    # 3. EXECUTIVE COMMAND CENTER (TWO-ROW GRID)
    # ============================================================================
    st.subheader("🚀 Triage Lead Command Center")
    
    # Calculations
    visit_agg = filtered_df.groupby('visit_key')['timestamp'].agg(lambda x: (x.max() - x.min()).total_seconds()/3600)
    avg_visit_los = visit_agg.mean()
    total_active_visits = filtered_df['visit_key'].nunique()
    
    if not clean_intra_df.empty:
        bottleneck_stats = clean_intra_df.groupby('transition')['step_duration'].mean()
        top_delay_val = bottleneck_stats.max()
        top_delay_name = bottleneck_stats.idxmax() # Full name preserved
    else:
        top_delay_val, top_delay_name = 0, "N/A"

    # Staffing Logic
    if avg_visit_los > 4:
        staff_status, staff_action, staff_icon, box_type = "🔴 CRITICAL", "Deploy Surge Staffing", "🚨", "error"
    elif avg_visit_los > 2.5:
        staff_status, staff_action, staff_icon, box_type = "🟡 ELEVATED", "Monitor Bed-Ahead Strategy", "⚠️", "warning"
    else:
        staff_status, staff_action, staff_icon, box_type = "🟢 OPTIMAL", "Standard Staffing Levels", "✅", "success"

    # ROW 1: High-Level Metrics
    r1_col1, r1_col2, r1_col3 = st.columns(3)
    r1_col1.metric("📊 Total Visit Volume", f"{total_active_visits:,}")
    r1_col2.metric("⏳ Avg. Length of Stay", f"{avg_visit_los:.1f} Hrs")
    r1_col3.metric("⚠️ Peak Step Delay", f"{top_delay_val:.0f} Mins")

    # ROW 2: Actionable Details (Full horizontal width for long text)
    r2_col1, r2_col2 = st.columns([1.5, 1])
    with r2_col1:
        st.write(f"**Current Primary Bottleneck:**")
        st.error(f"🛑 {top_delay_name}")
    with r2_col2:
        st.write(f"**System Status: {staff_status}**")
        if box_type == "error": st.error(f"{staff_icon} {staff_action}")
        elif box_type == "warning": st.warning(f"{staff_icon} {staff_action}")
        else: st.success(f"{staff_icon} {staff_action}")

    st.markdown("---")

    # --- SECTION 1: PROCESS DISCOVERY ---
    st.header("1️⃣ Patient Flow & Bottleneck Discovery")
    
    min_volume = st.slider("Min Patient Volume for Map Visualization", 1, 100, 10)
    
    dfg_counts = clean_intra_df['transition'].value_counts()
    dfg_counts = dfg_counts[dfg_counts >= min_volume]
    
    col1, col2 = st.columns([2, 1])
    with col1:
        G = nx.DiGraph()
        for trans, count in dfg_counts.items():
            u, v = trans.split(" → ")
            G.add_edge(u, v, weight=count)
        
        if G.nodes:
            pos = nx.spring_layout(G, k=2.5, seed=42)
            max_v = dfg_counts.max() if not dfg_counts.empty else 1
            edge_traces = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
                width = (G[edge[0]][edge[1]]['weight'] / max_v) * 10
                edge_traces.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], 
                                             line=dict(width=max(width, 1), color='rgba(50, 171, 96, 0.4)'), mode='lines'))
            node_trace = go.Scatter(x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()], 
                                   mode='markers+text', text=list(G.nodes()), textposition="top center", 
                                   marker=dict(size=10, color='DarkSlateGrey'))
            st.plotly_chart(go.Figure(data=edge_traces + [node_trace], layout=go.Layout(showlegend=False, height=500, margin=dict(t=0,b=0,l=0,r=0))), use_container_width=True)
    
    with col2:
        st.subheader("Slowest Transitions")
        st.table(clean_intra_df.groupby('transition')['step_duration'].mean().sort_values(ascending=False).head(5).rename("Mins"))

    # --- HOURLY HEATMAP ---
    st.header("🕒 Hourly Efficiency Heatmap")
    
    heatmap_data = clean_intra_df.groupby(['hour_of_day', 'transition'])['step_duration'].mean().unstack().fillna(0)
    fig_heat = px.imshow(heatmap_data.T, labels=dict(x="Hour of Day", y="Transition", color="Mins"),
                         color_continuous_scale='Reds', aspect="auto")
    st.plotly_chart(fig_heat, use_container_width=True)

    # --- SECTION 2: THE GOLDEN RULE ---
    st.header("2️⃣ Protocol Safety & Conformance")
    
    golden_path = ['Triage', 'Registration', 'Assessment', 'Discharge']
    g_cols = st.columns(len(golden_path))
    for i, step in enumerate(golden_path):
        g_cols[i].markdown(f"""<div style="background-color:#F0F2F6; padding:10px; border-radius:5px; text-align:center; border:1px solid #FFD700;"><b>{i+1}. {step}</b></div>""", unsafe_allow_html=True)
    
    v_groups = filtered_df.groupby('visit_key')
    compliant_count = 0
    for _, v_data in v_groups:
        actual_seq = v_data.sort_values('timestamp')['event'].tolist()
        relevant = [s for s in actual_seq if s in golden_path]
        clean_seq = [relevant[i] for i in range(len(relevant)) if i == 0 or relevant[i] != relevant[i-1]]
        indices = [golden_path.index(s) for s in clean_seq]
        if indices == sorted(indices) and len(indices) > 1:
            compliant_count += 1
    
    st.metric("Sequence Compliance Rate", f"{(compliant_count/len(v_groups)):.1%}")

    # --- SECTION 5: SIMULATION ---
    st.header("5️⃣ Capacity Simulation (Monte Carlo)")
    
    improvement = st.select_slider("Simulate Efficiency Gain (%)", options=[0, 10, 20, 30, 40, 50], value=10)
    sim_mean = avg_visit_los * (1 - (improvement/100))
    sim_data = np.random.normal(sim_mean, visit_agg.std() or 0.5, 1000)
    
    fig_sim = px.histogram(sim_data, nbins=50, title=f"Projected LOS Distr ({improvement}% Gain)", color_discrete_sequence=['#4CAF50'])
    fig_sim.add_vline(x=avg_visit_los, line_dash="dash", line_color="red", annotation_text="Baseline")
    st.plotly_chart(fig_sim, use_container_width=True)

    # --- SECTION 6: RED FLAG ALERTS ---
    st.header("6️⃣ Red Flag Alerts")
    limit = clean_intra_df['step_duration'].mean() + (2 * clean_intra_df['step_duration'].std())
    anoms = clean_intra_df[clean_intra_df['step_duration'] > limit]
    if not anoms.empty:
        target_v = st.selectbox("Investigate Delay:", sorted(anoms['visit_key'].unique()))
        st.table(filtered_df[filtered_df['visit_key'] == target_v][['timestamp', 'event', 'step_duration']])
    else:
        st.success("No critical delays detected.")

else:
    st.info("Please upload an ED Event Log CSV via the sidebar.")