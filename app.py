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
# 2. UI LAYOUT
# ============================================================================
st.set_page_config(page_title="ED Triage Lead Pro", layout="wide", page_icon="🏥")

st.title("🏥 ED Triage Lead Decision Support Tool")
st.markdown("""
*This dashboard provides real-time visibility into clinical workflows. Use these insights to identify patient flow obstructions, 
monitor protocol safety, and simulate staffing impacts.*
""")

uploaded_file = st.sidebar.file_uploader("Upload ED Event Log (CSV)", type=['csv'])

if uploaded_file:
    df = load_and_preprocess_data(uploaded_file)
    selected_triage = st.sidebar.multiselect("Filter Triage Levels", sorted(df['triage_name'].unique()), default=df['triage_name'].unique())
    filtered_df = df[df['triage_name'].isin(selected_triage)].copy()
    
    # Clean intra-visit transitions
    filtered_df['prev_event'] = filtered_df.groupby('visit_key')['event'].shift(1)
    clean_intra_df = filtered_df.dropna(subset=['prev_event', 'step_duration']).copy()
    clean_intra_df['transition'] = clean_intra_df['prev_event'] + " → " + clean_intra_df['event']

    # --- SECTION 1: PROCESS DISCOVERY ---
    st.header("1️⃣ Patient Flow & Bottleneck Discovery")
    with st.expander("📝 How to read this"):
        st.write("""
        **The Map:** Shows the 'Beaten Path' of patients. **Thicker lines** indicate higher patient volume. 
        **The Table:** Highlights the slowest clinical handoffs. If 'Triage → Assessment' is high, consider re-allocating nurses to the front-end.
        """)
    
    min_volume = st.slider("Filter: Show paths with at least X patients", 1, 100, 10)
    dfg_counts = clean_intra_df['transition'].value_counts()
    dfg_counts = dfg_counts[dfg_counts >= min_volume]
    
    col1, col2 = st.columns([2, 1])
    with col1:
        G = nx.DiGraph()
        for trans, count in dfg_counts.items():
            u, v = trans.split(" → ")
            G.add_edge(u, v, weight=count)
        
        if G.nodes:
            pos = nx.spring_layout(G, k=2, seed=42)
            max_v = dfg_counts.max() if not dfg_counts.empty else 1
            edge_traces = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
                width = (G[edge[0]][edge[1]]['weight'] / max_v) * 10
                edge_traces.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], 
                                             line=dict(width=max(width, 1), color='rgba(50, 171, 96, 0.4)'), mode='lines'))
            node_trace = go.Scatter(x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()], 
                                   mode='markers+text', text=list(G.nodes()), textposition="top center", marker=dict(size=12, color='DarkSlateGrey'))
            st.plotly_chart(go.Figure(data=edge_traces + [node_trace], layout=go.Layout(showlegend=False, height=450, margin=dict(t=0,b=0,l=0,r=0))), use_container_width=True)
    
    with col2:
        stats = clean_intra_df.groupby('transition')['step_duration'].mean().sort_values(ascending=False).head(5)
        st.table(stats.rename("Avg Delay (Mins)"))

    st.markdown("---")

    # --- HOURLY HEATMAP ---
    st.header("🕒 Hourly Efficiency Heatmap")
    with st.expander("📝 Clinical Utility"):
        st.write("""
        This heatmap identifies **peak congestion times**. Darker red squares represent times of day where specific 
        handoffs take significantly longer. Use this to justify 'surge' staffing (e.g., adding a resident at 2:00 PM).
        """)
    heatmap_data = clean_intra_df.groupby(['hour_of_day', 'transition'])['step_duration'].mean().unstack().fillna(0)
    fig_heat = px.imshow(heatmap_data.T, labels=dict(x="Hour of Day (24h)", y="Process Step", color="Mins"),
                         color_continuous_scale='Reds', aspect="auto")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # --- SECTION 2: THE GOLDEN RULE ---
    st.header("2️⃣ Protocol Safety & The Golden Rule")
    with st.expander("📝 Why this matters"):
        st.write("""
        We measure **Compliance** based on the correct chronological order of clinical milestones. 
        A patient should not be Discharged before they are Triaged. High non-compliance often signals 
        emergency bypasses or data entry errors.
        """)
    
    # VISUAL REFERENCE
    golden_path = ['Triage', 'Registration', 'Assessment', 'Discharge']
    st.subheader("Standard Clinical Protocol (Goal Sequence)")
    cols = st.columns(len(golden_path))
    for i, step in enumerate(golden_path):
        cols[i].markdown(f"""
            <div style="background-color:#F0F2F6; padding:10px; border-radius:10px; text-align:center; color:#1f1f1f; border:2px solid #FFD700; font-weight:bold;">
                Step {i+1}: {step}
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    v_groups = filtered_df.groupby('visit_key')
    compliant_count = 0
    for _, v_data in v_groups:
        actual_sequence = v_data.sort_values('timestamp')['event'].tolist()
        relevant_steps = [s for s in actual_sequence if s in golden_path]
        clean_seq = [relevant_steps[i] for i in range(len(relevant_steps)) if i == 0 or relevant_steps[i] != relevant_steps[i-1]]
        ideal_indices = [golden_path.index(s) for s in clean_seq]
        if ideal_indices == sorted(ideal_indices) and len(ideal_indices) > 1:
            compliant_count += 1

    c1, c2 = st.columns(2)
    c1.metric("Visits Evaluated", len(v_groups))
    c2.metric("Sequence Compliance", f"{(compliant_count/len(v_groups)):.1%}")

    st.markdown("---")

    # --- SECTION 5: SIMULATION ---
    st.header("5️⃣ Capacity & Flow Simulation")
    with st.expander("📝 Decision Support"):
        st.write("""
        Adjust the slider to simulate a reduction in **Length of Stay (LOS)**. 
        The graph shows how the entire ED population shifts toward faster throughput. 
        """)
    improvement = st.select_slider("Target Efficiency Gain (%)", options=[0, 10, 20, 30, 40, 50], value=10)
    visit_los = filtered_df.groupby('visit_key')['timestamp'].agg(lambda x: (x.max() - x.min()).total_seconds()/3600)
    current_avg = visit_los.mean()
    sim_mean = current_avg * (1 - (improvement/100))
    sim_data = np.random.normal(sim_mean, visit_los.std() or 0.5, 1000)
    
    fig_sim = px.histogram(sim_data, nbins=50, title="Projected Hours in ED (Simulated)", color_discrete_sequence=['#4CAF50'])
    fig_sim.add_vline(x=current_avg, line_dash="dash", line_color="red", annotation_text="Current Baseline")
    st.plotly_chart(fig_sim, use_container_width=True)

    # --- SECTION 6: RED FLAG ALERTS ---
    st.header("6️⃣ Red Flag Alerts (Individual Delays)")
    with st.expander("📝 Investigative Action"):
        st.write("""
        These visits contain clinical steps that are **statistically extreme** (Mean + 2SD). 
        Click on a visit key to review the timeline and identify if the delay was due to clinical complexity 
        or an operational failure.
        """)
    limit = clean_intra_df['step_duration'].mean() + (2 * clean_intra_df['step_duration'].std())
    anoms = clean_intra_df[clean_intra_df['step_duration'] > limit]
    if not anoms.empty:
        target_v = st.selectbox("Select Outlier Visit:", sorted(anoms['visit_key'].unique()))
        st.dataframe(filtered_df[filtered_df['visit_key'] == target_v][['timestamp', 'event', 'step_duration']])
    else:
        st.success("Zero critical delays detected in current view.")

else:
    st.info("Please upload your ED Log CSV to begin.")