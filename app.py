import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import networkx as nx

# ============================================================================
# 1. ANALYTICS ENGINE
# ============================================================================

@st.cache_data
def load_and_preprocess_data(file):
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    triage_map = {
        1: "1 - Immediate",
        2: "2 - Emergent",
        3: "3 - Urgent",
        4: "4 - Less Urgent",
        5: "5 - Non-Urgent"
    }
    df['triage_name'] = df['triage_code'].map(triage_map).fillna("Unknown")
    df = df.sort_values(['patient_id', 'timestamp']).reset_index(drop=True)
    df['is_admitted'] = df['disposition_desc'].str.contains('Admit|admit', na=False).astype(int)
    df['step_duration'] = df.groupby('patient_id')['timestamp'].diff().dt.total_seconds() / 60
    return df

@st.cache_data
def train_admission_model(data):
    p_df = data.groupby('patient_id').agg({
        'is_admitted': 'max', 'age': 'first', 'triage_code': 'first', 
        'gender': 'first', 'initial_zone': 'first', 'timestamp': 'first'
    }).reset_index().dropna()
    if p_df['is_admitted'].nunique() < 2: return None
    le_g, le_z = LabelEncoder(), LabelEncoder()
    X = pd.DataFrame({
        'age': p_df['age'], 'triage_code': p_df['triage_code'],
        'gender': le_g.fit_transform(p_df['gender'].astype(str)),
        'zone': le_z.fit_transform(p_df['initial_zone'].astype(str)),
        'hour': p_df['timestamp'].dt.hour
    })
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, p_df['is_admitted'])
    return model, ['Age', 'Triage Level', 'Gender', 'Zone', 'Hour'], model.feature_importances_

# ============================================================================
# 2. UI HEADER & SIDEBAR
# ============================================================================
st.set_page_config(page_title="ED Triage Intelligence", layout="wide", page_icon="🏥")

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; color: #1f77b4; font-weight: bold; }
    [data-testid="stMetricLabel"] { font-size: 16px; }
    .stDataFrame { border: 1px solid #e6e9ef; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏥 ED Triage Lead Decision Support Tool")

uploaded_file = st.sidebar.file_uploader("Upload ED Event Log (CSV)", type=['csv'])

if uploaded_file:
    df = load_and_preprocess_data(uploaded_file)
    
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("🔍 Global Filters")
    st.sidebar.markdown("""
    **Urgency Guide:**
    - <span style='color:red'>🔴 Immediate</span>
    - <span style='color:orange'>🟠 Emergent</span>
    - <span style='color:#fdbc06'>🟡 Urgent</span>
    - <span style='color:green'>🟢 Lower Acuity</span>
    """, unsafe_allow_html=True)
    
    triage_options = sorted(df['triage_name'].unique().tolist())
    selected_triage_names = st.sidebar.multiselect("Select Triage Levels", triage_options, default=triage_options)
    
    filtered_df = df[df['triage_name'].isin(selected_triage_names)]
    
    # --- TOP LEVEL SUMMARY ---
    st.header("📋 Current Shift Overview")
    triage_counts = filtered_df.groupby('triage_name')['patient_id'].nunique()
    
    if not triage_counts.empty:
        items_per_row = 3
        current_options = [opt for opt in triage_options if opt in triage_counts.index]
        rows = [current_options[i:i + items_per_row] for i in range(0, len(current_options), items_per_row)]
        
        for row_items in rows:
            cols = st.columns(items_per_row)
            for i, name in enumerate(row_items):
                count = triage_counts[name]
                display_label = name.split(" - ")[1] if " - " in name else name
                cols[i].metric(label=f"{display_label} Total", value=f"{count}")
    st.markdown("---")

    # --- SECTION 1: DISCOVERY & BOTTLENECKS ---
    st.header("1️⃣ Process Discovery: Patient Flow Volume")
    
    if not filtered_df.empty:
        bottleneck_data = []
        for p, p_df in filtered_df.groupby('patient_id'):
            p_df = p_df.sort_values('timestamp')
            events = p_df['event'].tolist()
            times = p_df['timestamp'].tolist()
            for i in range(len(events)-1):
                duration = (times[i+1] - times[i]).total_seconds() / 60
                bottleneck_data.append({'transition': f"{events[i]} → {events[i+1]}", 'duration': duration})
        
        if bottleneck_data:
            b_df = pd.DataFrame(bottleneck_data).groupby('transition')['duration'].mean().sort_values(ascending=False)
            top_n = min(3, len(b_df))
            bottleneck_text = "  \n".join([f"{i+1}. **{b_df.index[i]}**: {b_df.iloc[i]:.1f} mins avg" for i in range(top_n)])
            st.info(f"💡 **Triage Lead Guidance:** Thicker lines represent high-volume paths.  \n**Top Bottlenecks:** \n{bottleneck_text}")

    
    min_freq = st.slider("Min Patient Volume for Map", 1, 100, 10)
    
    dfg_counts = defaultdict(int)
    for p, p_df in filtered_df.groupby('patient_id'):
        events = p_df['event'].tolist()
        for i in range(len(events)-1):
            dfg_counts[(events[i], events[i+1])] += 1
    
    edges_to_draw = {k: v for k, v in dfg_counts.items() if v >= min_freq}
    G = nx.DiGraph()
    for (s, t), c in edges_to_draw.items(): G.add_edge(s, t, weight=c)
    
    if len(G.nodes) > 0:
        pos = nx.spring_layout(G, k=2.5, seed=42)
        edge_traces = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
            weight = G[edge[0]][edge[1]]['weight']
            width = np.interp(weight, [min_freq, max(edges_to_draw.values() or [1])], [1, 12])
            edge_traces.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                                          line=dict(width=width, color='rgba(52, 152, 219, 0.4)'),
                                          hoverinfo='text', text=f"Volume: {weight}", mode='lines'))
        
        node_trace = go.Scatter(x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()],
                                mode='markers+text', text=list(G.nodes()), textposition="top center",
                                marker=dict(size=22, color='DarkBlue', line=dict(width=2, color='white')))
        fig_dfg = go.Figure(data=edge_traces + [node_trace])
        fig_dfg.update_layout(height=500, showlegend=False, xaxis_visible=False, yaxis_visible=False, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_dfg, use_container_width=True)

    st.markdown("---")

    # --- SECTION 2: PROTOCOL SAFETY ---
    st.header("2️⃣ Safety Protocol Conformance")
    happy_path = ['Triage', 'Registration', 'Assessment', 'Discharge', 'Left ED']
    st.write(f"**Clinical Gold Standard Protocol:** {' → '.join(happy_path)}")
    
    total_pts = filtered_df['patient_id'].nunique()
    if total_pts > 0:
        deviations = []
        compliant_count = 0
        for p, p_df in filtered_df.groupby('patient_id'):
            actual = p_df.sort_values('timestamp')['event'].tolist()
            if all(s in actual for s in happy_path):
                indices = [actual.index(s) for s in happy_path]
                if indices == sorted(indices): compliant_count += 1
                else: deviations.append({"ID": p, "Issue": "Sequence Violation", "Trace": " → ".join(actual[:5])})
            else: deviations.append({"ID": p, "Issue": "Missing Step", "Trace": " → ".join(actual[:5])})
        
        c1, c2 = st.columns([1, 3])
        c1.metric("Compliance Rate", f"{(compliant_count/total_pts):.1%}")
        with c2:
            with st.expander("View Conformance Exceptions"):
                st.dataframe(pd.DataFrame(deviations).head(10), use_container_width=True, hide_index=True)

    st.markdown("---")

    # --- SECTION 3: QUEUE & LOAD BALANCE ---
    st.header("3️⃣ Queue Analytics & Load Balancing")
    
    wait_df = filtered_df.groupby('initial_zone')['step_duration'].mean().reset_index().sort_values('step_duration', ascending=False)
    fig_q = px.bar(wait_df, x='initial_zone', y='step_duration', 
                   title="Average Patient Wait Time (Minutes) per Zone",
                   labels={'step_duration': 'Avg Wait (Min)', 'initial_zone': 'Zone'},
                   color='step_duration', color_continuous_scale='Reds')
    st.plotly_chart(fig_q, use_container_width=True)
    
    st.subheader("Acuity-to-Zone Distribution (Load Balancing)")
    # Added Explanation for Triage Lead
    st.info("💡 **Operational Insight:** Look for high-density 'hot spots' (darker red). If high-acuity levels (Immediate/Emergent) are accumulating in zones with high wait times above, consider reallocating nursing staff to that zone immediately.")
    
    
    if not filtered_df.empty:
        heatmap_data = filtered_df.groupby(['triage_name', 'initial_zone'])['patient_id'].nunique().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='triage_name', columns='initial_zone', values='patient_id').fillna(0)
        
        fig_heat = px.imshow(heatmap_pivot, 
                             labels=dict(x="Hospital Zone", y="Triage Level", color="Patient Count"),
                             x=heatmap_pivot.columns,
                             y=heatmap_pivot.index,
                             color_continuous_scale='YlOrRd',
                             aspect="auto")
        fig_heat.update_xaxes(side="top")
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # --- SECTION 4: PREDICTIVE ADMISSION ---
    st.header("4️⃣ Predictive Admission Analytics")
    ml_res = train_admission_model(filtered_df)
    if ml_res:
        model, names, imps = ml_res
        feat_df = pd.DataFrame({'Feature': names, 'Impact': imps}).sort_values('Impact', ascending=False)
        st.plotly_chart(px.bar(feat_df, x='Impact', y='Feature', orientation='h', height=350, title="Key Drivers of Hospital Admission"), use_container_width=True)

    st.markdown("---")

    # --- SECTION 5: CAPACITY SIMULATION ---
    st.header("5️⃣ Capacity Simulation: 'What-If' Planning")
    improve = st.select_slider("Targeted Efficiency Gain", options=[0, 10, 20, 30, 40, 50], format_func=lambda x: f"{x}% Faster")
    if not filtered_df.empty:
        current_los = filtered_df.groupby('patient_id')['timestamp'].agg(lambda x: (x.max() - x.min()).total_seconds()/3600).mean()
        sim_data = np.random.normal(current_los * (1 - (improve/100)), current_los * 0.12, 1000)
        fig_sim = px.histogram(sim_data, nbins=40, title="Projected ED Length of Stay (Hours) Distribution", color_discrete_sequence=['#2ecc71'])
        fig_sim.add_vline(x=current_los, line_dash="dash", line_color="red", annotation_text="Current Baseline")
        st.plotly_chart(fig_sim, use_container_width=True)
        

    st.markdown("---")

    # --- SECTION 6: RED FLAG ALERTS ---
    st.header("6️⃣ Red Flag Alerts: Patient Deep-Dive")
    limit = filtered_df['step_duration'].mean() + (2 * filtered_df['step_duration'].std())
    anoms = filtered_df[filtered_df['step_duration'] > limit].sort_values('step_duration', ascending=False)
    
    if not anoms.empty:
        target_patient = st.selectbox("Select Patient to Investigate:", anoms['patient_id'].unique())
        p_history = filtered_df[filtered_df['patient_id'] == target_patient].sort_values('timestamp')
        st.dataframe(p_history[['timestamp', 'event', 'initial_zone', 'step_duration']].style.apply(
            lambda x: ['background-color: #ffcccc' if (isinstance(v, float) and v > limit) else '' for v in x], subset=['step_duration']
        ), use_container_width=True, height=300)
    else:
        st.success("No anomalies detected.")
else:
    st.info("👋 Welcome. Please upload an ED Event Log CSV to activate the dashboard.")