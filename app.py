import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict

# ============================================================================
# 1. HARD-CODED DATA CLEANING (Atomic Visit Logic)
# ============================================================================

@st.cache_data
def load_and_preprocess_data(file):
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort strictly
    df = df.sort_values(['patient_id', 'timestamp']).reset_index(drop=True)
    
    # --- 48-HOUR ISOLATION RULE ---
    df['gap_hours'] = df.groupby('patient_id')['timestamp'].diff().dt.total_seconds() / 3600
    df['is_new_visit'] = (df['gap_hours'].isna()) | (df['gap_hours'] > 48)
    df['visit_id'] = df.groupby('patient_id')['is_new_visit'].cumsum()
    
    # --- DURATION RESET ---
    # We calculate duration only within the visit. 
    # If it's a new visit, we force the duration to be NaN.
    df['step_duration'] = df.groupby(['patient_id', 'visit_id'])['timestamp'].diff().dt.total_seconds() / 60
    df.loc[df['is_new_visit'] == True, 'step_duration'] = np.nan
    
    return df

# ============================================================================
# 2. THE "BRIDGE-KILLER" BOTTLENECK ENGINE
# ============================================================================

def get_bottleneck_guidance(filtered_df):
    """
    This function creates the text you see in Section 1.
    It EXPLICITLY filters out any transition where visit IDs don't match.
    """
    # Create shift within patient group
    temp_df = filtered_df.copy()
    temp_df['prev_event'] = temp_df.groupby('patient_id')['event'].shift(1)
    temp_df['prev_visit_id'] = temp_df.groupby('patient_id')['visit_id'].shift(1)
    
    # MANDATORY: A transition is only valid if visit_id == prev_visit_id
    # This deletes "Left ED -> Triage" (the 20,000 min gap)
    valid_transitions = temp_df[
        (temp_df['visit_id'] == temp_df['prev_visit_id']) & 
        (temp_df['step_duration'].notna())
    ].copy()
    
    valid_transitions['path'] = valid_transitions['prev_event'] + " → " + valid_transitions['event']
    
    # Calculate stats
    stats = valid_transitions.groupby('path')['step_duration'].mean().sort_values(ascending=False)
    
    if stats.empty:
        return "No intra-visit bottlenecks detected.", valid_transitions
        
    top_3 = stats.head(3)
    # This creates the text dynamically based on the CLEAN data
    guidance = "  \n".join([f"{i+1}. **{top_3.index[i]}**: {top_3.iloc[i]:.1f} mins avg" for i in range(len(top_3))])
    
    return guidance, valid_transitions

# ============================================================================
# 3. UI LAYOUT
# ============================================================================

st.set_page_config(page_title="ED Triage Expert", layout="wide")
st.title("🏥 ED Process Intelligence")

uploaded_file = st.sidebar.file_uploader("Upload ED CSV", type=['csv'])

if uploaded_file:
    df = load_and_preprocess_data(uploaded_file)
    
    # Sidebar Filters
    triage_options = sorted(df['triage_code'].unique())
    selected_triage = st.sidebar.multiselect("Triage Level", triage_options, default=triage_options)
    
    # Apply filters
    filtered_df = df[df['triage_code'].isin(selected_triage)]
    
    # --- SECTION 1: DYNAMIC BOTTLENECK GUIDANCE ---
    st.header("1️⃣ Process Discovery: Intra-Visit Flow")
    
    # EXECUTE THE CLEANING ENGINE
    guidance_text, clean_data = get_bottleneck_guidance(filtered_df)
    
    # RENDER THE DYNAMIC TEXT (This replaces the old hard-coded string)
    st.info(f"💡 **Triage Lead Guidance:** Only intra-visit transitions (within 48hrs) are analyzed.  \n\n**Top Bottlenecks:** \n{guidance_text}")

    

    # --- SECTION 6: RED FLAG ALERTS ---
    st.header("6️⃣ Red Flag Alerts")
    
    if not clean_data.empty:
        # Threshold using the clean data
        limit = clean_data['step_duration'].mean() + (2 * clean_data['step_duration'].std())
        anomalies = clean_data[clean_data['step_duration'] > limit]
        
        if not anomalies.empty:
            # We use a composite key for the dropdown to ensure visit-level selection
            anom_visits = anomalies['visit_id'].unique()
            target_visit = st.selectbox("Investigate Visit #:", anom_visits)
            
            # Show the raw clinical context for that visit
            st.dataframe(df[df['visit_id'] == target_visit][['timestamp', 'event', 'step_duration']])
        else:
            st.success("No anomalies found in current triage selection.")

else:
    st.info("Please upload a CSV file.")