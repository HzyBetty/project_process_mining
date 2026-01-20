# ED Triage Lead Decision Support Tool

**Project Overview**  
This repository contains the ED Triage Lead Decision Support Tool, an AI-assisted process mining and predictive analytics application designed for Emergency Department (ED) operations. The tool transforms raw event logs into actionable, real-time insights, helping triage leads optimize patient flow, identify bottlenecks, monitor protocol compliance, and simulate the impact of staffing or process changes on patient length of stay (LOS).

The tool was developed as part of a Master of Management Analytics (MMA) program project and integrates Python, Streamlit, Plotly, machine learning, and network analysis to support decision-making in high-pressure clinical environments.

---

## 🎯 Key Features

1. **Flow Discovery & Load Balancing**
   - Interactive Directly-Follows Graph (DFG) visualizing patient pathways.
   - Identification of top bottlenecks and high-volume transitions.
   - Zone heatmaps to highlight patient accumulation and resource strain.

2. **Safety & Protocol Conformance**
   - Tracks compliance against the clinical gold standard protocol: Triage → Registration → Assessment → Discharge.
   - Flags sequence violations or missing steps that could indicate safety risks.

3. **Predictive Analytics**
   - Machine learning model predicts the probability of hospital admission.
   - Identifies key drivers (e.g., age, triage level, arrival time) influencing admissions.

4. **Capacity Simulation & “What-If” Planning**
   - Monte Carlo simulations project LOS under hypothetical efficiency improvements.
   - Supports proactive resource allocation and operational planning.

5. **Red Flag Alerts**
   - Highlights patients experiencing unusually long waits.
   - Allows deep-dive analysis for individual patient flow and delays.

---

## 🧑‍💻 Target User Persona

**Triage Lead** – responsible for managing patient inflow, prioritizing assessments, and preventing overcrowding.  
- Non-technical and time-constrained.  
- Needs actionable insights and visual dashboards rather than raw data.  
- Uses the tool for real-time decision-making and operational planning.

---

## ⚙️ Technical Requirements

- **Python Version:** 3.9+
- **Framework:** Streamlit
- **Key Libraries:** pandas, numpy, plotly, scikit-learn, networkx
- **Data:** `.csv` file containing the following columns:
  - `patient_id`, `timestamp`, `event`, `triage_code`, `initial_zone`, `disposition_desc`

See `requirements.txt` for full dependency list.  

---

## 🚀 Installation & Usage

1. Clone the repository:
```bash
git clone https://github.com/HzyBetty/project_process_mining.git
cd project_process_mining
```
2. Create a virtual environment:
```bash
py -3.11 -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```
3. Install dependencies:
```bash   
pip install -r requirements.txt
```
4. Lauch the app
```bash 
streamlit run app.py
```
5. Upload your ED event log CSV via the sidebar to activate the dashboard.

## Dashboard Overview

- **Global Filters (Sidebar):** Select triage levels, upload data, and view urgency legend.  
- **Flow Map:** Interactive visualization of patient transitions and bottlenecks.  
- **Protocol Conformance:** Compliance rate and flagged deviations from the standard process.  
- **Queue & Load Balancing:** Average wait time per zone and acuity-to-zone distribution.  
- **Predictive Admission Analytics:** Feature importance and probability predictions.  
- **Capacity Simulation:** "What-If" scenario slider showing potential LOS improvements.  
- **Red Flag Alerts:** Patient-level investigation of extreme delays.  

---

## 📁 Data Disclaimer

Due to privacy and size constraints, the ED event log is **not included** in this repository.  
You can test the application with your own anonymized ED event log formatted according to the technical requirements.

---

## 📝 References

- MMA Program, University of Toronto – Process Mining & Predictive Analytics coursework.  
- Streamlit & Plotly documentation.  
- Python libraries: pandas, numpy, scikit-learn, networkx.  

---

## ✅ Notes

- The application is designed for demonstration and portfolio purposes.  
- For real hospital deployment, further validation, security measures, and integration with clinical IT systems are required.



