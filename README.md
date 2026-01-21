# 🏥 ED Triage Lead Decision Support Tool

**Project Overview** This repository contains the **ED Triage Lead Decision Support Tool**, an AI-assisted process mining and predictive analytics application designed specifically for Emergency Department (ED) operations. The tool transforms raw, noisy event logs into actionable, real-time insights—empowering Triage Leads to optimize patient flow, monitor safety protocols, and simulate operational changes.

Developed as part of a **Master of Management Analytics (MMA)** program, this tool integrates Python, Streamlit, and advanced network analysis to solve the "Black Box" of ED throughput.

---

## 🧠 Core Methodology: Visit-Atomic Isolation
Unlike traditional analytics that may be skewed by repeat visitors or system errors, this tool features a custom **Isolation Engine**:
* **The 24-Hour Rule:** Automatically partitions patient data into distinct clinical encounters if a gap of 24+ hours is detected.
* **Start-Event Enforcement:** Forces a new "Visit ID" upon arrival events (e.g., Ambulance Arrival, Triage), eliminating "ghost transitions" (e.g., linking a discharge from last month to a triage from today) and ensuring bottleneck calculations are accurate to the minute.

---

## 🎯 Key Features

### 1. Dynamic Flow Discovery & Volume Slicing
* **Directly-Follows Graph (DFG):** Visualizes the clinical journey with **variable line thickness** mapped to patient volume.
* **Interactive Volume Slicer:** Allows users to filter out rare deviations to focus on the "Happy Path" or high-traffic bottlenecks.


### 2. Temporal Efficiency Heatmap
* **Hourly Delay Analysis:** A 24-hour heatmap identifying peak congestion windows for specific process handoffs.
* **Business Impact:** Provides a data-driven basis for "surge staffing" and precision resource allocation during identified "Red Zones."


### 3. Protocol Conformance & "The Golden Rule"
* **Sequence Tracking:** Monitors compliance against the clinical gold standard: **Triage → Registration → Assessment → Discharge**.
* **Relative Order Logic:** Features a flexible algorithm that validates the chronological order of milestones, identifying safety risks or documentation gaps without failing visits for "parallel processing" (e.g., bedside registration).


### 4. Predictive Admission Analytics
* **Random Forest Engine:** Predicts the probability of hospital admission based on initial triage metrics.
* **Resource Forecasting:** Identifies key drivers influencing admissions to help inpatient wards prepare for incoming surges.

### 5. Monte Carlo Capacity Simulation
* **Stochastic Forecasting:** Uses **Monte Carlo Logic** to run 1,000 simulated visits based on the actual historical distribution (Mean and Standard Deviation) of the ED's Length of Stay (LOS).
* **What-If Analysis:** Features an improvement slider to project how efficiency gains shift the LOS bell curve and reduce "Tail Risk" (dangerously long stays).


### 6. Red Flag Alerts
* **Anomaly Detection:** Automatically flags "Statistical Outliers" (delays exceeding Mean + 2SD).
* **Root Cause Deep-Dive:** Allows Triage Leads to inspect the specific timeline of delayed visits for operational auditing.

---

## 🧑‍💻 Target User Persona

**Triage Lead** – Responsible for managing patient inflow and preventing overcrowding.  
* **Need:** Actionable insights that translate "Data Science" into "Bed Management."
* **Goal:** Minimize "Left Without Being Seen" (LWBS) rates and optimize clinical throughput.

---

## ⚙️ Technical Requirements

* **Python Version:** 3.9+
* **Framework:** Streamlit
* **Key Libraries:** `pandas`, `numpy`, `plotly`, `scikit-learn`, `networkx`, `scipy`
* **Data Structure:** `.csv` file with columns: `patient_id`, `timestamp`, `event`, `triage_code`, `initial_zone`, `disposition_desc`, `age`, `gender`.

---

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/HzyBetty/project_process_mining.git](https://github.com/HzyBetty/project_process_mining.git)
   cd project_process_mining
   ```
2. **Create a virtual environment:**
   ```bash
   py -3.11 -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Launch the App:**
   ```bash
   streamlit run app.py
   ```
5. **Upload Data:** Use the sidebar to upload an ED log CSV and utilize the Volume Slicer to adjust the Flow Map.

---

## 📊 Dashboard Functionality: Six-Module Analysis

The application is divided into six strategic modules that allow a Triage Lead to move from high-level oversight to granular patient-level investigation:

1.  **Process Discovery & Flow Mapping:** Uses a Directly-Follows Graph (DFG) to visualize how patients move through the ED. Includes a **Volume Slicer** to identify the most common pathways versus rare deviations.
2.  **Safety Protocol Conformance:** Compares actual patient journeys against the "Golden Rule" clinical sequence. It uses relative ordering to calculate a **Compliance Rate**, identifying where protocols are being bypassed.
3.  **Real-Time Distribution:** Provides an immediate breakdown of patient volume across different hospital **Initial Zones**, helping leads identify which physical areas are currently over-capacity.
4.  **Admission Predictive Analytics:** Employs a **Random Forest Machine Learning** model to predict the likelihood of hospital admission based on arrival data, enabling early bed-request notifications.
5.  **Monte Carlo Capacity Simulation:** A "What-If" tool that uses stochastic modeling to project how process improvements (e.g., 10% faster triage) would shift the overall **Length of Stay (LOS)** distribution.
6.  **Red Flag Alert System:** Automatically flags visits that are **Statistical Outliers** (Mean + 2SD). This allows for a deep-dive audit of individual cases that experienced extreme delays.

---

## 📁 Data Disclaimer
Due to PHI (Protected Health Information) privacy constraints, the original event log is not included in this repository. Users may test the application using their own anonymized ED event log formatted according to the technical schema provided above.

---

## ✅ Notes
* **Purpose:** This tool is designed for demonstration and MMA portfolio purposes to showcase the intersection of process mining, data engineering, and clinical decision support.
* **Business Impact:** By identifying even a modest 10% efficiency gain via the Monte Carlo simulation, an average ED could potentially reduce total patient "wait-hours" by hundreds of hours per week, directly impacting LWBS (Left Without Being Seen) rates and patient satisfaction.