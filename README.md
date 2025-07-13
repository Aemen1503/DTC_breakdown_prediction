# 🚍 DTC Breakdown Prediction

This project is a **Machine Learning web application** that predicts the most likely reason behind a bus breakdown based on Diagnostic Trouble Codes (DTCs). It was built to assist maintenance teams in identifying the cause of faults more efficiently, helping reduce vehicle downtime and enable proactive maintenance.

🔗 **Live Streamlit App**: [Click here to try the app](https://dtcbreakdownprediction-fwryxwzv8mctxmvumtkaav.streamlit.app/)  
📁 **GitHub Repository**: https://github.com/Aemen1503/DTC_breakdown_prediction

---

## 📌 Project Overview

Breakdowns in buses often occur due to various mechanical or electrical faults, which are logged using standardized **DTCs**. This project uses these codes as input to a machine learning model trained on historical breakdown data to predict the **underlying reason** for a breakdown.

This solution is especially useful for:

- Fleet managers
- Vehicle maintenance teams
- Automotive engineers
- Transport departments

---

## 🔍 Features

- ✅ Predicts **breakdown reason** based on DTC input
- 🧠 Trained ML model with TF-IDF + Logistic Regression
- 📊 Option to input single or multiple DTC codes manually
- 📂 Supports batch predictions via CSV upload
- 🌐 Web-based UI using **Streamlit** for quick interaction
- 🔧 Displays prediction output clearly with model confidence

---

## 🧠 How It Works

1. **User Input**:
   - Manual DTC code entry or CSV file upload
2. **Preprocessing**:
   - Converts DTC codes into numerical vectors using TF-IDF
3. **Prediction**:
   - ML model predicts the most probable breakdown cause
4. **Output**:
   - Displays predicted reason with clear formatting

---

## 📁 Folder Structure

