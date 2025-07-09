import streamlit as st
import pandas as pd
import numpy as np
import joblib # To load your .pkl files

# --- 1. Load the Saved Model Components ---
try:
    loaded_model = joblib.load('final_breakdown_prediction_logistic_regression_model.pkl')
    loaded_label_encoder = joblib.load('final_label_encoder.pkl')
    loaded_preprocessor = joblib.load('final_preprocessor.pkl')
    st.success("✅ Model components loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model components: {e}")
    st.stop() # Stop the app if essential components can't be loaded

# --- 2. Define the Streamlit App Layout ---
st.set_page_config(page_title="DTC Bus Breakdown Predictor", layout="wide")
st.title("🚌 DTC Bus Breakdown Reason Predictor")
st.markdown("""
    Enter the bus details below to predict the most likely reason for its breakdown.
    This model uses historical DTC bus data to provide insights.
""")

# --- 3. Input Features from User ---
st.header("Bus & Breakdown Details")

# Organize inputs into columns or sections for better UX
col1, col2, col3 = st.columns(3)

with col1:
    depot_name = st.selectbox("Depot Name", options=['SNPD', 'BKD', 'INDPRST', 'TEHKHAND', ' ']) # Add relevant depot names
    bus_no = st.text_input("Bus No.", value="9200")
    route_no = st.text_input("Route No.", value="R101")
    breakdown_location = st.text_input("Breakdown Location", value="Mehrauli (T)")

with col2:
    sch_kms = st.number_input("Scheduled KMs", value=15000, min_value=0)
    act_kms = st.number_input("Actual KMs", value=14500, min_value=0)
    miss_kms_bd_only = st.number_input("Missed KMs due to BD only", value=500, min_value=0)
    other_reasons_kms = st.number_input("Other Reasons (KMs)", value=100, min_value=0)

with col3:
    repeated_single_instance = st.selectbox("Repeated or Single Instance", options=[0, 1], format_func=lambda x: "Single" if x == 0 else "Repeated")
    # Set default date to today for convenience
    breakdown_date = st.date_input("Breakdown Date", value=pd.to_datetime('today'))
    breakdown_time_str = st.time_input("Breakdown Time", value=pd.to_datetime("08:00:00").time())
    put_on_road_time_str = st.time_input("Put-on Road Time", value=pd.to_datetime("11:30:00").time())


# --- Derived Features (Changed/Fixed here) ---
# Ensure these derivations match *exactly* how they were done during training.
year = breakdown_date.year
month = breakdown_date.month
day = breakdown_date.day
dayofweek = breakdown_date.weekday() # Monday=0, Sunday=6

# Convert breakdown_date to a pandas Timestamp for easy extraction of weekofyear and quarter
# This fixes the AttributeError: 'Series' object has no attribute 'quarter'
temp_timestamp = pd.Timestamp(breakdown_date)
weekofyear = temp_timestamp.isocalendar().week
quarter = temp_timestamp.quarter

kms_difference = act_kms - sch_kms

# Convert time objects to full datetime for Time_to_Repair_Hours calculation
bd_datetime = pd.to_datetime(f"{breakdown_date} {breakdown_time_str}")
put_on_road_datetime = pd.to_datetime(f"{breakdown_date} {put_on_road_time_str}")

# Handle cases where put_on_road_time might be on the next day
if put_on_road_datetime < bd_datetime:
    put_on_road_datetime += pd.Timedelta(days=1)

time_to_repair_hours = (put_on_road_datetime - bd_datetime).total_seconds() / 3600
if time_to_repair_hours < 0: # Should not happen if timedelta logic is correct
    time_to_repair_hours = 0.1 # Small positive value to avoid issues

# Placeholder for cumulative breakdowns and days since last - these are tricky for single prediction
# For a live app, you'd need a database of past breakdowns for each bus.
# For now, use median/mode from training data or reasonable defaults for prediction context.
# Or exclude if your model can work without them or if they cause issues.
# For initial deployment, using a median/mode value from your training data for these is a common workaround.
# You'll need to know the median values you used for imputation during training.
# Let's assume you found median Breakdowns_Per_Bus_Cumulative and Days_Since_Last_Breakdown during training.
# Removed these as per the 'FINAL-FINAL REVISION' in your notebook
# breakdowns_per_bus_cumulative = 1 # Default, or median from training data
# days_since_last_breakdown = 7 # Default, or median from training data

# Breakdown Severity (Sch.KMs could be 0, handle division by zero)
breakdown_severity = miss_kms_bd_only / sch_kms if sch_kms > 0 else 0

# DayName and Breakdown_Time_Category
day_name = breakdown_date.strftime('%A') # e.g., 'Tuesday'

def get_time_category(time_obj):
    hour = time_obj.hour
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'
breakdown_time_category = get_time_category(breakdown_time_str)

# Corrective Action Taken and BD reported - assumed to be 'STD' and 'Yes' if not given by user
corrective_action_taken = st.selectbox("Corrective action taken by DMs", options=['STD', 'Minor Repair', 'Major Repair', 'Tow-away', 'Other']) # Add options from your data
bd_reported = st.selectbox("BD reported to control?", options=['Yes', 'No'])

# --- 4. Prepare Input for Prediction ---
# Create a DataFrame matching the EXACT structure of X_df from your training notebook
# (before the preprocessor was applied).
# This means it should contain both numerical and categorical columns as they were originally.

# Use your `numerical_cols` and `categorical_features_in_X_df` lists as a reference
# from where you defined them in your notebook (Step 6 "FINAL-FINAL REVISION")

# These were the lists from your notebook:
# numerical_cols = [
#     'Sch.KMs', 'Act.KMs', 'Miss KMs due to BD only', 'Other Reasons',
#     'Repeated or single instance during the week', 'Year', 'Month', 'Day',
#     'DayOfWeek', 'WeekOfYear', 'Quarter', 'KMs_Difference',
#     'Time_to_Repair_Hours', 'Breakdown_Severity'
# ]
# categorical_cols_for_input = [ # These were the ones actually fed to OHE in X_df
#     'Depot Name', 'Bus No.', 'Route No.', 'Breakdown location',
#     'Corrective action taken by DMs',
#     'BD whether reported to Depot Control/Regional Contrl/CCR',
#     'DayName', 'Breakdown_Time_Category'
# ]

# Reconstruct the input dataframe with all necessary features in the correct order/names
input_data = pd.DataFrame({
    'Sch.KMs': [sch_kms],
    'Act.KMs': [act_kms],
    'Miss KMs due to BD only': [miss_kms_bd_only],
    'Other Reasons': [other_reasons_kms],
    'Repeated or single instance during the week': [repeated_single_instance],
    'Year': [year],
    'Month': [month],
    'Day': [day],
    'DayOfWeek': [dayofweek],
    'WeekOfYear': [weekofyear],
    'Quarter': [quarter],
    'KMs_Difference': [kms_difference],
    'Time_to_Repair_Hours': [time_to_repair_hours],
    'Breakdown_Severity': [breakdown_severity],
    'Depot Name': [depot_name],
    'Bus No.': [bus_no],
    'Route No.': [route_no],
    'Breakdown location': [breakdown_location],
    'Corrective action taken by DMs': [corrective_action_taken],
    'BD whether reported to Depot Control/Regional Contrl/CCR': [bd_reported],
    'DayName': [day_name],
    'Breakdown_Time_Category': [breakdown_time_category]
})

# --- 5. Make Prediction ---
if st.button("Predict Breakdown Reason"):
    try:
        # Preprocess the input data
        processed_input = loaded_preprocessor.transform(input_data)

        # Make prediction
        predicted_label_encoded = loaded_model.predict(processed_input)
        predicted_reason = loaded_label_encoder.inverse_transform(predicted_label_encoded)[0]

        # Get prediction probabilities
        predicted_proba = loaded_model.predict_proba(processed_input)[0]
        top_n = 5 # Show top 5 reasons
        top_proba_indices = np.argsort(predicted_proba)[::-1][:top_n]
        top_proba_values = predicted_proba[top_proba_indices]
        top_proba_classes = loaded_label_encoder.inverse_transform(top_proba_indices)

        st.subheader("Prediction Result:")
        st.success(f"The most likely breakdown reason is: **{predicted_reason}**")

        st.subheader("Top Predicted Reasons & Probabilities:")
        proba_df = pd.DataFrame({
            "Reason": top_proba_classes,
            "Probability": top_proba_values
        })
        st.dataframe(proba_df.style.format({"Probability": "{:.2%}"}), hide_index=True)

    except ValueError as ve:
        st.error(f"Input Data Error: {ve}. Please check if all required fields are filled correctly and match the format expected by the model.")
        st.info("Tip: Ensure your input data columns and their types match what the model was trained on.")
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")
        st.warning("Please check your input values and try again. If the problem persists, the issue might be with the model or preprocessor.")

st.markdown("---")
st.caption("Developed by You for DTC Bus Operations")
