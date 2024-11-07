#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import shutil
import pandas as pd
from datetime import datetime
import streamlit as st
#from tkinterdnd2 import TkinterDnD, DND_FILES
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from tkinter import messagebox

# Set a random seed for reproducibility
def set_random_seed(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)

# Define the main folder path
MAINFOLDER = r"D:\APPdata"

# Create other paths relative to the main folder
training_file_path = os.path.join(MAINFOLDER, "Training", "Training.xlsx")  # FIXED TRAINING DATA
test_file_path = os.path.join(MAINFOLDER, "24hrData", "Dailydata.xlsx")  # DAILY DATA
excel_file_path = os.path.join(MAINFOLDER, "Breakdownrecords.xlsx")  # Recording excel for BD
folderpath = os.path.join(MAINFOLDER, "TemporaryData")  # Temporary dump files collector

uploaded_files = []  # List to keep track of uploaded files

# Streamlit UI
st.title("File Upload and Preprocessing")
st.markdown("Upload your files, and they will be preprocessed accordingly.")

# File Upload Section
uploaded_files = st.file_uploader("Upload Excel files", type=['xlsx'], accept_multiple_files=True)

# Show status
status_placeholder = st.empty()

# Function to clear old files from the folder
def clear_saved_files():
    try:
        # Clear old files in the folder
        for filename in os.listdir(folderpath):
            file_path = os.path.join(folderpath, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)  # Remove the file
            except Exception as e:
                status_placeholder.error(f"Error clearing old files: {e}")
                return
        status_placeholder.success("Saved files cleared successfully!")
    except Exception as e:
        status_placeholder.error(f"Error: {e}")

# Function to handle file saving (clear old files before saving new ones)
def save_files(uploaded_files):
    try:
        if not uploaded_files:
            status_placeholder.error("No files to save!")
            return

        # Clear old files in the folder before saving new files
        clear_saved_files()

        # Save each file from the uploaded list to the target folder
        for file in uploaded_files:
            with open(os.path.join(folderpath, file.name), "wb") as f:
                f.write(file.getbuffer())

        status_placeholder.success("Files saved successfully!")

    except Exception as e:
        status_placeholder.error(f"Error: {e}")

# Function to clear uploaded files list
def clear_uploaded_files():
    global uploaded_files
    uploaded_files = []  # Clear the list
    status_placeholder.success("Uploaded files list cleared!")

# Function to remove the last uploaded file from the list
def clear_last_uploaded_file():
    global uploaded_files
    if uploaded_files:
        uploaded_files.pop()  # Remove the last file from the list
        status_placeholder.success("Last uploaded file removed!")
    else:
        status_placeholder.warning("No files to remove!")

# Process files and apply preprocessing logic
def preprocess_files():
    try:
        # Step 1: Get all Excel files in the folder
        excel_files = [f for f in os.listdir(folderpath) if f.endswith('.xlsx')]

        # Step 2: Loop through each Excel file and preprocess
        for file in excel_files:
            file_path = os.path.join(folderpath, file)
            df = pd.read_excel(file_path)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns

            # Check for duplicate column names and give a warning if found
            duplicate_columns = df.columns[df.columns.duplicated()].tolist()
            if duplicate_columns:
                st.warning(f"File: {file}\nDuplicate columns found: {', '.join(duplicate_columns)}")

            # Step 4: Remove specific unnecessary columns
            columns_to_remove = [
                'plant_name', 'area_name', 'equipment_name', 'measurement_location_name',
                'avg_Vertical_velocity', 'avg_Axial_velocity', 'avg_Horizontal_velocity',
                'avg_total_acceleration', 'avg_audio', 'avg_temperature',
                'min_total_acceleration', 'min_Vertical_velocity', 'min_Axial_velocity',
                'min_Horizontal_velocity', 'min_temperature', 'min_audio'
            ]
            df.drop(columns=columns_to_remove, inplace=True, errors='ignore')

            # Step 5: Rename columns
            column_rename_map = {
                'max_total_acceleration': 'tot_acc',
                'max_Vertical_velocity': 'ver_vel',
                'max_Axial_velocity': 'ax_vel',
                'max_Horizontal_velocity': 'hor_vel',
                'max_temperature': 'temp',
                'max_audio': 'aud'
            }
            df.rename(columns=column_rename_map, inplace=True)

            # Step 6: Handle missing values for 'asset_name'
            if 'asset_name' in df.columns:
                df['asset_name'].fillna(method='ffill', inplace=True)
                df['asset_name'].fillna(method='bfill', inplace=True)

            # Step 7: Convert 'time' to datetime and remove timezone
            df['time'] = pd.to_datetime(df['time'], errors='coerce').dt.tz_localize(None)
            date_for_file = df['time'].dt.date.iloc[0]

            # Drop duplicates based on 'time' and keep the first occurrence
            df.drop_duplicates(subset='time', inplace=True)

            # Check for duplicate index values and show a warning
            if df['time'].duplicated().any():
                duplicate_rows = df[df['time'].duplicated()]
                warning_message = (
                    f"File: {file}\n"
                    f"Duplicate 'time' values found in rows:\n"
                    f"{duplicate_rows[['time']].to_string(index=False)}"
                )
                st.warning(warning_message)

            # Create a minute-wise time range
            start_time = pd.Timestamp(f"{date_for_file} 00:00:00")
            end_time = pd.Timestamp(f"{date_for_file} 23:59:00")
            full_time_range = pd.date_range(start=start_time, end=end_time, freq='T')

            # Set 'time' as the index and reindex with the full time range
            df.set_index('time', inplace=True)
            df = df.reindex(full_time_range)
            df.index.name = 'time'

            # Fill missing values
            non_numeric_cols = df.select_dtypes(exclude=['number']).ffill().bfill()
            numeric_cols = df.select_dtypes(include=['number']).fillna(0)
            resampled_numeric = numeric_cols.resample('10T', label='left', closed='left').max()
            resampled_non_numeric = non_numeric_cols.resample('10T').ffill()
            resampled_df = pd.concat([resampled_numeric, resampled_non_numeric], axis=1)
            resampled_df.reset_index(inplace=True)

            # Format 'time' to 'Date' and 'Time'
            resampled_df['Date'] = resampled_df['time'].dt.strftime('%d %b %Y')
            resampled_df['Time'] = resampled_df['time'].dt.strftime('%I:%M %p')
            resampled_df.insert(0, 'Sr No', range(1, len(resampled_df) + 1))
            ordered_columns = ['Date', 'Time', 'Sr No', 'tot_acc', 'ver_vel', 'ax_vel', 'hor_vel', 'temp', 'aud', 'asset_name']
            resampled_df = resampled_df[[col for col in ordered_columns if col in resampled_df.columns]]

            # Save the processed data back to the same file
            with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
                resampled_df.to_excel(writer, index=False)

        # Combining files
        asset_order = [
            "Grinding Machine 1 Gearbox", "Grinding Machine-1 Motor", "Grinding Machine 2 Gearbox",
            "Grinding Machine-2 Motor", "Grinding Machine 3 Gearbox", "Grinding Machine-3 Motor",
            "Grinding Machine 4 Gearbox", "Grinding Machine-4 Motor", "Grinding Machine 5 Gearbox",
            "Grinding Machine-5 Motor", "Grinding Machine 6 Gearbox", "Grinding Machine-6 Motor"
        ]

        combined_df = pd.DataFrame()
        for asset in asset_order:
            for file in excel_files:
                file_path = os.path.join(folderpath, file)
                try:
                    df = pd.read_excel(file_path)
                    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                except Exception as e:
                    continue

                asset_name_values = df['asset_name'].iloc[1:3].values
                if any(asset.strip() == value.strip() for value in asset_name_values):
                    df = df.drop(columns=['asset_name'], errors='ignore')
                    if combined_df.empty:
                        common_cols = ['Date', 'Time', 'Sr No']
                        combined_df = df[common_cols].copy()

                    df = df.drop(columns=['Date', 'Time', 'Sr No'], errors='ignore')
                    combined_df = pd.concat([combined_df, df], axis=1)

        if not combined_df.empty:
            combined_df.fillna(0, inplace=True)
            combined_df['Code'] = ''
            with pd.ExcelWriter(test_file_path, engine='openpyxl') as writer:
                combined_df.to_excel(writer, index=False)
            st.success("All files processed and combined successfully!")
        else:
            st.error("No data found or processed. Please check files and asset names.")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

## Streamlit buttons and actions
#if st.button("Clear Uploaded Files"):
#    clear_uploaded_files()
#
#if st.button("Clear Last Uploaded File"):
#    clear_last_uploaded_file()

if st.button("Save Files"):
    if uploaded_files:
        save_files(uploaded_files)
    else:
        st.error("Please upload files first.")

if st.button("Preprocess Files"):
    preprocess_files()





################breakdown records###########################


import streamlit as st
import pandas as pd
from datetime import datetime

# Path to the Excel file
#excel_file_path = "breakdown_data.xlsx"

# Function to save breakdown data to Excel
def save_breakdown_data():
    date = st.session_state.date_entry.strftime("%d-%m-%y")
    time = f"{st.session_state.hour_combobox}:{st.session_state.minute_combobox} {st.session_state.am_pm_combobox}"
    code = st.session_state.code_entry
    
    if not code:
        st.session_state.status = "Please fill the Breakdown Code!"
        st.session_state.status_color = "red"
        return

    try:
        df = pd.read_excel(excel_file_path)
        new_row = pd.DataFrame([[date, time, code]], columns=["Date", "Time", "Code"])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(excel_file_path, index=False)
        st.session_state.status = "Breakdown data saved successfully!"
        st.session_state.status_color = "green"
    except Exception as e:
        st.session_state.status = f"Error: {e}"
        st.session_state.status_color = "red"

# Function to clear the breakdown input fields
def clear_breakdown_fields():
    st.session_state.date_entry = datetime.now()
    st.session_state.hour_combobox = '12'
    st.session_state.minute_combobox = '00'
    st.session_state.am_pm_combobox = 'AM'
    st.session_state.code_entry = ''
    st.session_state.status = "Fields cleared!"
    st.session_state.status_color = "blue"

# Streamlit UI Setup
def display_ui():
    # Initialize session state if not already initialized
    if 'status' not in st.session_state:
        st.session_state.status = ""
        st.session_state.status_color = "black"
        st.session_state.date_entry = datetime.now()
        st.session_state.hour_combobox = '12'
        st.session_state.minute_combobox = '00'
        st.session_state.am_pm_combobox = 'AM'
        st.session_state.code_entry = ''

    st.title("Breakdown Record")

    # Date input
    st.session_state.date_entry = st.date_input("Date", value=st.session_state.date_entry)
    
    # Time selection
    time_column1, time_column2, time_column3 = st.columns(3)
    with time_column1:
        st.session_state.hour_combobox = st.selectbox("Hour", options=[f"{i:02d}" for i in range(1, 13)], index=int(st.session_state.hour_combobox)-1)
    with time_column2:
        st.session_state.minute_combobox = st.selectbox("Minute", options=[f"{i:02d}" for i in range(0, 60, 5)], index=int(st.session_state.minute_combobox)//5)
    with time_column3:
        st.session_state.am_pm_combobox = st.selectbox("AM/PM", options=["AM", "PM"], index=["AM", "PM"].index(st.session_state.am_pm_combobox))

    # Breakdown code input
    st.session_state.code_entry = st.text_input("Breakdown Code", value=st.session_state.code_entry)

    # Status display (Feedback to user)
    st.markdown(f"<p style='color:{st.session_state.status_color};'>{st.session_state.status}</p>", unsafe_allow_html=True)

    # Buttons for saving and clearing
    col1, col2 = st.columns(2)
    with col1:
        save_button = st.button("Save Breakdown")
        if save_button:
            save_breakdown_data()
    with col2:
        clear_button = st.button("Clear Fields")
        if clear_button:
            clear_breakdown_fields()

# Run the UI display
if __name__ == "__main__":
    display_ui()



################################        time prediction             #############################


import os
import pandas as pd
import joblib
import streamlit as st
from datetime import datetime
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# Define the path to save models within the main folder
model_folder_path = os.path.join(MAINFOLDER,"Models")

# Paths to the training and testing data files
#training_file_path = "training_file_path.xlsx"  # Path to your training data file
#test_file_path = "test_file_path.xlsx"  # Path to your test data file

# Set random seed for reproducibility
def set_random_seed():
    import numpy as np
    import tensorflow as tf
    np.random.seed(42)
    tf.random.set_seed(42)

# Function for time prediction
def predict_time():
    def load_test_data(file_path):
        df = pd.read_excel(file_path)
        serial_numbers = df.iloc[:, 2].values
        times = df.iloc[:, 1].values
        X_test = df.iloc[:, 3:74].values
        return df, X_test, serial_numbers, times

    def preprocess_test_data(X_test):
        scaler = joblib.load(os.path.join(model_folder_path, 'scalerfinT.pkl'))
        X_test_scaled = scaler.transform(X_test)
        return X_test_scaled

    def predict_time_to_breakdown(X_test_scaled):
        model = load_model(os.path.join(model_folder_path, 'trained_modelFINT.h5'))
        predictions = model.predict(X_test_scaled)
        return predictions

    def calculate_time_difference(times, predictions):
        time_to_breakdown_with_time = []
        for time_str, prediction in zip(times, predictions):
            time_obj = datetime.strptime(time_str, '%I:%M %p')
            midnight = datetime.combine(time_obj.date(), datetime.min.time())
            time_difference = (time_obj - midnight).total_seconds() / 3600
            adjusted_time_to_bd = prediction[0] + time_difference
            time_to_breakdown_with_time.append(adjusted_time_to_bd)
        return time_to_breakdown_with_time

    def find_minimum_time(time_to_breakdown_with_time):
        min_time = min(time_to_breakdown_with_time)
        return min_time

    # Set random seed for reproducibility
    set_random_seed()

    try:
        df, X_test, serial_numbers, times = load_test_data(test_file_path)  # Use predefined path
        X_test_scaled = preprocess_test_data(X_test)
        predictions = predict_time_to_breakdown(X_test_scaled)
        predictions_with_time = calculate_time_difference(times, predictions)
        min_time = find_minimum_time(predictions_with_time)

        # Display the result in Streamlit
        st.session_state.result = f"Minimum Time to Breakdown: {min_time:.2f} hours"
        st.session_state.status = "Prediction complete!"
        st.session_state.status_color = "green"
    except Exception as e:
        st.session_state.result = f"Error: {e}"
        st.session_state.status = "Prediction failed!"
        st.session_state.status_color = "red"

# Streamlit UI Setup
def display_ui():
    # Initialize session state if not already initialized
    if 'status' not in st.session_state:
        st.session_state.status = ""
        st.session_state.status_color = "black"
        st.session_state.result = ""

    st.title("Time Prediction for Breakdown")

    # Button to trigger time prediction
    predict_button = st.button("Predict Time to Breakdown")
    if predict_button:
        st.session_state.status = "Predicting..."
        st.session_state.status_color = "blue"
        predict_time()

    # Display status message
    st.markdown(f"<p style='color:{st.session_state.status_color};'>{st.session_state.status}</p>", unsafe_allow_html=True)

    # Display result
    if st.session_state.result:
        st.write(st.session_state.result)

# Run the UI display
if __name__ == "__main__":
    display_ui()

