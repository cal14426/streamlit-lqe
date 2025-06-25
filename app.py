import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Translation Edit Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- App Title and Description ---
st.title("📊 Interactive Translation Edit Dashboard")
st.write(
    "This app analyzes translation edits directly from your large, multi-sheet Excel report. "
    "Upload your Excel file to begin."
)

# --- File Uploader ---
# Now primarily looks for Excel files but still allows CSVs
uploaded_file = st.file_uploader(
    "Choose your large Excel report (.xlsx)",
    type=['xlsx', 'csv']
)

# The @st.cache_data decorator is crucial for performance.
# It tells Streamlit to only run this heavy processing function once for a given file.
@st.cache_data
def load_and_preprocess_data(file):
    """
    Loads data from an uploaded file, combines all sheets from an Excel file,
    filters for required columns, and performs cleaning.
    """
    # --- This is the new, integrated preprocessing logic ---
    required_columns = [
        'Target Locale',
        'Phase Timestamp: Translate',
        'Edit Distance: from Translate to Dist. Review',
        'Edit Distance: from Dist. Review to LSP Review'
    ]

    try:
        if file.name.endswith('.xlsx'):
            st.write("Log: Reading Excel file...")
            # Use a spinner to show the user that processing is happening
            with st.spinner("Reading all sheets from the large Excel file... This may take a moment."):
                # sheet_name=None reads all sheets into a dictionary of DataFrames
                all_sheets_df_dict = pd.read_excel(file, sheet_name=None)
            st.success("✅ Log: Successfully read all sheets from Excel.")
            
            with st.spinner("Combining sheets and cleaning data..."):
                # Combine all sheets into a single DataFrame
                df = pd.concat(all_sheets_df_dict.values(), ignore_index=True)
            st.success("✅ Log: Sheets combined.")
        else: # Handle CSV files as before
             st.write("Log: Reading CSV file...")
             df = pd.read_csv(file)
             st.success("✅ Log: CSV file loaded.")

    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
    
    # --- Check for required columns and filter ---
    st.write("Log: Checking for required columns...")
    available_columns = [col for col in required_columns if col in df.columns]
    if not available_columns:
        st.error(f"Error: None of the required columns were found in the file. Needed: {required_columns}")
        st.stop()
    st.write(f"Log: Found and kept the following columns: {available_columns}")
    df = df[available_columns]

    # --- This is your existing data cleaning logic ---
    st.write("Log: Cleaning date/time columns...")
    date_col = 'Phase Timestamp: Translate'
    df[date_col] = df[date_col].astype(str).replace('No Data', np.nan)
    df[date_col] = df[date_col].str.replace(r'\s[A-Z]{3,4}$', '', regex=True)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df.dropna(subset=[date_col], inplace=True)
    st.success("✅ Log: Date cleaning complete.")

    st.write("Log: Cleaning numeric edit distance columns...")
    for col in ['Edit Distance: from Translate to Dist. Review', 'Edit Distance: from Dist. Review to LSP Review']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    st.success("✅ Log: Numeric cleaning complete.")
    
    st.success("🎉 Preprocessing finished successfully!")
    return df

# --- Main App Logic ---
# Only proceed if a file has been uploaded
if uploaded_file is not None:
    # Load and preprocess the data using our cached function
    df = load_and_preprocess_data(uploaded_file)

    # --- UI Controls in Main Area ---
    st.header("Dashboard Controls")
    
    # Create two columns for controls
    col1, col2 = st.columns(2)
    
    with col1:
        timeframe = st.radio(
            "Select Period:",
            ('Quarterly', '6-Month', 'Annually'),
            index=2  # Default to 'Annually'
        )
    
    with col2:
        unique_locales = sorted(df['Target Locale'].unique().tolist())
        selected_languages = st.multiselect(
            "Select Languages:",
            options=unique_locales,
            default=unique_locales
        )

    # Add a separator
    st.divider()

    # --- Filtering and Plotting Logic ---
    if not selected_languages:
        st.warning("Please select at least one language above.")
    else:
        filtered_df = df[df['Target Locale'].isin(selected_languages)]

        dist_review_edits = filtered_df['Edit Distance: from Translate to Dist. Review'] > 0
        lsp_review_edits = filtered_df['Edit Distance: from Dist. Review to LSP Review'] > 0
        
        filtered_df['Distributor Review Changes'] = np.where(dist_review_edits, 1, 0)
        filtered_df['Post-Production Changes'] = np.where(lsp_review_edits, 1, 0)
        
        if filtered_df.empty or filtered_df[['Distributor Review Changes', 'Post-Production Changes']].sum().sum() == 0:
            st.info(f"No edit data found for the selected languages: {', '.join(selected_languages)}")
        else:
            date_col = 'Phase Timestamp: Translate'
            if timeframe == 'Quarterly':
                group_key = filtered_df[date_col].dt.year.astype(str) + '-Q' + filtered_df[date_col].dt.quarter.astype(str)
            elif timeframe == '6-Month':
                half = np.where(filtered_df[date_col].dt.month <= 6, 'H1', 'H2')
                group_key = filtered_df[date_col].dt.year.astype(str) + '-' + half
            else: # Annually
                group_key = filtered_df[date_col].dt.year
            
            plot_data = filtered_df.groupby(group_key)[['Distributor Review Changes', 'Post-Production Changes']].sum()
            x_labels = plot_data.index.astype(str)

            # Create the plotly figure
            st.write("Log: Generating chart...")
            fig = go.Figure(data=[
                go.Bar(name='Distributor Review Changes', x=x_labels, y=plot_data['Distributor Review Changes'], marker_color='#1f77b4'),
                go.Bar(name='Post-Production Changes (LSP)', x=x_labels, y=plot_data['Post-Production Changes'], marker_color='#ff7f0e')
            ])

            fig.update_layout(
                title_text='Edit Comparison for Selected Languages',
                barmode='group',
                xaxis_title='Time Period',
                yaxis_title='Number of Segments Edited',
                legend_title_text='Edit Stage'
            )
            st.success("✅ Log: Chart generation complete.")
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Awaiting Excel file upload...")

