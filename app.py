import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# --- Page Configuration ---
# Set the title and icon that appear in the browser tab and app window
st.set_page_config(
    page_title="Translation Edit Dashboard",
    page_icon="�",
    layout="wide"
)

# --- App Title and Description ---
st.title("📊 Interactive Translation Edit Dashboard")
st.write(
    "This app analyzes translation edits from an uploaded report. "
    "Upload your CSV or Excel file to begin."
)

# --- File Uploader and Data Caching ---
# Use a file uploader to allow users to provide their own data
uploaded_file = st.file_uploader(
    "Choose a CSV or Excel file",
    type=['csv', 'xlsx']
)

# The @st.cache_data decorator is crucial for performance.
# It tells Streamlit to only run this function once for a given file.
@st.cache_data
def load_data(file):
    """Loads and cleans the data from the uploaded file."""
    # Check if the file is CSV or Excel and load accordingly
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # --- This is your existing data cleaning logic ---
    col_trans_dist = 'Edit Distance: from Translate to Dist. Review'
    col_dist_lsp = 'Edit Distance: from Dist. Review to LSP Review'
    date_col = 'Phase Timestamp: Translate'
    
    # Ensure the required columns exist
    for col in [col_trans_dist, col_dist_lsp, date_col, 'Target Locale']:
        if col not in df.columns:
            # If a column is missing, stop and show an error message
            st.error(f"Error: Missing required column '{col}' in the uploaded file.")
            st.stop() # Halts the app execution

    # Clean and convert the timestamp column to datetime objects
    df[date_col] = df[date_col].astype(str).replace('No Data', np.nan)
    df[date_col] = df[date_col].str.replace(r'\s[A-Z]{3,4}$', '', regex=True)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df.dropna(subset=[date_col], inplace=True)

    # Convert edit distance columns to numeric types
    df[col_trans_dist] = pd.to_numeric(df[col_trans_dist], errors='coerce').fillna(0)
    df[col_dist_lsp] = pd.to_numeric(df[col_dist_lsp], errors='coerce').fillna(0)
    
    return df

# --- Main App Logic ---
# Only proceed if a file has been uploaded
if uploaded_file is not None:
    # Load the data using our cached function
    df = load_data(uploaded_file)

    # --- UI Controls in the Sidebar ---
    st.sidebar.header("Dashboard Controls")
    
    # Timeframe selection using st.radio (replaces ToggleButtons)
    timeframe = st.sidebar.radio(
        "Select Period:",
        ('Quarterly', '6-Month', 'Annually'),
        index=2 # Default to 'Annually'
    )

    # Language selection using st.multiselect (replaces checkboxes)
    unique_locales = sorted(df['Target Locale'].unique().tolist())
    selected_languages = st.sidebar.multiselect(
        "Select Languages:",
        options=unique_locales,
        default=unique_locales # Default to all languages selected
    )

    # --- Filtering and Plotting Logic ---
    # Show a message if no languages are selected
    if not selected_languages:
        st.warning("Please select at least one language in the sidebar.")
    else:
        # This is your existing data filtering and preparation logic
        filtered_df = df[df['Target Locale'].isin(selected_languages)]

        dist_review_edits = filtered_df['Edit Distance: from Translate to Dist. Review'] > 0
        lsp_review_edits = filtered_df['Edit Distance: from Dist. Review to LSP Review'] > 0
        
        filtered_df['Distributor Review Changes'] = np.where(dist_review_edits, 1, 0)
        filtered_df['Post-Production Changes'] = np.where(lsp_review_edits, 1, 0)
        
        # Check if there's any data left to plot after filtering
        if filtered_df.empty or filtered_df[['Distributor Review Changes', 'Post-Production Changes']].sum().sum() == 0:
            st.info(f"No edit data found for the selected languages: {', '.join(selected_languages)}")
        else:
            # This is your revised grouping logic
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

            # --- Create and Display the Plot ---
            # Create the plotly figure (this is the same as before)
            fig = go.Figure(data=[
                go.Bar(name='Distributor Review Changes', x=x_labels, y=plot_data['Distributor Review Changes'], marker_color='#1f77b4'),
                go.Bar(name='Post-Production Changes (LSP)', x=x_labels, y=plot_data['Post-Production Changes'], marker_color='#ff7f0e')
            ])

            fig.update_layout(
                title_text=f'Edit Comparison for Selected Languages',
                barmode='group',
                xaxis_title='Time Period',
                yaxis_title='Number of Segments Edited',
                legend_title_text='Edit Stage'
            )

            # Use st.plotly_chart to display the figure in the app
            st.plotly_chart(fig, use_container_width=True)

else:
    # Show a placeholder message if no file is uploaded yet
    st.info("Awaiting file upload...")