import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# --- Page Configuration ---
# Set the title and icon that appear in the browser tab and app window
st.set_page_config(
    page_title="Translation Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Title and Description ---
st.title("📊 Translation Analysis Dashboard")
st.write(
    "This app provides tools for analyzing translation edits and quality metrics. "
    "Use the tabs below to navigate between the 'Edit Dashboard' and 'Quality Analysis' sections."
)

# --- Data Loading Functions ---
@st.cache_data
def load_edit_data(file):
    """Loads and cleans the data from the uploaded translation edit file."""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, low_memory=False)
        else:
            df = pd.read_excel(file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    col_trans_dist = 'Edit Distance: from Translate to Dist. Review'
    col_dist_lsp = 'Edit Distance: from Dist. Review to LSP Review'
    date_col = 'Phase Timestamp: Translate'
    
    required_cols = [col_trans_dist, col_dist_lsp, date_col, 'Target Locale', 'Project']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Error: Missing required column '{col}' in the uploaded file.")
            st.stop()
            
    df = df[required_cols]

    df[date_col] = pd.to_datetime(df[date_col].str.replace(r'\s[A-Z]{3,4}$', '', regex=True), errors='coerce')
    df.dropna(subset=[date_col], inplace=True)

    df[col_trans_dist] = pd.to_numeric(df[col_trans_dist], errors='coerce').fillna(0)
    df[col_dist_lsp] = pd.to_numeric(df[col_dist_lsp], errors='coerce').fillna(0)
    
    return df

@st.cache_data
def load_quality_data(file):
    """Loads and cleans data from the quality analysis file."""
    try:
        df = pd.read_csv(file, low_memory=False)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    required_cols = ['target_locale', 'category_name', 'severity_name', 'target_finalized_date']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Error: Missing one or more required columns. Please ensure the file contains: {', '.join(required_cols)}")
        st.stop()

    df = df[required_cols]
    df = df.rename(columns={'target_finalized_date': 'date'})
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    
    return df

# --- UI Tabs ---
tab1, tab2 = st.tabs(["Translation Edit Dashboard", "Quality Analysis"])

with tab1:
    st.header("Translation Edit Analysis")
    uploaded_file = st.file_uploader("Choose a translation edit CSV file", type=['csv'], key="edit_uploader")

    if uploaded_file is not None:
        df = load_edit_data(uploaded_file)

        # --- UI Controls ---
        st.subheader("Dashboard Controls")
        control_cols = st.columns(3)
        with control_cols[0]:
            timeframe = st.radio(
                "Select Period:",
                ('Quarterly', '6-Month', 'Annually'),
                index=2,
                key='edit_timeframe'
            )
        with control_cols[1]:
            unique_locales = sorted(df['Target Locale'].unique().tolist())
            selected_languages = st.multiselect(
                "Select Languages:",
                options=unique_locales,
                default=unique_locales
            )
        with control_cols[2]:
            unique_projects = sorted(df['Project'].unique().tolist())
            selected_projects = st.multiselect(
                "Select Projects:",
                options=unique_projects,
                default=unique_projects
            )

        # --- Filtering and Plotting Logic ---
        if not selected_languages or not selected_projects:
            st.warning("Please select at least one language and project.")
        else:
            filtered_df = df[df['Target Locale'].isin(selected_languages) & df['Project'].isin(selected_projects)]

            filtered_df['Distributor Review Changes'] = np.where(filtered_df['Edit Distance: from Translate to Dist. Review'] > 0, 1, 0)
            filtered_df['Post-Production Changes'] = np.where(filtered_df['Edit Distance: from Dist. Review to LSP Review'] > 0, 1, 0)
            
            if filtered_df.empty or filtered_df[['Distributor Review Changes', 'Post-Production Changes']].sum().sum() == 0:
                st.info("No edit data found for the selected criteria.")
            else:
                date_col = 'Phase Timestamp: Translate'
                if timeframe == 'Quarterly':
                    group_key = filtered_df[date_col].dt.to_period('Q').astype(str)
                elif timeframe == '6-Month':
                    half = np.where(filtered_df[date_col].dt.month <= 6, 'H1', 'H2')
                    group_key = filtered_df[date_col].dt.year.astype(str) + '-' + half
                else:
                    group_key = filtered_df[date_col].dt.year
                
                plot_data = filtered_df.groupby(group_key)[['Distributor Review Changes', 'Post-Production Changes']].sum()
                x_labels = plot_data.index.astype(str)

                fig = go.Figure(data=[
                    go.Bar(name='Distributor Review Changes', x=x_labels, y=plot_data['Distributor Review Changes'], marker_color='#1f77b4'),
                    go.Bar(name='Post-Production Changes (LSP)', x=x_labels, y=plot_data['Post-Production Changes'], marker_color='#ff7f0e')
                ])
                fig.update_layout(
                    title_text='Edit Comparison for Selected Languages and Projects',
                    barmode='group',
                    xaxis_title='Time Period',
                    yaxis_title='Number of Segments Edited',
                    legend_title_text='Edit Stage'
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Awaiting file upload for edit analysis...")

with tab2:
    st.header("Translation Quality Analysis")
    quality_file = st.file_uploader("Upload a quality report CSV file", type=['csv'], key="quality_uploader")

    if quality_file is not None:
        quality_df = load_quality_data(quality_file)
        
        # --- UI Controls ---
        st.subheader("Analysis Controls")
        q_control_cols = st.columns(2)
        with q_control_cols[0]:
            q_timeframe = st.radio(
                "Select Period for Time Series:",
                ('Quarterly', '6-Month', 'Annually'),
                index=2,
                key='quality_timeframe'
            )
        with q_control_cols[1]:
            q_unique_locales = sorted(quality_df['target_locale'].unique().tolist())
            q_selected_languages = st.multiselect(
                "Select Languages:",
                options=q_unique_locales,
                default=q_unique_locales,
                key='quality_languages'
            )

        if not q_selected_languages:
            st.warning("Please select at least one language.")
        else:
            filtered_q_df = quality_df[quality_df['target_locale'].isin(q_selected_languages)].copy()

            if filtered_q_df.empty:
                st.info("No data available for the selected languages.")
            else:
                if 'severity_name' in filtered_q_df.columns:
                    filtered_q_df['severity_name'] = filtered_q_df['severity_name'].str.lower()
                # --- Pie Chart: Category Distribution ---
                st.subheader("Distribution of Issue Categories")
                category_counts = filtered_q_df['category_name'].value_counts()
                pie_fig = go.Figure(data=[go.Pie(labels=category_counts.index, values=category_counts.values, hole=.3)])
                pie_fig.update_layout(title_text='Overall Issue Category Distribution')
                st.plotly_chart(pie_fig, use_container_width=True)

                # --- Bar Chart: Severity per Category ---
                st.subheader("Issue Severity by Category")
                severity_counts = filtered_q_df.groupby(['category_name', 'severity_name']).size().unstack(fill_value=0)
                
                color_map = {'critical': 'red', 'major': 'yellow', 'minor': 'blue'}
                
                # Define the order for stacking
                severity_order = ['minor', 'major', 'critical']
                
                # Get available severities from the dataframe columns, respecting the desired order
                available_severities = [s for s in severity_order if s in severity_counts.columns]
                
                # Add any other severities that might exist in the data but are not in our defined order
                other_severities = [s for s in severity_counts.columns if s not in available_severities]

                sev_fig = go.Figure()
                for severity in available_severities + other_severities:
                    sev_fig.add_trace(go.Bar(
                        name=severity, 
                        x=severity_counts.index, 
                        y=severity_counts[severity],
                        marker_color=color_map.get(severity)
                    ))
                sev_fig.update_layout(
                    barmode='stack', 
                    title_text='Issue Severity by Category', 
                    xaxis_title='Category', 
                    yaxis_title='Count'
                )
                st.plotly_chart(sev_fig, use_container_width=True)
    else:
        st.info("Awaiting file upload for quality analysis...")