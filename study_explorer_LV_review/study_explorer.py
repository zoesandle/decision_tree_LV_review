import streamlit as st
import pandas as pd
import os
from io import BytesIO
import xlsxwriter

def extract_unique_terms(series, separator=";"):
    """
    Extract unique variable names from cells containing
    semicolon-separated lists.
    """

    terms = set()

    for value in series.dropna():

        for term in str(value).split(separator):

            term = term.strip()

            if term:
                terms.add(term)

    return sorted(terms)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Sandle et al. 2026 Study Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
    <style>
    .explore-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 10px;
        margin-bottom: 30px;
        text-align: center;
    }
    .explore-header h1 {
        margin: 0;
        font-size: 2.5em;
        font-weight: bold;
    }
    .section-title {
        color: #667eea;
        font-size: 1.3em;
        font-weight: bold;
        margin-top: 25px;
        margin-bottom: 15px;
        border-bottom: 2px solid #667eea;
        padding-bottom: 10px;
    }
    .stDataFrame {
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
    <div class="explore-header">
        <h1>🔍 Explore Studies from Sandle et al. 2026</h1>
        <p>Database of studies using Latent Variable Modeling and investigating externalizing behavior.
            \nFilter studies based on your criteria.</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAD DATA
# ============================================================================
def load_data(file_input):
    """Load data from Excel or CSV file
    
    Parameters:
    file_input: Either a file path (string) or Streamlit UploadedFile object
    """
    # Check if it's a Streamlit UploadedFile object
    if hasattr(file_input, 'name'):
        # It's an UploadedFile object
        file_name = file_input.name
        if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            df = pd.read_excel(file_input)
        else:
            df = pd.read_csv(file_input)
    else:
        # It's a file path string
        if file_input.endswith('.xlsx') or file_input.endswith('.xls'):
            df = pd.read_excel(file_input)
        else:
            df = pd.read_csv(file_input)
    return df

DEFAULT_FILE = "studies_data.xlsx"

uploaded_file = st.sidebar.file_uploader(
    "Upload a replacement database",
    type=["xlsx", "xls", "csv"]
)

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    df = load_data(DEFAULT_FILE)

    
# Display data info
st.sidebar.success(f"✅ Loaded {len(df)} studies")

# ============================================================================
# DATA VALIDATION & CLEANING
# ============================================================================
# Convert 0/1 columns to numeric if they're not already
predictor_outcome_cols = ['behavior_predictor', 'behavior_outcome', 
                          'cognition_predictor', 'cognition_outcome',
                          'brain_predictor', 'brain_outcome',
                          'p/cu_predictor', 'p/cu_outcome',
                          'behavior_mod', 'cog_mod', 'brain_mod', 'p/cu_mod']

for col in predictor_outcome_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# ============================================================================
# SIDEBAR: SELECTION FILTERS
# ============================================================================
st.markdown("<div class='section-title'>🔎 Selection Filters</div>", 
            unsafe_allow_html=True)

# Dictionary to store active filters
active_filters = {}
col3, col4 = st.columns(2)

with col3:
    st.subheader("👥 Sample Characteristics")
    
    # Age group filter
    if 'age range' in df.columns:
        age_options = extract_unique_terms(df['age range'])
        age_options = ['All'] + sorted(age_options)
        age_filter = st.multiselect(
            "Select age group(s):",
            age_options,
            default='All'
        )
        if 'All' not in age_filter and age_filter:
            active_filters['age_range'] = age_filter
    
    # Sample type filter (Community vs Forensic)
    if 'Type of Sample' in df.columns:
        # Convert to string and filter out NaN values
        sample_types = extract_unique_terms(df['Type of Sample'])
        sample_types = ['All'] + sorted(sample_types)
        sample_filter = st.multiselect(
            "Select sample type(s):",
            sample_types,
            default='All',
            help="Where the sample was recruited from"
        )
        if 'All' not in sample_filter and sample_filter:
            active_filters['sample_type'] = sample_filter

    # Diagnosis  filter
    if 'Diagnoses if present' in df.columns:
        # Convert to string and filter out NaN values
        diag_options = extract_unique_terms(df['Diagnoses if present'])
        diag_options = ['All'] + sorted(diag_options)
        diag_filter = st.multiselect(
            "Select diagnoses:",
            diag_options,
            default='All',
            help="Any diagnoses present (can also be the case in community samples)"
            )
        if 'All' not in diag_filter and diag_filter:
            active_filters['diagnoses'] = diag_filter


with col4:
    st.subheader("🌍 Study Characteristics")
    
    # Region filter
    if 'Region of Data Collection' in df.columns:
        # Convert to string and filter out NaN values
        regions = extract_unique_terms(df['Region of Data Collection'])
        regions = ['All'] + sorted(regions)
        region_filter = st.multiselect(
            "Select region(s):",
            regions,
            default='All'
        )
        if 'All' not in region_filter and region_filter:
            active_filters['region'] = region_filter
    
    # Follow-up studies
    if 'study design' in df.columns:
        # Convert to string and filter out NaN values
        followup_options = extract_unique_terms(df['study design'])
        followup_options = ['All'] + sorted(followup_options)
        followup_filter = st.multiselect(
            "Select study design type(s):",
            followup_options,
            default='All'
        )
        if 'All' not in followup_filter and followup_filter:
            active_filters['followup'] = followup_filter

st.markdown("---")


st.subheader("📊 Variables and Roles")
with st.expander("Behavioral Variables", expanded=True):

    if "Behavioral Variable" in df.columns:

        behavior_vars = extract_unique_terms(
            df["Behavioral Variable"]
        )

        selected = st.multiselect(
            "Select behavioral variables",
            behavior_vars
        )

        st.subheader("Design")
    
        role_options = st.multiselect(
        "Select variable role(s):",
        options=['Behavior is predictor', 'Behavior is outcome', 'Behavior is moderator'],
        default=[],
        help="Which roles behavior plays in the studies"
        )
        
        if role_options:
            active_filters['roles'] = role_options
        
        assessment_options = st.multiselect(
        "Select assessment type(s):",
        options=["Self Report", "Other Report", "Observational", "Clinical Assessment", 
                 "Interview", "Records"],
        default=[],
        help="How the behavior was assessed in the studies"
        )

        if assessment_options:
            active_filters['Behavioral Assessment Type'] = assessment_options

        if selected:
            active_filters["behav_vars"] = selected



with st.expander("Cognitive Variables", expanded=True):

    if "Cognitive Variable" in df.columns:

        cog_vars = extract_unique_terms(
            df["Cognitive Variable"]
        )

        selected = st.multiselect(
            "Select cognitive variables",
            cog_vars
        )
        
        st.subheader("Design")
    
        role_options = st.multiselect(
        "Select variable role(s):",
        options=['Cognition is predictor', 'Cognition is outcome', 'Cognition is moderator'],
        default=[],
        help="Which roles cognition plays in the studies"
        )
        
        if role_options:
            active_filters['roles'] = role_options

        assessment_options = st.multiselect(
        "Select assessment type(s):",
        options=["Self Report", "Other Report", "Experimental"],
        default=[],
        help="How the behavior was assessed in the studies"
        )

        if assessment_options:  
            active_filters['Cognitive Assessment Type'] = assessment_options


        if selected:
            active_filters["cog_vars"] = selected



with st.expander("Psychopathic or CU traits", expanded=True):

    if "Psychopathic or CU traits" in df.columns:

        cu_vars = extract_unique_terms(
            df["Psychopathic or CU traits"]
        )

        selected = st.multiselect(
            "Select Psychopathic or CU traits",
            cu_vars
        )

        st.subheader("Design")
    
        role_options = st.multiselect(
        "Select variable role(s):",
        options=['Psychopathic/CU traits are predictor', 'Psychopathic/CU traits are outcome', 'Psychopathic/CU traits are moderator'],
        default=[],
        help="Which roles psychopathic/cu traits plays in the studies"
        )
        
        if role_options:
            active_filters['roles'] = role_options

        
        assessment_options = st.multiselect(
        "Select assessment type(s):",
        options=["Self Report", "Other Report", "Interview"],
        default=[],
        help="How the behavior was assessed in the studies"
        )

        if assessment_options:
            active_filters['Psychopathic or CU traits assessment type'] = assessment_options


        if selected:
            active_filters["cu_traits"] = selected

with st.expander("Brain Variables (click to expand)", expanded=False):

    if "Brain Variable" in df.columns:

        brain_vars = extract_unique_terms(
            df["Brain Variable"]
        )

        selected = st.multiselect(
            "Select brain variables",
            brain_vars
        )
        
        st.subheader("Design")
    
        role_options = st.multiselect(
        "Select variable role(s):",
        options=['Brain is predictor', 'Brain is outcome'],
        default=[],
        help="Which roles brain plays in the studies"
        )
        
        if role_options:
            active_filters['roles'] = role_options


        if selected:
            active_filters["brain_vars"] = selected


st.markdown("---")

st.subheader("🔧 Modeling Approach")
if 'Model' in df.columns:
    # Convert to string and filter out NaN values
    model_options = extract_unique_terms(df['Model'])
    model_options = ['All'] + sorted(model_options)
    model_filter = st.multiselect(
        "Select model(s):",
        model_options,
        default='All',
        help="CFA, SEM, LPA, etc."
    )
    if 'All' not in model_filter and model_filter:
        active_filters['model'] = model_filter

# ============================================================================
# SIDEBAR: SORTING OPTIONS
# ============================================================================
st.markdown("<div class='section-title'>📊 Sorting Options</div>", 
            unsafe_allow_html=True)

col5, col6 = st.columns(2)

with col5:
    sort_by = st.selectbox(
        "Sort by:",
        options=['Year (Newest First)', 'Year (Oldest First)', 
                'Sample Size (Largest)', 'Sample Size (Smallest)',
                'Author (A-Z)', 'Study Name (A-Z)'],
        index=0
    )

# ============================================================================
# APPLY FILTERS
# ============================================================================
filtered_df = df.copy()

# Filter by specific behavioral variables
if 'behav_vars' in active_filters:
    mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
    for var in active_filters['behav_vars']:
        if 'Behavioral Variable' in filtered_df.columns:
            mask |= filtered_df['Behavioral Variable'].fillna('').astype(str).str.contains(var, case=False, regex=False)
    filtered_df = filtered_df[mask]

# Filter by specific cognitive variables
if 'cog_vars' in active_filters:
    mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
    for var in active_filters['cog_vars']:
        if 'Cognitive Variable' in filtered_df.columns:
            mask |= filtered_df['Cognitive Variable'].fillna('').astype(str).str.contains(var, case=False, regex=False)
    filtered_df = filtered_df[mask]

# Filter by specific brain variables
if 'brain_vars' in active_filters:
    mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
    for var in active_filters['brain_vars']:
        if 'Brain Variable' in filtered_df.columns:
            mask |= filtered_df['Brain Variable'].fillna('').astype(str).str.contains(var, case=False, regex=False)
    filtered_df = filtered_df[mask]

# Filter by specific CU/psychopathic traits
if 'cu_traits' in active_filters:
    mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
    for trait in active_filters['cu_traits']:
        if 'psychopathic/cu-traits' in filtered_df.columns:
            mask |= filtered_df['psychopathic/cu-traits'].fillna('').astype(str).str.contains(trait, case=False, regex=False)
    filtered_df = filtered_df[mask]

# Filter by variable roles
if 'roles' in active_filters:
    role_filter = active_filters['roles']
    role_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
    
    ######## BEHAVIOR ########
    if 'Behavior is predictor' in role_filter:
        role_mask |= (
            (filtered_df.get('behavior_predictor', 0) == 1) 
        )
    
    if 'Behavior is outcome' in role_filter:
        role_mask |= (
            (filtered_df.get('behavior_outcome', 0) == 1) 
        )
    
    if 'Behavior is moderator' in role_filter:
        role_mask |= (
            (filtered_df.get('behavior_mod', 0) == 1) 
        )

    ######## COGNITION ########
    if 'Cognition is predictor' in role_filter:
        role_mask |= (
            (filtered_df.get('cognition_predictor', 0) == 1) 
        )
    
    if 'Cognition is outcome' in role_filter:
        role_mask |= (
            (filtered_df.get('cognition_outcome', 0) == 1) 
        )
    
    if 'Cognition is moderator' in role_filter:
        role_mask |= (
            (filtered_df.get('cog_mod', 0) == 1) 
        )

    ######## BRAIN ########
    if 'Brain is predictor' in role_filter:
        role_mask |= (
            (filtered_df.get('brain_predictor', 0) == 1) 
        )
    if 'Brain is outcome' in role_filter:
        role_mask |= (
            (filtered_df.get('brain_outcome', 0) == 1) 
        )
    
    ######## PSYCHOPATHIC/CU TRAITS ########
    if 'Psychopathic/CU traits are predictor' in role_filter:
        role_mask |= (
            (filtered_df.get('p/cu_predictor', 0) == 1) 
        )
    if 'Psychopathic/CU traits are outcome' in role_filter:
        role_mask |= (
            (filtered_df.get('p/cu_outcome', 0) == 1) 
        )
    if 'Psychopathic/CU traits are moderator' in role_filter:
        role_mask |= (
            (filtered_df.get('p/cu_mod', 0) == 1) 
        )
    
    filtered_df = filtered_df[role_mask]

#filter by specific variable assessment types
if 'Behavioral Assessment Type' in active_filters:
    filtered_df = filtered_df[filtered_df['Behavioral Assessment Type'].isin(active_filters['Behavioral Assessment Type'])]

if 'Cognitive Assessment Type' in active_filters:
    filtered_df = filtered_df[filtered_df['Cognitive Assessment Type'].isin(active_filters['Cognitive Assessment Type'])]

if 'Psychopathic or CU traits assessment type' in active_filters:
    filtered_df = filtered_df[filtered_df['Psychopathic or CU traits assessment type'].isin(active_filters['Psychopathic or CU traits assessment type'])]

# Filter by age range
if 'age_range' in active_filters:
    filtered_df = filtered_df[filtered_df['age range'].isin(active_filters['age_range'])]

# Filter by sample type
if 'sample_type' in active_filters:
    filtered_df = filtered_df[filtered_df['Type of Sample'].isin(active_filters['sample_type'])]

# Filter by region
if 'region' in active_filters:
    filtered_df = filtered_df[filtered_df['Region of Data Collection'].isin(active_filters['region'])]

# Filter by follow-up
if 'followup' in active_filters:
    filtered_df = filtered_df[filtered_df['Follow up'].isin(active_filters['followup'])]

# Filter by model
if 'model' in active_filters:
    filtered_df = filtered_df[filtered_df['Model'].isin(active_filters['model'])]

# ============================================================================
# APPLY SORTING
# ============================================================================
if 'Year (Newest First)' in sort_by:
    if 'Year' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('Year', ascending=False)
elif 'Year (Oldest First)' in sort_by:
    if 'Year' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('Year', ascending=True)
elif 'Sample Size (Largest)' in sort_by:
    if 'sample size' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('sample size', ascending=False, na_position='last')
elif 'Sample Size (Smallest)' in sort_by:
    if 'sample size' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('sample size', ascending=True, na_position='last')
elif 'Author (A-Z)' in sort_by:
    if 'First author' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('First author', ascending=True)
elif 'Study Name (A-Z)' in sort_by:
    if 'Study Name' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('Study Name', ascending=True)

# Reset index for display
filtered_df = filtered_df.reset_index(drop=True)

# ============================================================================
# DISPLAY RESULTS
# ============================================================================
st.markdown("<div class='section-title'>📋 Studies</div>", 
            unsafe_allow_html=True)

result_text = f"Showing **{len(filtered_df)}** of **{len(df)}** studies"
st.info(result_text)

#drop columns for cleaner display
columns_to_drop = ['behavior_predictor', 'behavior_outcome', 
                   'cognition_predictor', 'cognition_outcome',
                   'brain_predictor', 'brain_outcome',
                   'p/cu_predictor', 'p/cu_outcome',
                   'behavior_mod', 'cog_mod', 'brain_mod', 'p/cu_mod']

filtered_df = filtered_df.drop(columns=[col for col in columns_to_drop if col in filtered_df.columns])

# Display table
st.dataframe(
    filtered_df,
    use_container_width=True,
    height=500
)

#col5, col6 = st.columns(2)

#with col5:
 ##   sort_by = st.selectbox(
   #     "Sort by:",
    #    options=['Year (Newest First)', 'Year (Oldest First)', 
     #           'Sample Size (Largest)', 'Sample Size (Smallest)',
      #          'Author (A-Z)', 'Study Name (A-Z)'],
       # index=0
    #)

# ============================================================================
# DOWNLOAD SECTION
# ============================================================================
st.markdown("<div class='section-title'>💾 Download Selection</div>", 
            unsafe_allow_html=True)

col_download1, col_download2, col_download3 = st.columns(3)

with col_download1:
    # Download as CSV
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name="selected_studies.csv",
        mime="text/csv",
        use_container_width=True
    )

with col_download2:
    # Download as Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        filtered_df.to_excel(writer, sheet_name='Studies', index=False)
    excel_data = output.getvalue()
    
    st.download_button(
        label="Download as Excel",
        data=excel_data,
        file_name="selected_studies.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

with col_download3:
    # Display summary statistics
    st.metric("Total Studies Selected", len(filtered_df))

# ============================================================================
# OPTIONAL: SHOW FILTER SUMMARY
# ============================================================================
if active_filters:
    st.markdown("---")
    st.markdown("<div class='section-title'>🏷️ Active Filters</div>", 
                unsafe_allow_html=True)
    
    filter_summary = []
    if 'behavior' in active_filters:
        filter_summary.append("✓ Behavioral Variables")
    if 'cognition' in active_filters:
        filter_summary.append("✓ Cognitive Variables")
    if 'brain' in active_filters:
        filter_summary.append("✓ Brain Variables")
    if 'cu' in active_filters:
        filter_summary.append("✓ Psychopathic/CU Traits")
    if 'roles' in active_filters:
        filter_summary.append(f"✓ Roles: {', '.join(active_filters['roles'])}")
    if 'age_range' in active_filters:
        filter_summary.append(f"✓ Age: {', '.join(active_filters['age_range'])}")
    if 'sample_type' in active_filters:
        filter_summary.append(f"✓ Sample: {', '.join(active_filters['sample_type'])}")
    if 'region' in active_filters:
        filter_summary.append(f"✓ Region: {', '.join(active_filters['region'])}")
    if 'model' in active_filters:
        filter_summary.append(f"✓ Model: {', '.join(active_filters['model'])}")
    
    if filter_summary:
        st.write(" | ".join(filter_summary))

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.9em; margin-top: 20px;'>
    <p>Please cite the corresponding study when using this tool.</p>
    </div>
    """, unsafe_allow_html=True)