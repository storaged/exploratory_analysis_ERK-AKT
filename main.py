import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import os
import re

erk_pathway_genes_global = [
    'EGF', 'EGFR', 'HRAS', 'KRAS', 'NRAS',
    'RAF1', 'BRAF', 'MAP2K1', 'MAP2K2',
    'MAPK1', 'MAPK3', 'ELK1', 'FOS', 'JUN'
]

# AKT/PI3K pathway genes (PID_PI3KCI_AKT_PATHWAY)
akt_pathway_genes_global = [
    'PIK3CA', 'PIK3CB', 'PIK3CD', 'PIK3CG',
    'AKT1', 'AKT2', 'AKT3', 'PTEN',
    'TSC1', 'TSC2', 'MTOR',
    'RPS6KB1', 'EIF4EBP1', 'GSK3B', 'BAD',
    'FOXO1', 'FOXO3', 'FOXO4'
]

# EGFR ligands (from UniProt & literature)
egfr_ligands_global = [
    'EGF', 'TGFA', 'AREG', 'BTC',
    'HBEGF', 'EPGN', 'EREG',
    'NRG1', 'NRG2'
]

# Set page config
st.set_page_config(
    page_title="RNA-seq Analysis Dashboard",
    page_icon="🧬",
    layout="wide"
)

# Define functions for analysis

def load_data(count_matrix_file):
    """Load count matrix from Excel file"""
    df = pd.read_excel(count_matrix_file)
    return df

def parse_sample_metadata(df):
    """Extract sample metadata from column names"""
    # Get sample columns (all except gene_id and symbol)
    sample_cols = [col for col in df.columns if col not in ['gene_id', 'symbol']]
    
    # Extract metadata from sample names
    metadata = []
    for col in sample_cols:
        parts = col.split('_')
        if len(parts) >= 2:
            mutation = parts[0]
            if len(parts) > 2:
                treatment = parts[1]
                replicate = parts[2]
            else:
                treatment = parts[1].split('rep')[0]
                replicate = 'rep' + parts[1].split('rep')[1] if 'rep' in parts[1] else 'rep1'
            
            metadata.append({
                'sample': col,
                'mutation': mutation,
                'treatment': treatment,
                'replicate': replicate
            })
    
    metadata_df = pd.DataFrame(metadata)
    return metadata_df

def filter_low_expression_genes(df, min_count=10):
    """Filter out genes with low expression counts"""
    count_cols = [col for col in df.columns if col not in ['gene_id', 'symbol']]
    # Calculate mean expression for each gene
    df['mean_counts'] = df[count_cols].mean(axis=1)
    # Filter genes with mean counts >= min_count
    filtered_df = df[df['mean_counts'] >= min_count].copy()
    filtered_df = filtered_df.drop('mean_counts', axis=1)
    return filtered_df

def normalize_counts(df):
    """Normalize counts using CPM (Counts Per Million)"""
    count_cols = [col for col in df.columns if col not in ['gene_id', 'symbol']]
    
    # Calculate library size (sum of counts) for each sample
    lib_sizes = df[count_cols].sum()
    
    # Normalize counts to CPM
    norm_df = df.copy()
    for col in count_cols:
        if lib_sizes[col] > 0:  # Avoid division by zero
            norm_df[col] = (df[col] * 1e6) / lib_sizes[col]
    
    return norm_df

def run_pca(df, metadata_df, n_components=3):
    """Perform PCA on expression data"""
    # Get count columns
    count_cols = [col for col in df.columns if col not in ['gene_id', 'symbol']]
    
    # Extract count matrix for PCA
    count_matrix = df[count_cols].T
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(count_matrix)
    
    # Run PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    
    # Create a DataFrame with PCA results
    pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
    pca_df['sample'] = count_cols
    
    # Merge with metadata
    pca_df = pd.merge(pca_df, metadata_df, on='sample')
    
    # Calculate explained variance
    explained_var = pca.explained_variance_ratio_ * 100
    
    return pca_df, explained_var

def de_analysis(df, metadata_df, mutation, control_mutation='WT', treatment='DMSO', alpha=0.05, fc_threshold=1):
    """Differential expression analysis comparing a mutation to control"""
    # Filter metadata
    test_samples = metadata_df[(metadata_df['mutation'] == mutation) & 
                              (metadata_df['treatment'] == treatment)]['sample'].tolist()
    
    control_samples = metadata_df[(metadata_df['mutation'] == control_mutation) & 
                                 (metadata_df['treatment'] == treatment)]['sample'].tolist()
    
    if len(test_samples) == 0 or len(control_samples) == 0:
        return None
    
    # Get gene information
    gene_info = df[['gene_id', 'symbol']].copy()
    
    # Initialize results
    results = []
    
    # For each gene, perform t-test between test and control
    for idx, row in df.iterrows():
        gene_id = row['gene_id']
        symbol = row['symbol']
        
        test_values = [row[sample] for sample in test_samples]
        control_values = [row[sample] for sample in control_samples]
        
        # Calculate mean expression
        test_mean = np.mean(test_values)
        control_mean = np.mean(control_values)
        
        # Calculate log2 fold change
        # Add small pseudocount to avoid log(0)
        log2fc = np.log2((test_mean + 0.1) / (control_mean + 0.1))
        
        # Perform t-test
        t_stat, p_value = ttest_ind(test_values, control_values, equal_var=False)
        
        results.append({
            'gene_id': gene_id,
            'symbol': symbol,
            'control_mean': control_mean,
            'test_mean': test_mean,
            'log2FC': log2fc,
            'p_value': p_value
        })
    
    # Create DataFrame
    de_results = pd.DataFrame(results)
    
    # Adjust p-values for multiple testing
    if not de_results.empty:
        de_results['padj'] = multipletests(de_results['p_value'], method='fdr_bh')[1]
        
        # Flag significant genes
        de_results['significant'] = (de_results['padj'] < alpha) & (abs(de_results['log2FC']) >= fc_threshold)
        
        # Sort by adjusted p-value
        de_results = de_results.sort_values('padj')
    
    return de_results

def treatment_effect_analysis(df, metadata_df, mutation, treatment, alpha=0.05, fc_threshold=1):
    """Analyze effect of treatment compared to DMSO for a specific mutation"""
    # Filter metadata
    test_samples = metadata_df[(metadata_df['mutation'] == mutation) & 
                              (metadata_df['treatment'] == treatment)]['sample'].tolist()
    
    control_samples = metadata_df[(metadata_df['mutation'] == mutation) & 
                                 (metadata_df['treatment'] == 'DMSO')]['sample'].tolist()
    
    if len(test_samples) == 0 or len(control_samples) == 0:
        return None
    else: 
        print(f"Test samples: {test_samples}")
        print(f"Control samples: {control_samples}")
        
    # Get gene information
    gene_info = df[['gene_id', 'symbol']].copy()
    
    # Initialize results
    results = []
    
    print("Starting differential expression analysis (number of genes):", len(df))
    # For each gene, perform t-test between treatment and DMSO
    for idx, row in df.iterrows():
        gene_id = row['gene_id']
        symbol = row['symbol']
        
        test_values = [row[sample] for sample in test_samples]
        control_values = [row[sample] for sample in control_samples]
        
        # Calculate mean expression
        test_mean = np.mean(test_values)
        control_mean = np.mean(control_values)
        
        # Calculate log2 fold change - add safeguards against zero values
        # Use a larger pseudocount to avoid very small numbers
        if test_mean == 0 and control_mean == 0:
            # Both are zero - no change
            log2fc = 0
        else:
            # Add pseudocount to both values to avoid division by zero
            log2fc = np.log2((test_mean + 1) / (control_mean + 1))
        
        # Skip t-test if all values are identical (would cause division by zero in t-test)
        if all(x == test_values[0] for x in test_values) and all(x == control_values[0] for x in control_values) and test_values[0] == control_values[0]:
            p_value = 1.0  # No difference
        else:
            try:
                # Perform t-test
                t_stat, p_value = ttest_ind(test_values, control_values, equal_var=False)
                
                # Handle NaN p-value (happens when all values in at least one group are identical)
                if np.isnan(p_value):
                    p_value = 1.0
            except:
                # Fallback for any other numerical issue
                p_value = 1.0
        
        results.append({
            'gene_id': gene_id,
            'symbol': symbol,
            'DMSO_mean': control_mean,
            'treatment_mean': test_mean,
            'log2FC': log2fc,
            'p_value': p_value
        })
    
    # Create DataFrame
    de_results = pd.DataFrame(results)
    
    # Adjust p-values for multiple testing
    if not de_results.empty:
        de_results['padj'] = multipletests(de_results['p_value'], method='fdr_bh')[1]
        
        # Flag significant genes
        de_results['significant'] = (de_results['padj'] < alpha) & (abs(de_results['log2FC']) >= fc_threshold)
        
        # Sort by adjusted p-value
        de_results = de_results.sort_values('padj')
    
    print(de_results.head())
    return de_results

def analyze_erk_akt_pathway(df, metadata_df):
    """Analyze expression of genes in ERK/AKT pathways"""
    # Define key genes in ERK/AKT pathways and EGFR ligands

    erk_pathway_genes = erk_pathway_genes_global

    # AKT/PI3K pathway genes (PID_PI3KCI_AKT_PATHWAY)
    akt_pathway_genes = akt_pathway_genes_global

    # EGFR ligands (from UniProt & literature)
    egfr_ligands = egfr_ligands_global

    
    # Combine all pathway genes
    pathway_genes = erk_pathway_genes + akt_pathway_genes + egfr_ligands
    
    # Filter gene expression data for these genes
    pathway_df = df[df['symbol'].isin(pathway_genes)].copy()
    
    # Get mutations and treatments
    mutations = metadata_df['mutation'].unique()
    treatments = metadata_df['treatment'].unique()
    
    # Prepare data structure for heatmap
    heatmap_data = []
    
    # For each pathway gene and mutation, calculate mean expression for DMSO treatment
    for gene_idx, gene_row in pathway_df.iterrows():
        gene = gene_row['symbol']
        gene_id = gene_row['gene_id']
        
        row_data = {'gene': gene, 'gene_id': gene_id}
        
        for mutation in mutations:
            # Get DMSO samples for this mutation
            dmso_samples = metadata_df[(metadata_df['mutation'] == mutation) & 
                                     (metadata_df['treatment'] == 'DMSO')]['sample'].tolist()
            
            if dmso_samples:
                # Calculate mean expression across replicates
                mean_expr = np.mean([gene_row[sample] for sample in dmso_samples])
                row_data[mutation] = mean_expr
            else:
                row_data[mutation] = np.nan
        
        heatmap_data.append(row_data)
    
    # Create DataFrame
    heatmap_df = pd.DataFrame(heatmap_data)
    
    return pathway_df, heatmap_df

def calculate_mutation_correlations(heatmap_df):
    """Calculate correlations between mutations based on pathway gene expression"""
    # Drop gene info columns
    expr_df = heatmap_df.drop(['gene', 'gene_id'], axis=1)
    
    # Calculate correlation matrix
    corr_matrix = expr_df.corr(method='pearson')
    
    return corr_matrix

def get_most_variable_genes(df, metadata_df, n=100):
    """Get the most variable genes across all samples"""
    # Get count columns
    count_cols = [col for col in df.columns if col not in ['gene_id', 'symbol']]
    
    # Calculate variance for each gene
    gene_vars = df[count_cols].var(axis=1)
    
    # Add variance to DataFrame
    var_df = df.copy()
    var_df['variance'] = gene_vars
    
    # Sort by variance and get top n genes
    top_var_genes = var_df.sort_values('variance', ascending=False).head(n).copy()
    top_var_genes = top_var_genes.drop('variance', axis=1)
    
    return top_var_genes

def generate_heatmap_data(df, metadata_df, genes=None, mutations=None, treatment='DMSO'):
    """Generate heatmap data for selected genes across mutations"""
    # Filter genes if provided
    if genes is not None:
        gene_df = df[df['symbol'].isin(genes)].copy()
    else:
        gene_df = df.copy()
    
    # Filter mutations if provided, otherwise use all
    if mutations is None:
        mutations = metadata_df['mutation'].unique()
    
    # Prepare heatmap data
    heatmap_data = []
    
    for gene_idx, gene_row in gene_df.iterrows():
        gene = gene_row['symbol']
        gene_id = gene_row['gene_id']
        
        row_data = {'gene': gene, 'gene_id': gene_id}
        
        for mutation in mutations:
            # Get samples for this mutation and treatment
            samples = metadata_df[(metadata_df['mutation'] == mutation) & 
                                (metadata_df['treatment'] == treatment)]['sample'].tolist()
            
            if samples:
                # Calculate mean expression across replicates
                mean_expr = np.mean([gene_row[sample] for sample in samples])
                row_data[mutation] = mean_expr
            else:
                row_data[mutation] = np.nan
        
        heatmap_data.append(row_data)
    
    # Create DataFrame
    heatmap_df = pd.DataFrame(heatmap_data)
    
    return heatmap_df

def plot_volcanoplot(de_results, title='Volcano Plot', max_genes_to_label=20, significant_only=True):
    """Generate interactive volcano plot for DE results"""
    if de_results is None or de_results.empty:
        return go.Figure()
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot - conditionally filter for significant genes
    if significant_only:
        highlight_genes = de_results['significant']
    else:
        # When not filtering by significance, we can highlight based on fold change alone
        highlight_genes = abs(de_results['log2FC']) >= 1
    
    # Plot all points
    fig.add_trace(go.Scatter(
        x=de_results['log2FC'],
        y=-np.log10(de_results['padj']),
        mode='markers',
        marker=dict(
            color=highlight_genes.map({True: 'red', False: 'gray'}),
            size=8,
            opacity=0.7
        ),
        text=de_results['symbol'],
        hovertemplate='<b>%{text}</b><br>log2FC: %{x:.2f}<br>-log10(padj): %{y:.2f}<extra></extra>'
    ))
    
    # Label top genes based on filter setting
    if highlight_genes.sum() > 0:
        top_genes = de_results[highlight_genes].sort_values('padj').head(max_genes_to_label)
        
        for idx, row in top_genes.iterrows():
            fig.add_annotation(
                x=row['log2FC'],
                y=-np.log10(row['padj']),
                text=row['symbol'],
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-20
            )
    
    # Add horizontal line for p-value cutoff
    fig.add_shape(
        type='line',
        x0=min(de_results['log2FC']) - 0.5,
        x1=max(de_results['log2FC']) + 0.5,
        y0=-np.log10(0.05),
        y1=-np.log10(0.05),
        line=dict(color='blue', dash='dash')
    )
    
    # Add vertical lines for fold change cutoff
    fig.add_shape(
        type='line',
        x0=1,
        x1=1,
        y0=0,
        y1=max(-np.log10(de_results['padj'].replace(0, 1e-300))) + 1,
        line=dict(color='blue', dash='dash')
    )
    
    fig.add_shape(
        type='line',
        x0=-1,
        x1=-1,
        y0=0,
        y1=max(-np.log10(de_results['padj'].replace(0, 1e-300))) + 1,
        line=dict(color='blue', dash='dash')
    )
    
    # Update layout with increased size for better visibility
    fig.update_layout(
        title=title,
        xaxis_title='log<sub>2</sub> Fold Change',
        yaxis_title='-log<sub>10</sub> adjusted p-value',
        hovermode='closest',
        width=900,   # Increased width
        height=700,  # Increased height
        margin=dict(l=50, r=50, t=80, b=50)  # Adjusted margins
    )
    
    return fig

def plot_enrichment_analysis(up_genes, down_genes, pathway_genes):
    """Perform simple enrichment analysis for ERK/AKT pathway genes"""
    # Define pathway categories
    erk_pathway_genes = erk_pathway_genes_global
    akt_pathway_genes = akt_pathway_genes_global
    egfr_ligands = egfr_ligands_global
    
    # Count overlaps
    up_erk = len(set(up_genes).intersection(set(erk_pathway_genes)))
    up_akt = len(set(up_genes).intersection(set(akt_pathway_genes)))
    up_egfr = len(set(up_genes).intersection(set(egfr_ligands)))
    
    down_erk = len(set(down_genes).intersection(set(erk_pathway_genes)))
    down_akt = len(set(down_genes).intersection(set(akt_pathway_genes)))
    down_egfr = len(set(down_genes).intersection(set(egfr_ligands)))
    
    # Create enrichment data
    enrichment_data = {
        'Pathway': ['ERK', 'AKT', 'EGFR Ligands'],
        'Upregulated': [up_erk, up_akt, up_egfr],
        'Downregulated': [down_erk, down_akt, down_egfr]
    }
    
    enrichment_df = pd.DataFrame(enrichment_data)
    
    # Melt the dataframe for easy plotting
    enrichment_df_melted = pd.melt(
        enrichment_df, 
        id_vars=['Pathway'],
        value_vars=['Upregulated', 'Downregulated'],
        var_name='Regulation',
        value_name='Count'
    )
    
    # Create bar chart with increased size
    fig = px.bar(
        enrichment_df_melted,
        x='Pathway',
        y='Count',
        color='Regulation',
        barmode='group',
        title='Pathway Enrichment Analysis',
        labels={'Count': 'Number of Genes', 'Pathway': 'Signaling Pathway'}
    )
    
    # Update layout to increase size
    fig.update_layout(
        width=800,
        height=600,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

# Main app layout
def main():
    st.title("RNA-seq Analysis Dashboard 🧬")
    st.write("Upload your count matrix file to analyze RNA-seq data and explore patterns in gene expression across mutations and treatments.")
    
    # Sidebar
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload count matrix (Excel/CSV file)", type=["xlsx", "csv"])
    
    # Load demo data if no file is uploaded
    if uploaded_file is None:
        st.sidebar.info("Please upload a count matrix file to begin analysis.")
        st.info("No data uploaded yet. Upload your count matrix file to begin analysis.")
        st.info("Your file should contain columns with sample names in the format: [Mutation]_[Treatment]_[Replicate] (e.g., AKT-E17K_DMSO_rep1)")
        
        # Explain the RNA-seq analysis workflow
        st.header("RNA-seq Analysis Workflow")
        st.write("""
        This app will help you analyze your RNA-seq data with the following steps:
        
        1. **Data Loading & Processing**:
           - Load count matrix
           - Extract sample metadata
           - Filter low-expression genes
           - Normalize counts
        
        2. **Exploratory Analysis**:
           - PCA visualization
           - Clustering of mutations based on expression profiles
           - Heatmaps of top variable genes
        
        3. **Differential Expression Analysis**:
           - Compare mutations to wild-type
           - Analyze treatment effects
           - Volcano plots of differential expression results
        
        4. **Pathway Analysis**:
           - Focus on ERK/AKT signaling pathway genes
           - EGFR ligand expression analysis
           - Treatment response patterns
        """)
        
        return

    # When data is uploaded, start processing
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, sep =";")
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.sidebar.error("Unsupported file format. Please upload a CSV or Excel file.")
            st.error("Please upload a file with .csv or .xlsx extension.")
            return
            
        # Check if required columns exist
        required_cols = ['gene_id', 'symbol']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.sidebar.error(f"Missing required columns: {', '.join(missing_cols)}")
            st.error(f"Your file must contain the following columns: {', '.join(required_cols)}")
            return
            
        # Check if there are enough sample columns
        sample_cols = [col for col in df.columns if col not in ['gene_id', 'symbol']]
        if len(sample_cols) < 1:
            st.sidebar.error("No sample columns found in the file.")
            st.error("Your file must contain at least one sample column in addition to gene_id and symbol.")
            return
        
        st.sidebar.success("Data loaded successfully!")
        
        # Extract metadata
        metadata_df = parse_sample_metadata(df)
        
        # Validate metadata extraction
        if len(metadata_df) == 0:
            st.sidebar.error("Could not parse sample metadata from column names.")
            st.error("Please ensure your sample columns follow the format: [Mutation]_[Treatment]_[Replicate]")
            return
            
    except pd.errors.EmptyDataError:
        st.sidebar.error("The uploaded file is empty.")
        st.error("The file you uploaded does not contain any data. Please check your file and try again.")
        return
    except pd.errors.ParserError:
        st.sidebar.error("Error parsing the file.")
        st.error("The file format could not be parsed correctly. Please ensure it is a valid CSV or Excel file.")
        return
    except Exception as e:
        st.sidebar.error("An error occurred while processing the file.")
        st.error(f"Error details: {str(e)}")
        return

    # Data filtering options
    st.sidebar.header("Data Processing")
    min_count = st.sidebar.slider("Minimum expression threshold", min_value=0, max_value=50, value=10)
    
    # Filter low-expression genes
    filtered_df = filter_low_expression_genes(df, min_count=min_count)
    
    # Normalize counts
    normalized_df = normalize_counts(filtered_df)
    
    # Display basic stats
    st.sidebar.write(f"Total genes: {len(df)}")
    st.sidebar.write(f"Genes after filtering: {len(filtered_df)}")
    
    # Get unique mutations and treatments
    mutations = sorted(metadata_df['mutation'].unique())
    treatments = sorted(metadata_df['treatment'].unique())
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Mutation Analysis", "Treatment Effects", "Pathway Analysis"])
    
    # Tab 1: Overview
    with tab1:
        st.header("Dataset Overview")
        
        # Display metadata summary
        st.subheader("Experiment Metadata")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Number of samples:** {len(metadata_df)}")
            st.write(f"**Number of mutations:** {len(mutations)}")
            st.write(f"**Mutations:** {', '.join(mutations)}")
        
        with col2:
            st.write(f"**Number of treatments:** {len(treatments)}")
            st.write(f"**Treatments:** {', '.join(treatments)}")
        
        # Sample distribution
        st.subheader("Sample Distribution")
        sample_counts = metadata_df.groupby(['mutation', 'treatment']).size().reset_index(name='count')
        sample_counts_pivot = sample_counts.pivot(index='mutation', columns='treatment', values='count').fillna(0)
        
        fig = px.imshow(
            sample_counts_pivot,
            labels=dict(x="Treatment", y="Mutation", color="Number of samples"),
            x=sample_counts_pivot.columns,
            y=sample_counts_pivot.index,
            color_continuous_scale='Viridis',
            title="Number of Samples by Mutation and Treatment"
        )
        # Increase figure size
        fig.update_layout(width=900, height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Run PCA
        st.subheader("Principal Component Analysis (PCA)")
        pca_df, explained_var = run_pca(normalized_df, metadata_df)
        
        # PCA plot
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Color by mutation or treatment
            color_by = st.radio("Color by:", ["mutation", "treatment"])
            
            fig = px.scatter(
                pca_df, 
                x='PC1', 
                y='PC2', 
                color=color_by,
                hover_data=['sample'],
                labels={'PC1': f'PC1 ({explained_var[0]:.1f}%)', 
                        'PC2': f'PC2 ({explained_var[1]:.1f}%)'},
                title="PCA Plot"
            )
            # Increase figure size
            fig.update_layout(width=800, height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Explained Variance:**")
            for i, var in enumerate(explained_var):
                st.write(f"PC{i+1}: {var:.1f}%")
        
        # Most variable genes
        st.subheader("Top Variable Genes")
        num_var_genes = st.slider("Number of top variable genes to show", min_value=10, max_value=100, value=50)
        top_var_genes = get_most_variable_genes(normalized_df, metadata_df, n=num_var_genes)
        
        # Heatmap of top variable genes
        st.write("Heatmap of top variable genes across mutations (DMSO treatment)")
        st.info("This heatmap shows the expression patterns of genes with the highest variance across all samples, specifically in DMSO treatment. These genes are most likely to reveal differences between mutations.")
        
        # Generate heatmap data
        heatmap_df = generate_heatmap_data(top_var_genes, metadata_df, treatment='DMSO')
        
        # Prepare data for heatmap - use matplotlib/seaborn for better control over rotation
        heatmap_genes = heatmap_df['gene'].tolist()
        heatmap_values = heatmap_df.drop(['gene', 'gene_id'], axis=1).values
        heatmap_columns = heatmap_df.columns[2:].tolist()
        
        # Create a rotated heatmap using matplotlib
        plt.figure(figsize=(14, 8))
        sns.heatmap(
            heatmap_values.T,  # Transpose for rotation
            cmap="viridis",
            xticklabels=heatmap_genes,
            yticklabels=heatmap_columns,
            cbar_kws={'label': 'Expression'}
        )
        plt.title("Expression of Top Variable Genes Across Mutations", fontsize=16)
        plt.xlabel("Gene")
        plt.ylabel("Mutation")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(plt.gcf())
    
    # Tab 2: Mutation Analysis
    with tab2:
        st.header("Differential Expression Analysis of Mutations")
        
        # Select mutation and control
        col1, col2 = st.columns(2)
        with col1:
            selected_mutation = st.selectbox("Select mutation:", mutations, index=0)
        with col2:
            control_mutation = st.selectbox("Control mutation:", mutations, index=mutations.index('WT') if 'WT' in mutations else 0)
        
        if selected_mutation == control_mutation:
            st.warning("Please select different mutations for comparison")
        else:
            # Run DE analysis
            de_results = de_analysis(normalized_df, metadata_df, selected_mutation, control_mutation)
            
            if de_results is None or de_results.empty:
                st.error(f"Not enough data for comparison between {selected_mutation} and {control_mutation}")
            else:
                # Summary of DE results
                st.subheader("Differential Expression Results")
                
                # Count significant genes
                num_sig = de_results['significant'].sum()
                num_up = sum((de_results['significant']) & (de_results['log2FC'] > 0))
                num_down = sum((de_results['significant']) & (de_results['log2FC'] < 0))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Significant DEGs", num_sig)
                with col2:
                    st.metric("Upregulated", num_up)
                with col3:
                    st.metric("Downregulated", num_down)
                
                # Volcano plot with explanation and significance filter checkbox
                st.subheader("Volcano Plot")
                st.info(f"This plot shows differential expression between {selected_mutation} and {control_mutation}. Red dots represent genes with significant changes.")
                
                # Add checkbox to filter for significant genes only
                show_significant_only = st.checkbox("Show significant genes only", value=True, key="sig_only_mutation")
                
                volcano_fig = plot_volcanoplot(
                    de_results, 
                    title=f'{selected_mutation} vs {control_mutation}',
                    significant_only=show_significant_only
                )
                st.plotly_chart(volcano_fig, use_container_width=True)
                
                # Top DEGs tables with filter checkbox
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Top Upregulated Genes")
                    
                    # Determine which genes to show based on filter setting
                    if show_significant_only:
                        top_up = de_results[(de_results['significant']) & (de_results['log2FC'] > 0)].sort_values('padj').head(20)
                    else:
                        top_up = de_results[de_results['log2FC'] > 0].sort_values('padj').head(20)
                        
                    if not top_up.empty:
                        st.dataframe(top_up[['symbol', 'log2FC', 'padj', 'significant']])
                    else:
                        st.write("No upregulated genes found with current filter settings")
                
                with col2:
                    st.subheader("Top Downregulated Genes")
                    
                    # Determine which genes to show based on filter setting
                    if show_significant_only:
                        top_down = de_results[(de_results['significant']) & (de_results['log2FC'] < 0)].sort_values('padj').head(20)
                    else:
                        top_down = de_results[de_results['log2FC'] < 0].sort_values('padj').head(20)
                        
                    if not top_down.empty:
                        st.dataframe(top_down[['symbol', 'log2FC', 'padj', 'significant']])
                    else:
                        st.write("No downregulated genes found with current filter settings")
                
                # Pathway enrichment analysis with explanation
                st.subheader("Pathway Enrichment Analysis")
                st.info("This analysis examines how many differentially expressed genes are part of key signaling pathways. If the plot appears empty, it means no significant enrichment was found in these pathways for the selected comparison.")
                
                # Get significant genes
                up_genes = de_results[(de_results['significant']) & (de_results['log2FC'] > 0)]['symbol'].tolist()
                down_genes = de_results[(de_results['significant']) & (de_results['log2FC'] < 0)]['symbol'].tolist()
                
                # Define pathway genes
                erk_akt_genes = ['MAPK1', 'MAPK3', 'MAP2K1', 'MAP2K2', 'RAF1', 'BRAF', 'ARAF', 'KRAS', 'NRAS', 'HRAS',
                                'AKT1', 'AKT2', 'AKT3', 'PIK3CA', 'PIK3CB', 'PIK3CD', 'PIK3CG', 'PTEN', 'MTOR', 'TSC1', 'TSC2',
                                'EGF', 'TGFA', 'HBEGF', 'BTC', 'AREG', 'EREG', 'EPGN']
                
                # Plot enrichment
                enrichment_fig = plot_enrichment_analysis(up_genes, down_genes, erk_akt_genes)
                st.plotly_chart(enrichment_fig, use_container_width=True)
                
                # If no enrichment, provide explanation
                if len(set(up_genes).intersection(erk_akt_genes)) == 0 and len(set(down_genes).intersection(erk_akt_genes)) == 0:
                    st.write("No ERK/AKT pathway genes or EGFR ligands were found among the differentially expressed genes in this comparison.")
                
                # Export results
                st.subheader("Export Results")
                
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')
                
                csv = convert_df_to_csv(de_results)
                st.download_button(
                    label="Download DE results as CSV",
                    data=csv,
                    file_name=f'{selected_mutation}_vs_{control_mutation}_DE_results.csv',
                    mime='text/csv',
                )
    
    # Tab 3: Treatment Effects
    with tab3:
        st.header("Analysis of Treatment Effects")
        
        # Select mutation and treatment
        col1, col2 = st.columns(2)
        with col1:
            selected_mutation_t = st.selectbox("Select mutation:", mutations, index=0, key="mutation_treatment")
        with col2:
            treatments_no_dmso = [t for t in treatments if t != 'DMSO']
            if treatments_no_dmso:
                selected_treatment = st.selectbox("Select treatment:", treatments_no_dmso)
            else:
                st.error("No treatments other than DMSO found in the data")
                selected_treatment = None
        
        if selected_treatment:
            # Run treatment effect analysis
            treatment_results = treatment_effect_analysis(normalized_df, metadata_df, selected_mutation_t, selected_treatment)
            
            if treatment_results is None or treatment_results.empty:
                st.error(f"Not enough data for comparison between {selected_treatment} and DMSO in {selected_mutation_t}")
            else:
                # Summary of treatment effect results
                st.subheader("Treatment Effect Results")
                
                # Count significant genes
                num_sig = treatment_results['significant'].sum()
                num_up = sum((treatment_results['significant']) & (treatment_results['log2FC'] > 0))
                num_down = sum((treatment_results['significant']) & (treatment_results['log2FC'] < 0))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Significant DEGs", num_sig)
                with col2:
                    st.metric("Upregulated", num_up)
                with col3:
                    st.metric("Downregulated", num_down)
                
                # Volcano plot with significance filter checkbox
                st.subheader("Volcano Plot")
                st.info(f"This plot shows how {selected_treatment} affects gene expression in {selected_mutation_t} cells compared to DMSO control.")
                
                # Add checkbox to filter for significant genes only
                show_significant_only_t = st.checkbox("Show significant genes only", value=True, key="sig_only_treatment")
                
                volcano_fig = plot_volcanoplot(
                    treatment_results, 
                    title=f'Effect of {selected_treatment} vs DMSO in {selected_mutation_t}',
                    significant_only=show_significant_only_t
                )
                st.plotly_chart(volcano_fig, use_container_width=True)
                
                # Top DEGs tables with filter checkbox
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Top Upregulated Genes")
                    
                    # Determine which genes to show based on filter setting
                    if show_significant_only_t:
                        top_up = treatment_results[(treatment_results['significant']) & (treatment_results['log2FC'] > 0)].sort_values('padj').head(20)
                    else:
                        top_up = treatment_results[treatment_results['log2FC'] > 0].sort_values('padj').head(20)
                        
                    if not top_up.empty:
                        st.dataframe(top_up[['symbol', 'log2FC', 'padj', 'significant']])
                    else:
                        st.write("No upregulated genes found with current filter settings")
                
                with col2:
                    st.subheader("Top Downregulated Genes")
                    
                    # Determine which genes to show based on filter setting
                    if show_significant_only_t:
                        top_down = treatment_results[(treatment_results['significant']) & (treatment_results['log2FC'] < 0)].sort_values('padj').head(20)
                    else:
                        top_down = treatment_results[treatment_results['log2FC'] < 0].sort_values('padj').head(20)
                        
                    if not top_down.empty:
                        st.dataframe(top_down[['symbol', 'log2FC', 'padj', 'significant']])
                    else:
                        st.write("No downregulated genes found with current filter settings")
                
                # Treatment comparison across mutations with explanation
                st.subheader("Treatment Response Across Mutations")
                st.write("Compare how different mutations respond to this treatment")
                st.info(f"This chart compares how different mutations respond to {selected_treatment}. The height of the bars represents the number of significantly up- or down-regulated genes in each mutation. Empty or missing bars indicate minimal response to the treatment.")
                
                # Get available mutations with enough data
                treatment_response_data = []
                
                for mutation in mutations:
                    result = treatment_effect_analysis(normalized_df, metadata_df, mutation, selected_treatment)
                    if result is not None and not result.empty:
                        num_sig_up = sum((result['significant']) & (result['log2FC'] > 0))
                        num_sig_down = sum((result['significant']) & (result['log2FC'] < 0))
                        
                        treatment_response_data.append({
                            'mutation': mutation,
                            'upregulated': num_sig_up,
                            'downregulated': num_sig_down,
                            'total_DEGs': num_sig_up + num_sig_down
                        })
                
                if treatment_response_data:
                    response_df = pd.DataFrame(treatment_response_data)
                    
                    # Plot bar chart of DEG counts by mutation with increased size
                    fig = px.bar(
                        response_df,
                        x='mutation',
                        y=['upregulated', 'downregulated'],
                        barmode='group',
                        title=f'Response to {selected_treatment} Across Mutations',
                        labels={'mutation': 'Mutation', 'value': 'Number of DEGs', 'variable': 'Regulation'}
                    )
                    fig.update_layout(width=900, height=600)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("Not enough data to compare treatment response across mutations")
                
                # Export results
                st.subheader("Export Results")
                
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')
                
                csv = convert_df_to_csv(treatment_results)
                st.download_button(
                    label="Download treatment effect results as CSV",
                    data=csv,
                    file_name=f'{selected_mutation_t}_{selected_treatment}_vs_DMSO_results.csv',
                    mime='text/csv',
                )
    
    # Tab 4: Pathway Analysis
    with tab4:
        st.header("ERK/AKT Pathway Analysis")
        
        # Analyze ERK/AKT pathway genes
        pathway_df, heatmap_df = analyze_erk_akt_pathway(normalized_df, metadata_df)
        
        if pathway_df.empty:
            st.warning("No ERK/AKT pathway genes found in the dataset")
        else:
            st.subheader("Expression of ERK/AKT Pathway Genes Across Mutations")
            st.info("This heatmap shows the expression levels of ERK/AKT pathway genes and EGFR ligands across different mutations. These genes are key regulators of cell growth, proliferation, and survival pathways often dysregulated in cancer.")
            
            # Number of pathway genes found
            st.write(f"Found {len(pathway_df)} out of 28 ERK/AKT pathway genes and EGFR ligands in the dataset")
            st.write("Genes found: " + ", ".join(pathway_df['symbol'].tolist()))
            
            # Heatmap of pathway gene expression
            # Drop gene_id column for heatmap
            heatmap_plot_df = heatmap_df.drop('gene_id', axis=1).copy()
            
            # Prepare data for heatmap
            heatmap_genes = heatmap_plot_df['gene'].tolist()
            heatmap_mutations = heatmap_plot_df.columns[1:].tolist()
            heatmap_values = heatmap_plot_df.set_index('gene')[heatmap_mutations].values
            
            # Create heatmap - rotated by transposing the matrix
            fig = px.imshow(
                heatmap_values.T,  # Transpose for rotation
                labels=dict(x="Gene", y="Mutation", color="Expression"),  # Swap x and y labels
                x=heatmap_genes,  # Genes on x-axis
                y=heatmap_mutations,  # Mutations on y-axis
                color_continuous_scale='Viridis',
                title="Expression of ERK/AKT Pathway Genes Across Mutations"
            )
            
            # Increase figure size and adjust for better visibility
            fig.update_layout(
                width=1000, 
                height=700,
                xaxis=dict(tickangle=45)  # Angle gene names for better readability
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Mutation correlations based on pathway genes with explanation
            st.subheader("Mutation Similarity Based on Pathway Gene Expression")
            st.info("This correlation heatmap shows how similar mutations are to each other based on the expression patterns of ERK/AKT pathway genes. Higher correlation (darker red) indicates more similar pathway activity.")
            
            # Calculate correlations
            corr_matrix = calculate_mutation_correlations(heatmap_df)
            
            # Create heatmap with increased size
            fig = px.imshow(
                corr_matrix,
                labels=dict(x="Mutation", y="Mutation", color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.index,
                color_continuous_scale='RdBu_r',
                title="Correlation Between Mutations Based on Pathway Gene Expression"
            )
            fig.update_layout(width=800, height=700)
            st.plotly_chart(fig, use_container_width=True)
            
            # EGFR ligand expression analysis with explanation
            st.subheader("EGFR Ligand Expression Analysis")
            st.info("This chart shows the expression levels of various EGFR ligands across mutations. These ligands can activate EGFR signaling and subsequent downstream pathways.")
            
            # Filter for EGFR ligands
            egfr_ligands = ['EGF', 'TGFA', 'HBEGF', 'BTC', 'AREG', 'EREG', 'EPGN']
            ligand_df = heatmap_df[heatmap_df['gene'].isin(egfr_ligands)].copy()
            
            if not ligand_df.empty:
                # Prepare data for bar chart
                ligand_data = []
                for idx, row in ligand_df.iterrows():
                    gene = row['gene']
                    for mutation in mutations:
                        if mutation in row:
                            ligand_data.append({
                                'gene': gene,
                                'mutation': mutation,
                                'expression': row[mutation]
                            })
                
                ligand_chart_df = pd.DataFrame(ligand_data)
                
                # Create bar chart with increased size
                fig = px.bar(
                    ligand_chart_df,
                    x='gene',
                    y='expression',
                    color='mutation',
                    barmode='group',
                    title='EGFR Ligand Expression Across Mutations',
                    labels={'gene': 'EGFR Ligand', 'expression': 'Expression Level', 'mutation': 'Mutation'}
                )
                fig.update_layout(width=900, height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No EGFR ligand genes found in the dataset")
                st.write("The EGFR ligands we look for are: EGF, TGFA, HBEGF, BTC, AREG, EREG, and EPGN.")
            
            # Treatment effect on pathway genes with explanation
            st.subheader("Treatment Effect on ERK/AKT Pathway Genes")
            st.info("This analysis shows how specific treatments affect the expression of ERK/AKT pathway genes in a selected mutation. It helps identify which pathway components are responsive to treatment.")
            
            # Select mutation and treatment
            col1, col2 = st.columns(2)
            with col1:
                selected_mutation_p = st.selectbox("Select mutation:", mutations, index=0, key="mutation_pathway")
            with col2:
                if treatments_no_dmso:
                    selected_treatment_p = st.selectbox("Select treatment:", treatments_no_dmso, key="treatment_pathway")
                else:
                    st.error("No treatments other than DMSO found in the data")
                    selected_treatment_p = None
            
            if selected_treatment_p:
                # Get treatment effect on pathway genes
                treatment_results = treatment_effect_analysis(
                    normalized_df[normalized_df['symbol'].isin(pathway_df['symbol'])],
                    metadata_df,
                    selected_mutation_p,
                    selected_treatment_p
                )
                
                if treatment_results is not None and not treatment_results.empty:
                    # Sort by absolute log2FC
                    treatment_results['abs_log2FC'] = abs(treatment_results['log2FC'])
                    treatment_results = treatment_results.sort_values('abs_log2FC', ascending=False)

                    # Add checkbox to filter for significant genes
                    show_significant_only_p = st.checkbox(
                        "Show significant genes only", 
                        value=False, 
                        key="sig_only_pathway",
                        help="When checked, only genes with adjusted p-value < 0.05 and |log2FC| > 1 will be highlighted"
                    )

                    # compute significance column
                    treatment_results['significant'] = (treatment_results['padj'] < 0.05) & (treatment_results['abs_log2FC'] > 1)
                    
                    # Filter data based on checkbox
                    if show_significant_only_p:
                        display_data = treatment_results[treatment_results['significant']]
                        if display_data.empty:
                            st.warning("No genes meet the significance criteria. Showing all genes instead.")
                            display_data = treatment_results
                    else:
                        display_data = treatment_results
                    
                    # Create bar chart with increased size
                    fig = px.bar(
                        display_data,
                        x='symbol',
                        y='log2FC',
                        color='significant',
                        title=f'Effect of {selected_treatment_p} on Pathway Genes in {selected_mutation_p}',
                        labels={'symbol': 'Gene', 'log2FC': 'log2 Fold Change', 'significant': 'Significant'}
                    )
                    fig.update_layout(width=900, height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add explanation for empty results
                    if sum(treatment_results['significant']) == 0:
                        st.write(f"No significant changes were detected in pathway genes after {selected_treatment_p} treatment in {selected_mutation_p} cells. Consider unchecking the 'Show significant genes only' option to see all changes.")
                else:
                    st.warning("Not enough data to analyze treatment effect on pathway genes")
                    st.write(f"There might not be enough replicates or the data quality is insufficient for {selected_mutation_p} with {selected_treatment_p} treatment.")

if __name__ == "__main__":
    main()