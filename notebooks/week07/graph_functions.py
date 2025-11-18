# import libraries
import mygene
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
# astropy is a library for astronomy and astrophysics! but has some very nice statistical correlation tools we want to use
import astropy.stats
from scipy.cluster.hierarchy import linkage, leaves_list

# Filter out low-expressed genes from the dataset
# As we are going to explore effects of different thresholds, we will create a function for this
def filter_low_expression_genes(data, threshold=1.0):
    """
    Filter out low-expressed genes from the dataset.

    Calculates the mean expression level for each gene and filters out
    genes whose mean expression level is below the specified threshold.

    Parameters:
    data (DataFrame): Expression data with genes as columns.
    threshold (float): Minimum mean expression level to retain a gene.
                       Default is 1.0.

    Returns:
    DataFrame: Filtered data with genes above the threshold.
    """
    # Calculate the mean expression for each gene
    gene_means = data.mean(axis=0)
    # Filter out genes with mean expression below the threshold
    mask = gene_means >= threshold
    filtered_data = data.loc[:, mask]
    return filtered_data

# Filter out genes based on their variance
# As we are going to explore effects of variance, we will create a function for this
def filter_high_variance_genes(data, threshold):
    """
    Filter out genes with variance below the specified threshold.

    Calculates the variance for each gene and filters out genes whose 
    variance is below the specified threshold.

    Parameters:
    data (DataFrame): Gene expression data with genes as columns and samples as rows.
    threshold (float): Minimum variance level to retain a gene.

    Returns:
    DataFrame: Filtered data with genes having variance above the threshold.
    """

    # Calculate the variance for each gene (column)
    gene_variances = data.var(axis=0)
    # Create a boolean mask to filter out genes with variance below the threshold
    mask = gene_variances >= threshold
    # Apply the mask to filter the DataFrame
    filtered_data = data.loc[:, mask]
    return filtered_data

# define a function to do the gene mapping
def rename_ensembl_to_gene_names(df, chunk_size=1000):
    """
    Renames Ensembl gene IDs to gene names using mygene.

    NB we chunk the requests to avoid hitting the rate limit.
    
    Parameters:
    df (pd.DataFrame): DataFrame with Ensembl gene IDs as columns.
    chunk_size (int): Number of Ensembl IDs to query at a time.
    
    Returns:
    pd.DataFrame: DataFrame with gene names as columns, excluding genes that couldn't be mapped.
    """
    
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Remove the `.number` suffix from ENSG IDs
    df_copy.columns = df_copy.columns.str.split('.').str[0]

    # Initialize mygene client
    mg = mygene.MyGeneInfo()

    # Split ENSG IDs into smaller chunks
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    ensg_ids = df_copy.columns.tolist()
    gene_mappings = []

    unmapped_genes = []

    # send requests in chunks
    for chunk in chunks(ensg_ids, chunk_size):
        result = mg.querymany(chunk, scopes='ensembl.gene', fields='symbol', species='human')
        gene_mappings.extend(result)

    # Create a mapping from ENSG to gene symbol, handle missing mappings
    ensg_to_gene = {item['query']: item.get('symbol', None) for item in gene_mappings}
    
    # Log the unmapped genes
    batch_unmapped_genes = [gene for gene in ensg_ids if ensg_to_gene.get(gene) is None]
    if batch_unmapped_genes:
        # Add unmapped genes to the list
        unmapped_genes.extend(batch_unmapped_genes)

    # Filter the DataFrame to only include columns that have been mapped
    df_filtered = df_copy.loc[:, df_copy.columns.isin(ensg_to_gene.keys())]

    # Further filter to ensure we have the same number of columns as mapped gene names
    df_filtered = df_filtered.loc[:, [ensg for ensg in df_filtered.columns if ensg_to_gene[ensg] is not None]]

    # Assign new column names
    df_filtered.columns = [ensg_to_gene[ensg] for ensg in df_filtered.columns]

    # Handle duplicate gene names by aggregating them (e.g., by taking the mean)
    df_final = df_filtered.T.groupby(df_filtered.columns).mean().T

    return df_final, set(unmapped_genes)

# function to calculate absolute biweight midcorrelation 
def calc_abs_bicorr(data):
    """
    Calculate the absolute biweight midcorrelation matrix for numeric data.

    Parameters:
    data (pd.DataFrame): Input DataFrame with numeric data.

    Returns:
    pd.DataFrame: DataFrame containing the absolute biweight midcorrelation matrix.
    """

    # Select only numeric data
    data = data._get_numeric_data()
    cols = data.columns
    idx = cols.copy()
    mat = data.to_numpy(dtype=float, na_value=np.nan, copy=False)
    mat = mat.T

    K = len(cols)
    correl = np.empty((K, K), dtype=np.float32)

    # Calculate biweight midcovariance
    bicorr = astropy.stats.biweight_midcovariance(mat, modify_sample_size=True)

    for i in range(K):
        for j in range(K):
            if i == j:
                correl[i, j] = 1.0
            else:
                denominator = np.sqrt(bicorr[i, i] * bicorr[j, j])
                if denominator != 0:
                    correl[i, j] = bicorr[i, j] / denominator
                else:
                    correl[i, j] = 0  # Or handle it in another appropriate way

    return pd.DataFrame(data=np.abs(correl), index=idx, columns=cols, dtype=np.float32)

# Create a graph from the correlation matrix using a specified threshold
def create_graph_from_correlation(correlation_matrix, threshold=0.8):
    """
    Creates a graph from a correlation matrix using a specified threshold.

    Parameters:
    correlation_matrix (pd.DataFrame): DataFrame containing the correlation matrix.
    threshold (float): Threshold for including edges based on correlation value.

    Returns:
    G (nx.Graph): Graph created from the correlation matrix.
    """
    G = nx.Graph()

    # Add nodes
    for node in correlation_matrix.columns:
        G.add_node(node)

    # Add edges with weights above the threshold
    for i in range(correlation_matrix.shape[0]):
        for j in range(i + 1, correlation_matrix.shape[1]):
            if i != j:  # Ignore the diagonal elements
                weight = correlation_matrix.iloc[i, j]
                if abs(weight) >= threshold:
                    G.add_edge(correlation_matrix.index[i], correlation_matrix.columns[j], weight=weight)

    return G

# Print basic information about the graph
def print_graph_info(G):
    """
    Print basic information about a NetworkX graph.

    
    Parameters:
    G (nx.Graph): The NetworkX graph.
    """
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print("Sample nodes:", list(G.nodes)[:10])  # Print first 10 nodes as a sample
    print("Sample edges:", list(G.edges(data=True))[:10])  # Print first 10 edges as a sample
    
    info_str = "Graph type: "
    is_directed = G.is_directed()
    if is_directed:
        info_str += "directed"
    else:
        info_str += "undirected"
    print(info_str)

    # Check for self-loops
    self_loops = list(nx.selfloop_edges(G))
    if self_loops:
        print(f"Number of self-loops: {len(self_loops)}")
        print("Self-loops:", self_loops)
    else:
        print("No self-loops in the graph.")

    # density of the graph
    density = nx.density(G)
    print(f"Graph density: {density}")

    # Find and print the number of connected components
    num_connected_components = nx.number_connected_components(G)
    print(f"Number of connected components: {num_connected_components}")

    # Calculate and print the clustering coefficient of the graph
    clustering_coeff = nx.average_clustering(G)
    print(f"Average clustering coefficient: {clustering_coeff}")
    
    # Function to visualize the graph
def visualise_graph(G, title='Gene Co-expression Network'):
    """
    Visualizes the graph using Matplotlib and NetworkX.

    Parameters:
    G (nx.Graph): Graph to visualize.
    title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, k=0.1)  # k controls the distance between nodes
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, width=0.2, alpha=0.5)
    plt.title(title)
    plt.show()

# here we have a function to plot the correlation matrices as heatmaps
def plot_correlation_matrices(correlation_matrices):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Adjust to 1 row and 3 columns
    axes = axes.flatten()
    
    for i, (key, matrix) in enumerate(correlation_matrices.items()):
        # Perform hierarchical clustering
        Z = linkage(matrix, method='average')  # You can use other methods like 'single', 'complete', etc.
        idx = leaves_list(Z)
        
        # Reorder matrix
        ordered_matrix = matrix.iloc[idx, :].iloc[:, idx]
        
        # Plot heatmap
        sns.heatmap(ordered_matrix, ax=axes[i], cmap='coolwarm', cbar=True, xticklabels=False, yticklabels=False)
        axes[i].set_title(f'{key.capitalize()} Correlation Matrix')
        
        # Set square aspect ratio
        axes[i].set_aspect('equal', adjustable='box')
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
# function to calculate the edge weight distribution
def visualise_edge_weight_distribution(G):
    """
    Visualizes the distribution of edge weights.

    Parameters:
    edge_weights (list): List of edge weights.
    """
    plt.figure(figsize=(10, 6))
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    # Histogram
    sns.histplot(edge_weights, bins=30, kde=False)
    
    plt.title('Distribution of Edge Weights')
    plt.xlabel('Edge Weight')
    plt.ylabel('Frequency')
    plt.show()
