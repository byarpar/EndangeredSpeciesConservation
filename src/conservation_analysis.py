# src/conservation_analysis.py
"""
Conservation Analysis Module

This module provides specialized analysis and reporting tools for conservation biologists,
environmental consultants, and wildlife management professionals. It includes functions
for calculating conservation metrics, prioritizing species for protection, estimating
resource requirements, and generating professional reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Define IUCN Red List categories and their numerical values
IUCN_CATEGORIES = {
    'Critically Endangered': 4,
    'Endangered': 3,
    'Vulnerable': 2,
    'Near Threatened': 1,
    'Least Concern': 0
}

# Reverse mapping for display
IUCN_CATEGORIES_REV = {v: k for k, v in IUCN_CATEGORIES.items()}

# Define colors for IUCN categories
IUCN_COLORS = {
    'Critically Endangered': '#d7191c',  # Red
    'Endangered': '#fdae61',             # Orange
    'Vulnerable': '#ffffbf',             # Yellow
    'Near Threatened': '#abd9e9',        # Light Blue
    'Least Concern': '#2c7bb6'           # Dark Blue
}

def calculate_conservation_priority_score(df):
    """
    Calculate a conservation priority score for each species based on multiple factors.
    
    The score considers:
    - Current conservation status
    - Population size (smaller populations get higher priority)
    - Reproduction rate (lower rates get higher priority)
    - Habitat loss (higher loss gets higher priority)
    - Climate change vulnerability (higher vulnerability gets higher priority)
    
    Args:
        df: DataFrame with species data
        
    Returns:
        DataFrame with added conservation priority score
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Map conservation status to numerical values
    if 'conservation_status' in result_df.columns:
        status_values = result_df['conservation_status'].map(IUCN_CATEGORIES).fillna(0)
    else:
        status_values = np.zeros(len(result_df))
    
    # Normalize population size (inverse relationship - smaller populations get higher priority)
    if 'population_size' in result_df.columns:
        pop_max = result_df['population_size'].max()
        pop_values = 1 - (result_df['population_size'] / pop_max)
    else:
        pop_values = np.zeros(len(result_df))
    
    # Normalize reproduction rate (inverse relationship - lower rates get higher priority)
    if 'reproduction_rate' in result_df.columns:
        repro_max = result_df['reproduction_rate'].max()
        repro_values = 1 - (result_df['reproduction_rate'] / repro_max)
    else:
        repro_values = np.zeros(len(result_df))
    
    # Use habitat loss directly (higher loss gets higher priority)
    if 'habitat_loss' in result_df.columns:
        habitat_values = result_df['habitat_loss']
    else:
        habitat_values = np.zeros(len(result_df))
    
    # Use temperature change as a proxy for climate vulnerability
    if 'temperature_change' in result_df.columns:
        temp_max = result_df['temperature_change'].max()
        temp_values = result_df['temperature_change'] / temp_max
    else:
        temp_values = np.zeros(len(result_df))
    
    # Calculate weighted priority score (0-100 scale)
    # Weights can be adjusted based on conservation priorities
    priority_score = (
        30 * (status_values / 4) +           # 30% weight to conservation status
        20 * pop_values +                    # 20% weight to population size
        15 * repro_values +                  # 15% weight to reproduction rate
        20 * habitat_values +                # 20% weight to habitat loss
        15 * temp_values                     # 15% weight to climate vulnerability
    ) * 100
    
    # Add to dataframe
    result_df['conservation_priority_score'] = priority_score.round(1)
    
    return result_df

def identify_priority_species(df, threshold=70):
    """
    Identify species that should be prioritized for conservation efforts.
    
    Args:
        df: DataFrame with species data including conservation_priority_score
        threshold: Priority score threshold (default: 70)
        
    Returns:
        DataFrame with priority species
    """
    if 'conservation_priority_score' not in df.columns:
        df = calculate_conservation_priority_score(df)
    
    # Filter species above the threshold
    priority_species = df[df['conservation_priority_score'] >= threshold].copy()
    
    # Sort by priority score (descending)
    priority_species = priority_species.sort_values('conservation_priority_score', ascending=False)
    
    return priority_species

def estimate_conservation_resources(df, base_cost=10000, area_factor=1000, population_factor=0.5):
    """
    Estimate resources needed for conservation efforts.
    
    Args:
        df: DataFrame with species data
        base_cost: Base cost per species in USD
        area_factor: Cost multiplier for habitat area
        population_factor: Cost reduction factor for larger populations
        
    Returns:
        DataFrame with added resource estimates
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Calculate conservation cost based on status, habitat loss, and population
    if 'conservation_status' in result_df.columns:
        status_values = result_df['conservation_status'].map(IUCN_CATEGORIES).fillna(0)
    else:
        status_values = np.zeros(len(result_df))
    
    # Status multiplier (higher status = higher cost)
    status_multiplier = 1 + (status_values / 2)  # 1x for Least Concern, 3x for Critically Endangered
    
    # Habitat loss factor (higher loss = higher cost)
    if 'habitat_loss' in result_df.columns:
        habitat_factor = 1 + result_df['habitat_loss']  # 1x to 2x based on habitat loss
    else:
        habitat_factor = np.ones(len(result_df))
    
    # Population size discount (larger populations get a discount)
    if 'population_size' in result_df.columns:
        pop_max = result_df['population_size'].max()
        pop_discount = 1 - (population_factor * result_df['population_size'] / pop_max)
    else:
        pop_discount = np.ones(len(result_df))
    
    # Calculate estimated conservation cost
    conservation_cost = base_cost * status_multiplier * habitat_factor * pop_discount
    
    # Add to dataframe
    result_df['estimated_conservation_cost'] = conservation_cost.round(0).astype(int)
    
    # Calculate staff resources needed (in person-months)
    staff_months = (conservation_cost / 5000).round(1)
    result_df['estimated_staff_months'] = staff_months
    
    return result_df

def cluster_species_by_traits(df, n_clusters=3):
    """
    Cluster species by their ecological traits to identify groups that might
    benefit from similar conservation approaches.
    
    Args:
        df: DataFrame with species data
        n_clusters: Number of clusters to create
        
    Returns:
        DataFrame with added cluster assignments
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Select features for clustering
    features = ['habitat_loss', 'temperature_change', 'pollution_level', 
                'population_size', 'reproduction_rate']
    
    # Filter features that exist in the dataframe
    features = [f for f in features if f in result_df.columns]
    
    if not features:
        print("No valid features found for clustering")
        return result_df
    
    # Extract and scale features
    X = result_df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster assignments to dataframe
    result_df['conservation_cluster'] = clusters
    
    return result_df, kmeans

def generate_conservation_report(df, output_dir='reports', filename=None):
    """
    Generate a comprehensive conservation report with visualizations.
    
    Args:
        df: DataFrame with species data
        output_dir: Directory to save the report
        filename: Name of the report file (default: auto-generated)
        
    Returns:
        Path to the generated report
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'conservation_report_{timestamp}.html'
    
    # Ensure we have all the necessary data
    if 'conservation_priority_score' not in df.columns:
        df = calculate_conservation_priority_score(df)
    
    if 'estimated_conservation_cost' not in df.columns:
        df = estimate_conservation_resources(df)
    
    # Create HTML report
    with open(os.path.join(output_dir, filename), 'w') as f:
        # Write HTML header
        f.write('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Conservation Priority Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #2c7bb6; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .critical { background-color: #ffdddd; }
                .endangered { background-color: #ffffcc; }
                .vulnerable { background-color: #e6f3ff; }
                .summary { background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                .chart-container { margin: 20px 0; text-align: center; }
                img { max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <h1>Conservation Priority Report</h1>
            <p>Generated on: ''' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '''</p>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <p>This report analyzes ''' + str(len(df)) + ''' species to identify conservation priorities 
                based on their conservation status, population trends, and environmental threats.</p>
                
                <h3>Key Findings:</h3>
                <ul>
        ''')
        
        # Add key findings
        priority_species = identify_priority_species(df)
        f.write(f'<li><strong>{len(priority_species)}</strong> species identified as high conservation priorities</li>')
        
        if 'conservation_status' in df.columns:
            status_counts = df['conservation_status'].value_counts()
            if 'Critically Endangered' in status_counts:
                f.write(f'<li><strong>{status_counts.get("Critically Endangered", 0)}</strong> Critically Endangered species</li>')
            if 'Endangered' in status_counts:
                f.write(f'<li><strong>{status_counts.get("Endangered", 0)}</strong> Endangered species</li>')
        
        total_cost = df['estimated_conservation_cost'].sum()
        f.write(f'<li>Estimated total conservation cost: <strong>${total_cost:,.2f}</strong></li>')
        
        f.write('''
                </ul>
            </div>
            
            <h2>Conservation Priority Analysis</h2>
        ''')
        
        # Generate and save priority score distribution chart
        plt.figure(figsize=(10, 6))
        sns.histplot(df['conservation_priority_score'], bins=20, kde=True)
        plt.title('Distribution of Conservation Priority Scores')
        plt.xlabel('Priority Score')
        plt.ylabel('Number of Species')
        plt.axvline(x=70, color='red', linestyle='--', label='Priority Threshold')
        plt.legend()
        
        chart_path = os.path.join(output_dir, 'priority_distribution.png')
        plt.savefig(chart_path)
        plt.close()
        
        f.write('''
            <div class="chart-container">
                <img src="priority_distribution.png" alt="Priority Score Distribution">
                <p>Distribution of conservation priority scores across all species.</p>
            </div>
        ''')
        
        # Top priority species table
        f.write('''
            <h2>Top Priority Species</h2>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Name</th>
                    <th>Conservation Status</th>
                    <th>Priority Score</th>
                    <th>Est. Conservation Cost</th>
                </tr>
        ''')
        
        # Sort by priority score and take top 10
        top_species = df.sort_values('conservation_priority_score', ascending=False).head(10)
        
        for _, species in top_species.iterrows():
            status = species.get('conservation_status', 'Unknown')
            row_class = ''
            
            if status == 'Critically Endangered':
                row_class = 'critical'
            elif status == 'Endangered':
                row_class = 'endangered'
            elif status == 'Vulnerable':
                row_class = 'vulnerable'
            
            f.write(f'''
                <tr class="{row_class}">
                    <td>{species.get('id', 'N/A')}</td>
                    <td>{species.get('name', 'Unknown')}</td>
                    <td>{status}</td>
                    <td>{species['conservation_priority_score']:.1f}</td>
                    <td>${species['estimated_conservation_cost']:,.2f}</td>
                </tr>
            ''')
        
        f.write('</table>')
        
        # Generate and save status breakdown chart
        if 'conservation_status' in df.columns:
            plt.figure(figsize=(10, 6))
            status_counts = df['conservation_status'].value_counts()
            colors = [IUCN_COLORS.get(status, '#999999') for status in status_counts.index]
            status_counts.plot(kind='bar', color=colors)
            plt.title('Species by Conservation Status')
            plt.xlabel('Conservation Status')
            plt.ylabel('Number of Species')
            plt.xticks(rotation=45)
            
            chart_path = os.path.join(output_dir, 'status_breakdown.png')
            plt.savefig(chart_path, bbox_inches='tight')
            plt.close()
            
            f.write('''
                <h2>Conservation Status Breakdown</h2>
                <div class="chart-container">
                    <img src="status_breakdown.png" alt="Conservation Status Breakdown">
                    <p>Distribution of species across IUCN Red List categories.</p>
                </div>
            ''')
        
        # Generate and save cost analysis chart
        plt.figure(figsize=(10, 6))
        
        if 'conservation_status' in df.columns:
            # Group by status and calculate mean cost
            status_costs = df.groupby('conservation_status')['estimated_conservation_cost'].mean().sort_values()
            colors = [IUCN_COLORS.get(status, '#999999') for status in status_costs.index]
            status_costs.plot(kind='bar', color=colors)
            plt.title('Average Conservation Cost by Status')
            plt.xlabel('Conservation Status')
            plt.ylabel('Average Cost (USD)')
            plt.xticks(rotation=45)
        else:
            # Just show overall cost distribution
            sns.histplot(df['estimated_conservation_cost'], bins=20)
            plt.title('Distribution of Estimated Conservation Costs')
            plt.xlabel('Estimated Cost (USD)')
            plt.ylabel('Number of Species')
        
        chart_path = os.path.join(output_dir, 'cost_analysis.png')
        plt.savefig(chart_path, bbox_inches='tight')
        plt.close()
        
        f.write('''
            <h2>Resource Requirements Analysis</h2>
            <div class="chart-container">
                <img src="cost_analysis.png" alt="Conservation Cost Analysis">
                <p>Analysis of estimated conservation costs across different conservation statuses.</p>
            </div>
        ''')
        
        # Generate threat analysis visualization if data available
        if all(col in df.columns for col in ['habitat_loss', 'temperature_change', 'pollution_level']):
            plt.figure(figsize=(12, 8))
            
            # Create a grid of subplots
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot habitat loss
            sns.boxplot(x='conservation_status', y='habitat_loss', data=df, ax=axes[0])
            axes[0].set_title('Habitat Loss by Conservation Status')
            axes[0].set_xlabel('Conservation Status')
            axes[0].set_ylabel('Habitat Loss')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Plot temperature change
            sns.boxplot(x='conservation_status', y='temperature_change', data=df, ax=axes[1])
            axes[1].set_title('Temperature Change by Conservation Status')
            axes[1].set_xlabel('Conservation Status')
            axes[1].set_ylabel('Temperature Change')
            axes[1].tick_params(axis='x', rotation=45)
            
            # Plot pollution level
            sns.boxplot(x='conservation_status', y='pollution_level', data=df, ax=axes[2])
            axes[2].set_title('Pollution Level by Conservation Status')
            axes[2].set_xlabel('Conservation Status')
            axes[2].set_ylabel('Pollution Level')
            axes[2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            chart_path = os.path.join(output_dir, 'threat_analysis.png')
            plt.savefig(chart_path, bbox_inches='tight')
            plt.close()
            
            f.write('''
                <h2>Threat Analysis</h2>
                <div class="chart-container">
                    <img src="threat_analysis.png" alt="Threat Analysis">
                    <p>Analysis of key threats (habitat loss, temperature change, pollution) across conservation statuses.</p>
                </div>
            ''')
        
        # Perform clustering analysis
        clustered_df, kmeans = cluster_species_by_traits(df)
        
        # Generate cluster visualization
        if 'conservation_cluster' in clustered_df.columns:
            # Create a 3D scatter plot if we have the right features
            if all(col in df.columns for col in ['habitat_loss', 'temperature_change', 'population_size']):
                from mpl_toolkits.mplot3d import Axes3D
                
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Plot each cluster
                for cluster in range(kmeans.n_clusters):
                    cluster_data = clustered_df[clustered_df['conservation_cluster'] == cluster]
                    ax.scatter(
                        cluster_data['habitat_loss'],
                        cluster_data['temperature_change'],
                        cluster_data['population_size'],
                        label=f'Cluster {cluster+1}'
                    )
                
                ax.set_xlabel('Habitat Loss')
                ax.set_ylabel('Temperature Change')
                ax.set_zlabel('Population Size')
                ax.set_title('Species Clusters by Key Traits')
                plt.legend()
                
                chart_path = os.path.join(output_dir, 'cluster_analysis.png')
                plt.savefig(chart_path, bbox_inches='tight')
                plt.close()
                
                f.write('''
                    <h2>Species Clustering Analysis</h2>
                    <div class="chart-container">
                        <img src="cluster_analysis.png" alt="Cluster Analysis">
                        <p>Clustering of species based on key ecological traits to identify groups that may benefit from similar conservation approaches.</p>
                    </div>
                ''')
                
                # Analyze clusters
                cluster_analysis = clustered_df.groupby('conservation_cluster').agg({
                    'habitat_loss': 'mean',
                    'temperature_change': 'mean',
                    'pollution_level': 'mean',
                    'population_size': 'mean',
                    'reproduction_rate': 'mean',
                    'conservation_priority_score': 'mean',
                    'estimated_conservation_cost': 'mean',
                    'id': 'count'
                }).round(2)
                
                cluster_analysis = cluster_analysis.rename(columns={'id': 'count'})
                
                f.write('''
                    <h3>Cluster Characteristics</h3>
                    <table>
                        <tr>
                            <th>Cluster</th>
                            <th>Count</th>
                            <th>Avg. Habitat Loss</th>
                            <th>Avg. Temp. Change</th>
                            <th>Avg. Population</th>
                            <th>Avg. Priority Score</th>
                            <th>Avg. Cost</th>
                        </tr>
                ''')
                
                for cluster, row in cluster_analysis.iterrows():
                    f.write(f'''
                        <tr>
                            <td>Cluster {cluster+1}</td>
                            <td>{int(row['count'])}</td>
                            <td>{row['habitat_loss']:.2f}</td>
                            <td>{row['temperature_change']:.2f}</td>
                            <td>{row['population_size']:.0f}</td>
                            <td>{row['conservation_priority_score']:.1f}</td>
                            <td>${row['estimated_conservation_cost']:,.2f}</td>
                        </tr>
                    ''')
                
                f.write('</table>')
        
        # Recommendations section
        f.write('''
            <h2>Conservation Recommendations</h2>
            <ol>
        ''')
        
        # Add recommendations based on analysis
        priority_count = len(identify_priority_species(df))
        
        f.write(f'''
            <li><strong>Prioritize {priority_count} high-risk species</strong> with conservation priority scores above 70 for immediate action.</li>
        ''')
        
        if 'conservation_status' in df.columns and 'Critically Endangered' in df['conservation_status'].values:
            f.write('''
                <li><strong>Develop emergency conservation plans</strong> for all Critically Endangered species, focusing on habitat protection and population recovery.</li>
            ''')
        
        if 'habitat_loss' in df.columns:
            high_habitat_loss = df[df['habitat_loss'] > 0.7]
            if len(high_habitat_loss) > 0:
                f.write(f'''
                    <li><strong>Address severe habitat loss</strong> for {len(high_habitat_loss)} species experiencing habitat degradation above 70%.</li>
                ''')
        
        if 'temperature_change' in df.columns:
            climate_vulnerable = df[df['temperature_change'] > 1.5]
            if len(climate_vulnerable) > 0:
                f.write(f'''
                    <li><strong>Implement climate adaptation strategies</strong> for {len(climate_vulnerable)} species highly vulnerable to temperature changes.</li>
                ''')
        
        if 'conservation_cluster' in clustered_df.columns:
            f.write(f'''
                <li><strong>Develop cluster-specific conservation approaches</strong> for the {kmeans.n_clusters} identified ecological groups to maximize resource efficiency.</li>
            ''')
        
        f.write(f'''
            <li><strong>Allocate an estimated ${total_cost:,.2f} in conservation funding</strong> across all species, with priority given to those with highest conservation scores.</li>
        ''')
        
        f.write('''
            </ol>
            
            <h2>Methodology</h2>
            <p>This report uses a multi-factor analysis approach to assess conservation priorities:</p>
            <ul>
                <li><strong>Conservation Priority Score:</strong> Calculated based on conservation status, population size, reproduction rate, habitat loss, and climate vulnerability.</li>
                <li><strong>Resource Estimation:</strong> Conservation costs estimated based on species status, habitat requirements, and population factors.</li>
                <li><strong>Cluster Analysis:</strong> Species grouped by ecological traits to identify groups that may benefit from similar conservation approaches.</li>
            </ul>
            
            <p><em>Report generated by the Enviro Wise Conservation Analysis System</em></p>
        </body>
        </html>
        ''')
    
    print(f"Conservation report generated: {os.path.join(output_dir, filename)}")
    return os.path.join(output_dir, filename)

def plot_conservation_priority_map(df, x_feature='habitat_loss', y_feature='population_size'):
    """
    Create a conservation priority heatmap to visualize species distribution
    across two key ecological variables.
    
    Args:
        df: DataFrame with species data
        x_feature: Feature to plot on x-axis
        y_feature: Feature to plot on y-axis
        
    Returns:
        matplotlib figure
    """
    if 'conservation_priority_score' not in df.columns:
        df = calculate_conservation_priority_score(df)
    
    # Check if features exist
    if x_feature not in df.columns or y_feature not in df.columns:
        print(f"Features {x_feature} and/or {y_feature} not found in dataframe")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create custom colormap for priority scores
    cmap = LinearSegmentedColormap.from_list(
        'priority_cmap', 
        [(0, 'green'), (0.5, 'yellow'), (0.7, 'orange'), (1, 'red')]
    )
    
    # Create scatter plot
    scatter = ax.scatter(
        df[x_feature], 
        df[y_feature],
        c=df['conservation_priority_score'],
        cmap=cmap,
        alpha=0.7,
        s=100,
        edgecolor='k'
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Conservation Priority Score')
    
    # Add labels and title
    ax.set_xlabel(x_feature.replace('_', ' ').title())
    ax.set_ylabel(y_feature.replace('_', ' ').title())
    ax.set_title('Conservation Priority Map')
    
    # If we have conservation status, add markers by status
    if 'conservation_status' in df.columns:
        # Create legend for conservation status
        status_markers = []
        for status, color in IUCN_COLORS.items():
            if status in df['conservation_status'].values:
                status_data = df[df['conservation_status'] == status]
                ax.scatter(
                    status_data[x_feature],
                    status_data[y_feature],
                    facecolors='none',
                    edgecolors=color,
                    linewidth=2,
                    s=150,
                    alpha=0.7
                )
                status_markers.append(mpatches.Patch(color=color, label=status))
        
        # Add legend
        ax.legend(handles=status_markers, title='Conservation Status', 
                 loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add threshold lines
    ax.axhline(y=df[y_feature].median(), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=df[x_feature].median(), color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig

def create_species_profile(species_data, output_dir='profiles'):
    """
    Create a detailed profile for a single species with visualizations and recommendations.
    
    Args:
        species_data: Series or dict with species data
        output_dir: Directory to save the profile
        
    Returns:
        Path to the generated profile
    """
    # Convert to Series if dict
    if isinstance(species_data, dict):
        species_data = pd.Series(species_data)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get species name or ID for filename
    species_name = species_data.get('name', species_data.get('id', 'unknown_species'))
    filename = f"{species_name.replace(' ', '_').lower()}_profile.html"
    
    # Calculate additional metrics if needed
    if 'conservation_priority_score' not in species_data:
        # Create a single-row DataFrame to use our existing functions
        temp_df = pd.DataFrame([species_data])
        temp_df = calculate_conservation_priority_score(temp_df)
        temp_df = estimate_conservation_resources(temp_df)
        species_data = temp_df.iloc[0]
    
    # Create HTML profile
    with open(os.path.join(output_dir, filename), 'w') as f:
        # Write HTML header
        f.write(f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Species Profile: {species_data.get('name', 'Unknown Species')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c7bb6; }}
                .profile-header {{ 
                    background-color: #f0f0f0; 
                    padding: 20px; 
                    border-radius: 5px; 
                    margin-bottom: 20px;
                    display: flex;
                    justify-content: space-between;
                }}
                .profile-header-text {{ flex: 2; }}
                .profile-header-status {{ 
                    flex: 1; 
                    text-align: center; 
                    padding: 15px; 
                    border-radius: 5px; 
                    margin-left: 20px;
                    color: white;
                    font-weight: bold;
                }}
                .metrics {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }}
                .metric-card {{ 
                    flex: 1; 
                    min-width: 200px; 
                    background-color: #f9f9f9; 
                    padding: 15px; 
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                .chart-container {{ margin: 20px 0; }}
                .recommendations {{ background-color: #e6f3ff; padding: 15px; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
        ''')
        
        # Header with status color
        status = species_data.get('conservation_status', 'Unknown')
        status_color = IUCN_COLORS.get(status, '#999999')
        
        f.write(f'''
            <div class="profile-header">
                <div class="profile-header-text">
                    <h1>{species_data.get('name', 'Unknown Species')}</h1>
                    <p><strong>ID:</strong> {species_data.get('id', 'N/A')}</p>
                    <p><strong>Habitat:</strong> {species_data.get('habitat', 'Unknown')}</p>
                    <p><strong>Region:</strong> {species_data.get('region', 'Unknown')}</p>
                </div>
                <div class="profile-header-status" style="background-color: {status_color};">
                    <div style="font-size: 18px;">Conservation Status</div>
                    <div style="font-size: 24px; margin-top: 10px;">{status}</div>
                </div>
            </div>
            
            <h2>Conservation Metrics</h2>
            <div class="metrics">
                <div class="metric-card">
                    <div>Priority Score</div>
                    <div class="metric-value">{species_data.get('conservation_priority_score', 'N/A')}</div>
                    <div>Out of 100</div>
                </div>
                
                <div class="metric-card">
                    <div>Population Size</div>
        ''')
        
        # Handle population size formatting - check if it's a number or string
        pop_size = species_data.get('population_size', 'N/A')
        if isinstance(pop_size, (int, float)):
            f.write(f'<div class="metric-value">{pop_size:,}</div>')
        else:
            f.write(f'<div class="metric-value">{pop_size}</div>')
            
        f.write('''
                    <div>Individuals</div>
                </div>
                
                <div class="metric-card">
                    <div>Habitat Loss</div>
        ''')
        
        # Handle habitat loss formatting - check if it's a number or string
        habitat_loss = species_data.get('habitat_loss', 'N/A')
        if isinstance(habitat_loss, (int, float)):
            f.write(f'<div class="metric-value">{habitat_loss:.1%}</div>')
        else:
            f.write(f'<div class="metric-value">{habitat_loss}</div>')
            
        f.write('''
                    <div>Of original habitat</div>
                </div>
                
                <div class="metric-card">
                    <div>Est. Conservation Cost</div>
        ''')
        
        # Handle conservation cost formatting - check if it's a number or string
        cons_cost = species_data.get('estimated_conservation_cost', 'N/A')
        if isinstance(cons_cost, (int, float)):
            f.write(f'<div class="metric-value">${cons_cost:,.0f}</div>')
        else:
            f.write(f'<div class="metric-value">{cons_cost}</div>')
            
        f.write('''
                    <div>USD</div>
                </div>
            </div>
            
            <h2>Threat Analysis</h2>
            <table>
                <tr>
                    <th>Threat Factor</th>
                    <th>Value</th>
                    <th>Severity</th>
                </tr>
        ''')
        
        # Add threat factors
        threats = [
            ('Primary Threat', species_data.get('primary_threat', 'Unknown'), None),
            ('Habitat Loss', species_data.get('habitat_loss', 'N/A'), 
             'High' if isinstance(species_data.get('habitat_loss', 0), (int, float)) and species_data.get('habitat_loss', 0) > 0.6 else 
             'Medium' if isinstance(species_data.get('habitat_loss', 0), (int, float)) and species_data.get('habitat_loss', 0) > 0.3 else 'Low'),
            ('Temperature Change', species_data.get('temperature_change', 'N/A'), 
             'High' if isinstance(species_data.get('temperature_change', 0), (int, float)) and species_data.get('temperature_change', 0) > 1.5 else 
             'Medium' if isinstance(species_data.get('temperature_change', 0), (int, float)) and species_data.get('temperature_change', 0) > 0.8 else 'Low'),
            ('Pollution Level', species_data.get('pollution_level', 'N/A'), 
             'High' if isinstance(species_data.get('pollution_level', 0), (int, float)) and species_data.get('pollution_level', 0) > 0.6 else 
             'Medium' if isinstance(species_data.get('pollution_level', 0), (int, float)) and species_data.get('pollution_level', 0) > 0.3 else 'Low'),
        ]
        
        for threat, value, severity in threats:
            # Format value if it's a number
            if isinstance(value, (int, float)):
                if threat == 'Habitat Loss' or threat == 'Pollution Level':
                    value_str = f"{value:.1%}"
                else:
                    value_str = f"{value:.2f}"
            else:
                value_str = str(value)
            
            # Set color based on severity
            color = ''
            if severity == 'High':
                color = 'background-color: #ffdddd;'
            elif severity == 'Medium':
                color = 'background-color: #ffffcc;'
            elif severity == 'Low':
                color = 'background-color: #e6f3ff;'
            
            f.write(f'''
                <tr style="{color}">
                    <td>{threat}</td>
                    <td>{value_str}</td>
                    <td>{severity if severity else 'N/A'}</td>
                </tr>
            ''')
        
        f.write('</table>')
        
        # Conservation recommendations
        priority_score = species_data.get('conservation_priority_score', 0)
        status = species_data.get('conservation_status', 'Unknown')
        habitat_loss = species_data.get('habitat_loss', 0)
        population_size = species_data.get('population_size', 0)
        
        f.write('''
            <h2>Conservation Recommendations</h2>
            <div class="recommendations">
                <ol>
        ''')
        
        # Generate recommendations based on species data
        if status == 'Critically Endangered' or status == 'Endangered':
            f.write('''
                <li><strong>Implement emergency conservation measures</strong> including habitat protection, captive breeding programs, and strict enforcement of protection laws.</li>
            ''')
        
        if isinstance(habitat_loss, (int, float)) and habitat_loss > 0.5:
            f.write('''
                <li><strong>Prioritize habitat restoration and protection</strong> to counter significant habitat loss. Establish protected areas and corridors to connect fragmented habitats.</li>
            ''')
        
        if isinstance(population_size, (int, float)) and population_size < 1000:
            f.write('''
                <li><strong>Develop population recovery plan</strong> with genetic management strategies to maintain genetic diversity in this small population.</li>
            ''')
        
        temp_change = species_data.get('temperature_change', 0)
        if isinstance(temp_change, (int, float)) and temp_change > 1.0:
            f.write('''
                <li><strong>Implement climate adaptation strategies</strong> including identifying and protecting climate refugia and considering assisted migration if necessary.</li>
            ''')
        
        pollution = species_data.get('pollution_level', 0)
        if isinstance(pollution, (int, float)) and pollution > 0.5:
            f.write('''
                <li><strong>Address pollution sources</strong> affecting this species' habitat through policy advocacy, cleanup efforts, and pollution monitoring programs.</li>
            ''')
        
        # Handle conservation cost formatting for recommendation
        cons_cost = species_data.get('estimated_conservation_cost', 0)
        if isinstance(cons_cost, (int, float)):
            f.write(f'''
                <li><strong>Secure funding of approximately ${cons_cost:,.0f}</strong> for conservation efforts over the next 5 years.</li>
            ''')
        else:
            f.write('''
                <li><strong>Secure appropriate funding</strong> for conservation efforts over the next 5 years.</li>
            ''')
            
        f.write('''
                <li><strong>Establish monitoring program</strong> to track population trends, habitat conditions, and effectiveness of conservation interventions.</li>
            </ol>
            </div>
            
            <p><em>Profile generated by the Enviro Wise Conservation Analysis System on {}</em></p>
        </body>
        </html>
        '''.format(datetime.now().strftime('%Y-%m-%d')))
    
    print(f"Species profile generated: {os.path.join(output_dir, filename)}")
    return os.path.join(output_dir, filename)

def create_temporal_trend_visualization(df, years=5, output_dir='reports'):
    """
    Create a visualization showing temporal trends in conservation status.
    
    This visualization helps conservation professionals track changes over time
    and evaluate the effectiveness of conservation interventions.
    
    Args:
        df: DataFrame with species data
        years: Number of years to simulate in the trend
        output_dir: Directory to save the visualization
        
    Returns:
        Path to the generated visualization
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a figure
    plt.figure(figsize=(12, 8))
    
    # Define status categories and colors
    statuses = ['Critically Endangered', 'Endangered', 'Vulnerable', 
                'Near Threatened', 'Least Concern']
    colors = [IUCN_COLORS[status] for status in statuses]
    
    # Generate simulated historical data
    # In a real application, this would use actual historical data
    np.random.seed(42)  # For reproducibility
    
    # Current counts (from the dataframe)
    current_counts = {}
    if 'conservation_status' in df.columns:
        for status in statuses:
            current_counts[status] = len(df[df['conservation_status'] == status])
    else:
        # If no status data, create random data
        total = len(df)
        current_counts = {
            'Critically Endangered': int(total * 0.1),
            'Endangered': int(total * 0.15),
            'Vulnerable': int(total * 0.2),
            'Near Threatened': int(total * 0.25),
            'Least Concern': int(total * 0.3)
        }
    
    # Generate historical trends
    historical_data = {}
    for status in statuses:
        # Start with current count
        current = current_counts.get(status, 0)
        
        # Generate historical data with some random variation
        # Different trends for different statuses
        if status == 'Critically Endangered':
            # Increasing trend (worsening situation)
            trend = np.linspace(current * 0.7, current, years)
            noise = np.random.normal(0, current * 0.05, years)
        elif status == 'Endangered':
            # Slightly increasing trend
            trend = np.linspace(current * 0.8, current, years)
            noise = np.random.normal(0, current * 0.04, years)
        elif status == 'Vulnerable':
            # Stable with fluctuations
            trend = np.ones(years) * current
            noise = np.random.normal(0, current * 0.08, years)
        elif status == 'Near Threatened':
            # Slightly decreasing (improving)
            trend = np.linspace(current * 1.1, current, years)
            noise = np.random.normal(0, current * 0.06, years)
        else:  # Least Concern
            # Increasing (improving)
            trend = np.linspace(current * 0.9, current, years)
            noise = np.random.normal(0, current * 0.03, years)
        
        # Combine trend and noise, ensure positive values
        historical_data[status] = np.maximum(trend + noise, 0).astype(int)
    
    # Create x-axis (years)
    current_year = datetime.now().year
    x_years = range(current_year - years + 1, current_year + 1)
    
    # Plot the data
    for i, status in enumerate(statuses):
        plt.plot(x_years, historical_data[status], marker='o', linewidth=2, 
                 color=colors[i], label=status)
    
    # Add conservation intervention markers
    # In a real application, these would be actual intervention dates
    interventions = {
        current_year - 3: "Habitat Protection Policy",
        current_year - 2: "Captive Breeding Program",
        current_year - 1: "Anti-poaching Initiative"
    }
    
    for year, label in interventions.items():
        plt.axvline(x=year, color='gray', linestyle='--', alpha=0.7)
        plt.text(year, plt.ylim()[1] * 0.95, label, rotation=90, 
                 verticalalignment='top', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Number of Species')
    plt.title('Temporal Trends in Conservation Status')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    output_path = os.path.join(output_dir, 'temporal_trends.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_funding_impact_visualization(df, output_dir='reports'):
    """
    Create a visualization showing the relationship between conservation funding
    and conservation outcomes.
    
    This visualization helps conservation managers and funders understand the
    return on investment for conservation efforts.
    
    Args:
        df: DataFrame with species data
        output_dir: Directory to save the visualization
        
    Returns:
        Path to the generated visualization
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check if required columns exist
    required_cols = ['conservation_funding', 'vulnerability_score', 'conservation_status']
    if not all(col in df.columns for col in required_cols):
        # Create synthetic data if needed
        df = df.copy()
        if 'conservation_funding' not in df.columns:
            df['conservation_funding'] = np.random.lognormal(10, 1, size=len(df))
        if 'vulnerability_score' not in df.columns:
            df['vulnerability_score'] = np.random.beta(2, 5, size=len(df))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 1. Funding vs. Vulnerability Score
    if 'conservation_status' in df.columns:
        # Color by conservation status
        status_colors = [IUCN_COLORS.get(status, '#999999') for status in df['conservation_status']]
        scatter = ax1.scatter(df['conservation_funding'], df['vulnerability_score'], 
                             c=status_colors, alpha=0.7, s=50)
        
        # Create legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=IUCN_COLORS[status], 
                                     markersize=8, label=status)
                          for status in IUCN_COLORS if status in df['conservation_status'].values]
        ax1.legend(handles=legend_elements, title='Conservation Status')
    else:
        # Simple scatter plot if no status information
        ax1.scatter(df['conservation_funding'], df['vulnerability_score'], 
                   alpha=0.7, s=50, color='#2c7bb6')
    
    # Add trendline
    z = np.polyfit(df['conservation_funding'], df['vulnerability_score'], 1)
    p = np.poly1d(z)
    ax1.plot(sorted(df['conservation_funding']), p(sorted(df['conservation_funding'])), 
            'r--', alpha=0.7)
    
    # Calculate correlation
    corr = np.corrcoef(df['conservation_funding'], df['vulnerability_score'])[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=ax1.transAxes,
            bbox=dict(facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('Conservation Funding (USD)')
    ax1.set_ylabel('Vulnerability Score')
    ax1.set_title('Funding Impact on Vulnerability')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Average funding by conservation status
    if 'conservation_status' in df.columns:
        # Group by status and calculate mean funding
        status_funding = df.groupby('conservation_status')['conservation_funding'].mean().sort_values()
        
        # Create bar chart
        bars = ax2.bar(status_funding.index, status_funding.values, 
                      color=[IUCN_COLORS.get(status, '#999999') for status in status_funding.index])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 5000,
                    f'${height:,.0f}',
                    ha='center', va='bottom', rotation=0)
        
        ax2.set_xlabel('Conservation Status')
        ax2.set_ylabel('Average Funding (USD)')
        ax2.set_title('Average Funding by Conservation Status')
        plt.xticks(rotation=45)
        ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
    else:
        # If no status information, show funding distribution
        ax2.hist(df['conservation_funding'], bins=15, color='#2c7bb6', alpha=0.7)
        ax2.set_xlabel('Conservation Funding (USD)')
        ax2.set_ylabel('Number of Species')
        ax2.set_title('Distribution of Conservation Funding')
        ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'funding_impact.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_intervention_effectiveness_chart(df, output_dir='reports'):
    """
    Create a visualization showing the effectiveness of different conservation
    interventions.
    
    This visualization helps conservation managers evaluate which interventions
    are most effective for different species and threats.
    
    Args:
        df: DataFrame with species data
        output_dir: Directory to save the visualization
        
    Returns:
        Path to the generated visualization
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define intervention types and their simulated effectiveness
    # In a real application, this would be based on actual intervention data
    interventions = [
        'Habitat Protection',
        'Captive Breeding',
        'Anti-poaching Measures',
        'Habitat Restoration',
        'Legal Protection',
        'Public Education',
        'Translocation',
        'Invasive Species Control'
    ]
    
    # Simulated effectiveness data by species group
    # In a real application, this would be based on actual outcome data
    np.random.seed(42)  # For reproducibility
    
    # Get species groups from data or use defaults
    if 'species_group' in df.columns:
        species_groups = df['species_group'].unique()
    else:
        species_groups = ['Mammal', 'Bird', 'Reptile', 'Amphibian', 'Fish', 'Plant']
    
    # Generate effectiveness data (0-100%)
    effectiveness_data = {}
    for group in species_groups:
        # Different interventions have different effectiveness for different groups
        if group == 'Mammal':
            base = np.array([75, 65, 80, 60, 70, 50, 55, 65])
        elif group == 'Bird':
            base = np.array([70, 60, 65, 75, 65, 60, 70, 55])
        elif group == 'Reptile':
            base = np.array([65, 55, 60, 70, 60, 45, 65, 75])
        elif group == 'Amphibian':
            base = np.array([60, 50, 45, 80, 55, 50, 60, 70])
        elif group == 'Fish':
            base = np.array([55, 45, 50, 65, 60, 55, 40, 80])
        else:  # Plant or other
            base = np.array([80, 40, 30, 85, 65, 60, 70, 75])
        
        # Add some random variation
        noise = np.random.normal(0, 5, len(interventions))
        effectiveness = np.clip(base + noise, 0, 100)
        effectiveness_data[group] = effectiveness
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Create a grouped bar chart
    x = np.arange(len(interventions))
    width = 0.8 / len(species_groups)
    
    # Plot bars for each species group
    for i, group in enumerate(species_groups):
        offset = (i - len(species_groups)/2 + 0.5) * width
        plt.bar(x + offset, effectiveness_data[group], width, label=group)
    
    # Add labels and title
    plt.xlabel('Intervention Type')
    plt.ylabel('Effectiveness (%)')
    plt.title('Conservation Intervention Effectiveness by Species Group')
    plt.xticks(x, interventions, rotation=45, ha='right')
    plt.legend(title='Species Group')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add effectiveness threshold line
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Effectiveness Threshold')
    
    # Ensure everything fits
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'intervention_effectiveness.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    # Create a second visualization: heatmap of intervention effectiveness
    plt.figure(figsize=(12, 8))
    
    # Convert effectiveness data to a matrix
    effectiveness_matrix = np.array([effectiveness_data[group] for group in species_groups])
    
    # Create heatmap
    sns.heatmap(effectiveness_matrix, annot=True, fmt='.1f', cmap='YlGnBu',
               xticklabels=interventions, yticklabels=species_groups)
    
    plt.title('Intervention Effectiveness Heatmap')
    plt.tight_layout()
    
    # Save the heatmap
    heatmap_path = os.path.join(output_dir, 'intervention_heatmap.png')
    plt.savefig(heatmap_path, bbox_inches='tight')
    plt.close()
    
    return output_path, heatmap_path

def create_resource_allocation_visualization(df, output_dir='reports'):
    """
    Create a visualization showing optimal resource allocation for conservation.
    
    This visualization helps conservation managers and funders make decisions
    about how to allocate limited resources across species and interventions.
    
    Args:
        df: DataFrame with species data
        output_dir: Directory to save the visualization
        
    Returns:
        Path to the generated visualization
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate total estimated conservation cost if available
    if 'estimated_conservation_cost' in df.columns:
        total_cost = df['estimated_conservation_cost'].sum()
    else:
        # Generate synthetic data if not available
        df = df.copy()
        df['estimated_conservation_cost'] = np.random.lognormal(10, 1, size=len(df))
        total_cost = df['estimated_conservation_cost'].sum()
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Cost allocation by conservation status (pie chart)
    ax1 = fig.add_subplot(gs[0, 0])
    
    if 'conservation_status' in df.columns:
        # Group by status and sum costs
        status_costs = df.groupby('conservation_status')['estimated_conservation_cost'].sum()
        
        # Create pie chart
        wedges, texts, autotexts = ax1.pie(
            status_costs, 
            labels=status_costs.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=[IUCN_COLORS.get(status, '#999999') for status in status_costs.index]
        )
        
        # Style the text
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_color('white')
        
        ax1.set_title('Resource Allocation by Conservation Status')
    else:
        # If no status information, show a message
        ax1.text(0.5, 0.5, 'Conservation status data not available', 
                ha='center', va='center', fontsize=12)
        ax1.axis('off')
    
    # 2. Cost allocation by species group (horizontal bar chart)
    ax2 = fig.add_subplot(gs[0, 1])
    
    if 'species_group' in df.columns:
        # Group by species group and sum costs
        group_costs = df.groupby('species_group')['estimated_conservation_cost'].sum().sort_values()
        
        # Calculate percentages
        percentages = (group_costs / group_costs.sum()) * 100
        
        # Create horizontal bar chart
        bars = ax2.barh(group_costs.index, group_costs.values)
        
        # Add percentage labels
        for i, (cost, percentage) in enumerate(zip(group_costs.values, percentages)):
            ax2.text(cost + (total_cost * 0.01), i, f'{percentage:.1f}%', 
                    va='center', fontsize=9)
        
        ax2.set_xlabel('Estimated Cost (USD)')
        ax2.set_title('Resource Allocation by Species Group')
        ax2.grid(True, linestyle='--', alpha=0.7, axis='x')
    else:
        # If no species group information, show a message
        ax2.text(0.5, 0.5, 'Species group data not available', 
                ha='center', va='center', fontsize=12)
        ax2.axis('off')
    
    # 3. Optimal resource allocation based on priority score (scatter plot)
    ax3 = fig.add_subplot(gs[1, :])
    
    if 'conservation_priority_score' in df.columns:
        # Create scatter plot of cost vs. priority score
        scatter = ax3.scatter(
            df['conservation_priority_score'], 
            df['estimated_conservation_cost'],
            alpha=0.7,
            s=50,
            c=df['estimated_conservation_cost'] / df['conservation_priority_score'],
            cmap='viridis'
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Cost per Priority Point (lower is better)')
        
        # Add quadrant lines
        median_score = df['conservation_priority_score'].median()
        median_cost = df['estimated_conservation_cost'].median()
        
        ax3.axvline(x=median_score, color='gray', linestyle='--', alpha=0.7)
        ax3.axhline(y=median_cost, color='gray', linestyle='--', alpha=0.7)
        
        # Add quadrant labels
        ax3.text(df['conservation_priority_score'].min() + 5, median_cost * 1.1, 
                'Low Priority, High Cost', fontsize=10, ha='left')
        ax3.text(median_score * 1.1, df['estimated_conservation_cost'].min() + 1000, 
                'High Priority, Low Cost', fontsize=10, ha='left')
        
        # Add regression line
        z = np.polyfit(df['conservation_priority_score'], df['estimated_conservation_cost'], 1)
        p = np.poly1d(z)
        ax3.plot(sorted(df['conservation_priority_score']), 
                p(sorted(df['conservation_priority_score'])), 
                'r--', alpha=0.7)
        
        # Highlight optimal allocation region
        optimal_species = df[
            (df['conservation_priority_score'] > median_score) & 
            (df['estimated_conservation_cost'] < median_cost)
        ]
        
        if len(optimal_species) > 0:
            ax3.scatter(
                optimal_species['conservation_priority_score'],
                optimal_species['estimated_conservation_cost'],
                s=100,
                facecolors='none',
                edgecolors='red',
                linewidth=2,
                label='Optimal Allocation'
            )
            
            # Add annotation about optimal allocation
            optimal_count = len(optimal_species)
            optimal_cost = optimal_species['estimated_conservation_cost'].sum()
            optimal_pct = (optimal_cost / total_cost) * 100
            
            ax3.text(0.02, 0.98, 
                    f'Optimal Allocation:\n{optimal_count} species\n${optimal_cost:,.0f} ({optimal_pct:.1f}% of total budget)',
                    transform=ax3.transAxes,
                    va='top',
                    bbox=dict(facecolor='white', alpha=0.8))
        
        ax3.set_xlabel('Conservation Priority Score')
        ax3.set_ylabel('Estimated Cost (USD)')
        ax3.set_title('Optimal Resource Allocation Strategy')
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend()
    else:
        # If no priority score information, show a message
        ax3.text(0.5, 0.5, 'Conservation priority score data not available', 
                ha='center', va='center', fontsize=12)
        ax3.axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'resource_allocation.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_habitat_restoration_tracker(df, output_dir='reports'):
    """
    Create a visualization tracking habitat restoration progress.
    
    This visualization helps conservation field workers and managers track
    the progress of habitat restoration efforts over time.
    
    Args:
        df: DataFrame with species data
        output_dir: Directory to save the visualization
        
    Returns:
        Path to the generated visualization
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Define habitat types
    if 'habitat' in df.columns:
        habitat_types = df['habitat'].unique()
    else:
        habitat_types = [
            'Tropical Forest', 'Temperate Forest', 'Grassland', 'Wetland', 
            'Marine', 'Coral Reef', 'Desert', 'Mountain'
        ]
    
    # Generate simulated restoration data
    # In a real application, this would be based on actual restoration data
    np.random.seed(42)  # For reproducibility
    
    # 1. Current restoration progress by habitat type
    restoration_progress = {}
    restoration_targets = {}
    
    for habitat in habitat_types:
        # Random progress between 10% and 90%
        progress = np.random.uniform(0.1, 0.9)
        restoration_progress[habitat] = progress
        
        # Target is always higher than current progress
        target = min(1.0, progress + np.random.uniform(0.1, 0.3))
        restoration_targets[habitat] = target
    
    # Sort by progress
    sorted_habitats = sorted(restoration_progress.keys(), 
                            key=lambda x: restoration_progress[x])
    
    # Create progress bars
    for i, habitat in enumerate(sorted_habitats):
        progress = restoration_progress[habitat]
        target = restoration_targets[habitat]
        
        # Progress bar
        ax1.barh(i, progress, color='#2c7bb6', alpha=0.7)
        
        # Target marker
        ax1.plot([target], [i], 'ro', markersize=8)
        
        # Progress percentage
        ax1.text(progress + 0.02, i, f'{progress:.0%}', va='center')
    
    ax1.set_yticks(range(len(sorted_habitats)))
    ax1.set_yticklabels(sorted_habitats)
    ax1.set_xlim(0, 1.1)
    ax1.set_xlabel('Restoration Progress')
    ax1.set_title('Habitat Restoration Progress')
    ax1.grid(True, linestyle='--', alpha=0.7, axis='x')
    
    # Add legend for target
    ax1.plot([], [], 'ro', markersize=8, label='Target')
    ax1.legend(loc='lower right')
    
    # 2. Temporal progress for a selected habitat
    # Choose the habitat with median progress
    median_habitat = sorted_habitats[len(sorted_habitats) // 2]
    
    # Generate temporal data (quarterly for 3 years)
    quarters = 12
    quarters_labels = [f'Q{i%4+1}\n{2023+i//4}' for i in range(quarters)]
    
    # Generate progress data with an upward trend
    start_progress = 0.1
    current_progress = restoration_progress[median_habitat]
    
    # Create a sigmoid-like curve for progress
    x = np.linspace(-5, 5, quarters)
    sigmoid = 1 / (1 + np.exp(-x))
    progress_values = start_progress + (current_progress - start_progress) * sigmoid
    
    # Add some random noise
    noise = np.random.normal(0, 0.02, quarters)
    progress_values = np.clip(progress_values + noise, 0, 1)
    
    # Plot the progress line
    ax2.plot(range(quarters), progress_values, marker='o', linewidth=2, 
             color='#2c7bb6', label='Actual Progress')
    
    # Add target line
    target_line = np.ones(quarters) * restoration_targets[median_habitat]
    ax2.plot(range(quarters), target_line, 'r--', label='Target')
    
    # Add intervention markers
    interventions = {
        2: "Initial Planting",
        5: "Invasive Species Removal",
        8: "Supplemental Planting"
    }
    
    for quarter, label in interventions.items():
        ax2.axvline(x=quarter, color='gray', linestyle='--', alpha=0.7)
        ax2.text(quarter, ax2.get_ylim()[0] + 0.05, label, rotation=90, 
                 verticalalignment='bottom', alpha=0.7)
    
    ax2.set_xticks(range(quarters))
    ax2.set_xticklabels(quarters_labels, rotation=45)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel('Restoration Progress')
    ax2.set_title(f'Restoration Progress Over Time: {median_habitat}')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'habitat_restoration_tracker.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_stakeholder_engagement_visualization(df, output_dir='reports'):
    """
    Create a visualization showing stakeholder engagement in conservation efforts.
    
    This visualization helps conservation managers understand and improve
    stakeholder engagement strategies.
    
    Args:
        df: DataFrame with species data
        output_dir: Directory to save the visualization
        
    Returns:
        Path to the generated visualization
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define stakeholder groups and their characteristics
    stakeholders = [
        'Local Communities',
        'Government Agencies',
        'NGOs',
        'Research Institutions',
        'Private Sector',
        'Indigenous Groups',
        'Tourists/Visitors'
    ]
    
    # Define engagement metrics
    metrics = [
        'Involvement Level',
        'Resource Contribution',
        'Knowledge Sharing',
        'Decision Influence',
        'Benefit Sharing'
    ]
    
    # Generate simulated engagement data
    # In a real application, this would be based on actual stakeholder data
    np.random.seed(42)  # For reproducibility
    
    engagement_data = {}
    for stakeholder in stakeholders:
        # Different base values for different stakeholders
        if stakeholder == 'Local Communities':
            base = np.array([0.7, 0.4, 0.6, 0.5, 0.8])
        elif stakeholder == 'Government Agencies':
            base = np.array([0.8, 0.7, 0.5, 0.9, 0.6])
        elif stakeholder == 'NGOs':
            base = np.array([0.9, 0.8, 0.9, 0.7, 0.7])
        elif stakeholder == 'Research Institutions':
            base = np.array([0.6, 0.5, 0.9, 0.6, 0.4])
        elif stakeholder == 'Private Sector':
            base = np.array([0.5, 0.8, 0.4, 0.6, 0.5])
        elif stakeholder == 'Indigenous Groups':
            base = np.array([0.7, 0.3, 0.8, 0.4, 0.7])
        else:  # Tourists/Visitors
            base = np.array([0.4, 0.2, 0.3, 0.1, 0.3])
        
        # Add some random variation
        noise = np.random.normal(0, 0.05, len(metrics))
        values = np.clip(base + noise, 0, 1)
        engagement_data[stakeholder] = values
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Radar chart for stakeholder engagement
    ax1 = fig.add_subplot(2, 2, 1, polar=True)
    
    # Number of metrics
    N = len(metrics)
    
    # Angle for each metric
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Plot each stakeholder
    for stakeholder in stakeholders:
        values = engagement_data[stakeholder].tolist()
        values += values[:1]  # Close the loop
        
        ax1.plot(angles, values, linewidth=2, label=stakeholder)
        ax1.fill(angles, values, alpha=0.1)
    
    # Set labels
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics)
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    ax1.set_title('Stakeholder Engagement Profile')
    
    # Add legend
    ax1.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # 2. Heatmap of engagement metrics
    ax2 = fig.add_subplot(2, 2, 2)
    
    # Create engagement matrix
    engagement_matrix = np.array([engagement_data[stakeholder] for stakeholder in stakeholders])
    
    # Create heatmap
    sns.heatmap(engagement_matrix, annot=True, fmt='.2f', cmap='YlGnBu',
               xticklabels=metrics, yticklabels=stakeholders, ax=ax2)
    
    ax2.set_title('Stakeholder Engagement Heatmap')
    
    # 3. Engagement vs. Conservation Impact scatter plot
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Calculate average engagement for each stakeholder
    avg_engagement = {s: np.mean(engagement_data[s]) for s in stakeholders}
    
    # Generate simulated conservation impact data
    # In a real application, this would be based on actual impact data
    conservation_impact = {}
    for stakeholder in stakeholders:
        # Impact is correlated with engagement but with some variation
        base_impact = avg_engagement[stakeholder] * 0.8
        noise = np.random.normal(0, 0.1)
        impact = np.clip(base_impact + noise, 0, 1)
        conservation_impact[stakeholder] = impact
    
    # Create scatter plot
    ax3.scatter(
        [avg_engagement[s] for s in stakeholders],
        [conservation_impact[s] for s in stakeholders],
        s=100,
        alpha=0.7
    )
    
    # Add labels for each point
    for i, stakeholder in enumerate(stakeholders):
        ax3.annotate(
            stakeholder,
            (avg_engagement[stakeholder], conservation_impact[stakeholder]),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    # Add regression line
    x = [avg_engagement[s] for s in stakeholders]
    y = [conservation_impact[s] for s in stakeholders]
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax3.plot(sorted(x), p(sorted(x)), 'r--', alpha=0.7)
    
    # Calculate correlation
    corr = np.corrcoef(x, y)[0, 1]
    ax3.text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=ax3.transAxes,
            bbox=dict(facecolor='white', alpha=0.8))
    
    ax3.set_xlabel('Average Engagement Level')
    ax3.set_ylabel('Conservation Impact')
    ax3.set_title('Engagement vs. Conservation Impact')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # 4. Engagement strategy recommendations
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Calculate engagement gaps (difference from ideal engagement of 0.8)
    ideal_engagement = 0.8
    engagement_gaps = {}
    
    for stakeholder in stakeholders:
        gaps = ideal_engagement - engagement_data[stakeholder]
        engagement_gaps[stakeholder] = {
            metric: gap for metric, gap in zip(metrics, gaps) if gap > 0
        }
    
    # Create a table of top recommendations
    recommendations = []
    
    for stakeholder in stakeholders:
        if engagement_gaps[stakeholder]:
            # Get the metric with the largest gap
            top_metric = max(engagement_gaps[stakeholder].items(), key=lambda x: x[1])[0]
            gap = engagement_gaps[stakeholder][top_metric]
            
            if gap > 0.1:  # Only include significant gaps
                recommendations.append({
                    'Stakeholder': stakeholder,
                    'Metric': top_metric,
                    'Gap': gap,
                    'Priority': 'High' if gap > 0.3 else 'Medium' if gap > 0.2 else 'Low'
                })
    
    # Sort by gap size (descending)
    recommendations = sorted(recommendations, key=lambda x: x['Gap'], reverse=True)
    
    # Create a table
    if recommendations:
        table_data = [
            [r['Stakeholder'], r['Metric'], f"{r['Gap']:.2f}", r['Priority']]
            for r in recommendations[:8]  # Show top 8 recommendations
        ]
        
        ax4.axis('tight')
        ax4.axis('off')
        table = ax4.table(
            cellText=table_data,
            colLabels=['Stakeholder', 'Engagement Metric', 'Gap', 'Priority'],
            loc='center',
            cellLoc='center'
        )
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Color code priorities
        for i, row in enumerate(recommendations[:8]):
            if row['Priority'] == 'High':
                table[(i+1, 3)].set_facecolor('#ffcccc')
            elif row['Priority'] == 'Medium':
                table[(i+1, 3)].set_facecolor('#ffffcc')
            else:
                table[(i+1, 3)].set_facecolor('#ccffcc')
        
        ax4.set_title('Engagement Improvement Recommendations')
    else:
        ax4.text(0.5, 0.5, 'No significant engagement gaps identified', 
                ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, 'stakeholder_engagement.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    return output_path
