# src/data_processing.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import json
# Try to import geopandas, but make it optional
try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("Warning: geopandas not installed. Geospatial visualizations will be disabled.")
# Import shapely only if geopandas is available
if GEOPANDAS_AVAILABLE:
    from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def generate_synthetic_data(n_samples=100):
    """
    Generate synthetic data for endangered species analysis.
    
    In a real-world scenario, this would be replaced with data from:
    - Field surveys
    - Remote sensing data
    - Government databases (IUCN Red List, GBIF)
    - Satellite tracking data
    - Environmental monitoring stations
    """
    print("Generating synthetic endangered species dataset...")
    
    # Define realistic categories
    habitat_types = ['Tropical Forest', 'Temperate Forest', 'Grassland', 'Wetland', 
                     'Marine', 'Coral Reef', 'Desert', 'Mountain', 'Tundra']
    regions = ['Africa', 'Asia', 'Europe', 'North America', 'South America', 
               'Oceania', 'Antarctica']
    threats = ['Habitat Loss', 'Climate Change', 'Pollution', 'Invasive Species', 
               'Disease', 'Overexploitation', 'Poaching', 'Infrastructure Development']
    species_groups = ['Mammal', 'Bird', 'Reptile', 'Amphibian', 'Fish', 'Invertebrate', 'Plant']
    
    # Generate random data
    np.random.seed(42)  # For reproducibility
    
    # Create realistic date range for monitoring (last 5 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    species_data = []
    for i in range(n_samples):
        # Basic species information
        species_group = np.random.choice(species_groups)
        habitat_type = np.random.choice(habitat_types)
        region = np.random.choice(regions)
        
        # Environmental factors
        habitat_loss = np.random.beta(2, 5)  # Beta distribution for more realistic values
        temperature_change = np.random.normal(1.2, 0.8)  # Normal distribution centered around 1.2°C
        pollution_level = np.random.beta(2, 4)
        
        # Population metrics
        if species_group == 'Mammal':
            population_size = int(np.random.lognormal(7, 1.5))  # Lognormal for realistic population sizes
        elif species_group == 'Bird':
            population_size = int(np.random.lognormal(8, 1.8))
        elif species_group == 'Invertebrate':
            population_size = int(np.random.lognormal(10, 2))
        else:
            population_size = int(np.random.lognormal(7.5, 1.7))
            
        population_trend = np.random.choice(['Decreasing', 'Stable', 'Increasing'], 
                                           p=[0.6, 0.3, 0.1])  # Most endangered species are decreasing
        
        reproduction_rate = np.random.beta(2, 5) * 0.5  # Beta distribution for more realistic values
        genetic_diversity = np.random.beta(2, 3)  # Higher is better
        
        # Geographic coordinates (for mapping)
        # Generate realistic coordinates based on region
        if region == 'North America':
            latitude = np.random.uniform(25, 50)
            longitude = np.random.uniform(-125, -70)
        elif region == 'Europe':
            latitude = np.random.uniform(35, 60)
            longitude = np.random.uniform(-10, 30)
        elif region == 'Asia':
            latitude = np.random.uniform(10, 45)
            longitude = np.random.uniform(70, 140)
        elif region == 'Africa':
            latitude = np.random.uniform(-35, 35)
            longitude = np.random.uniform(-20, 50)
        elif region == 'South America':
            latitude = np.random.uniform(-40, 10)
            longitude = np.random.uniform(-80, -35)
        elif region == 'Oceania':
            latitude = np.random.uniform(-40, -10)
            longitude = np.random.uniform(115, 180)
        else:  # Antarctica
            latitude = np.random.uniform(-75, -60)
            longitude = np.random.uniform(-180, 180)
        
        # Conservation efforts
        protected_area = np.random.choice([True, False], p=[0.4, 0.6])
        conservation_funding = np.random.lognormal(10, 1) if protected_area else np.random.lognormal(8, 1.5)
        
        # Calculate vulnerability score (target variable)
        # Higher score means more endangered
        vulnerability_score = (
            0.3 * habitat_loss + 
            0.2 * abs(temperature_change) + 
            0.15 * pollution_level - 
            0.1 * (np.log(population_size) / 10) - 
            0.1 * reproduction_rate -
            0.05 * genetic_diversity +
            0.1 * (0 if protected_area else 0.5)
        )
        
        # Adjust vulnerability based on population trend
        if population_trend == 'Decreasing':
            vulnerability_score += 0.1
        elif population_trend == 'Increasing':
            vulnerability_score -= 0.1
            
        # Ensure vulnerability is between 0 and 1
        vulnerability_score = max(0, min(1, vulnerability_score))
        
        # Assign conservation status based on vulnerability score
        if vulnerability_score > 0.7:
            conservation_status = 'Critically Endangered'
        elif vulnerability_score > 0.5:
            conservation_status = 'Endangered'
        elif vulnerability_score > 0.3:
            conservation_status = 'Vulnerable'
        elif vulnerability_score > 0.1:
            conservation_status = 'Near Threatened'
        else:
            conservation_status = 'Least Concern'
        
        # Generate random observation date
        days_from_start = np.random.randint(0, (end_date - start_date).days)
        observation_date = start_date + timedelta(days=days_from_start)
        
        # Primary threat
        primary_threat = np.random.choice(threats)
        
        species_data.append({
            'id': f'SP{i+1:04d}',
            'name': f'Species {i+1}',
            'species_group': species_group,
            'habitat': habitat_type,
            'region': region,
            'latitude': latitude,
            'longitude': longitude,
            'primary_threat': primary_threat,
            'habitat_loss': habitat_loss,
            'temperature_change': temperature_change,
            'pollution_level': pollution_level,
            'population_size': population_size,
            'population_trend': population_trend,
            'reproduction_rate': reproduction_rate,
            'genetic_diversity': genetic_diversity,
            'protected_area': protected_area,
            'conservation_funding': conservation_funding,
            'observation_date': observation_date.strftime('%Y-%m-%d'),
            'vulnerability_score': vulnerability_score,
            'conservation_status': conservation_status
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(species_data)
    print(f"Generated dataset with {len(df)} species records")
    
    # Create some exploratory visualizations
    create_exploratory_visualizations(df)
    
    return df

def create_exploratory_visualizations(df):
    """Create exploratory visualizations of the dataset."""
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # 1. Conservation status distribution
    plt.figure(figsize=(12, 6))
    status_counts = df['conservation_status'].value_counts().sort_index()
    sns.barplot(x=status_counts.index, y=status_counts.values)
    plt.title('Distribution of Conservation Status')
    plt.xlabel('Conservation Status')
    plt.ylabel('Number of Species')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'conservation_status_distribution.png'))
    plt.close()
    
    # 2. Species groups by conservation status
    plt.figure(figsize=(14, 8))
    species_status = pd.crosstab(df['species_group'], df['conservation_status'])
    species_status.plot(kind='bar', stacked=True)
    plt.title('Conservation Status by Species Group')
    plt.xlabel('Species Group')
    plt.ylabel('Number of Species')
    plt.legend(title='Conservation Status')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'species_group_by_status.png'))
    plt.close()
    
    # 3. Primary threats distribution
    plt.figure(figsize=(14, 6))
    threat_counts = df['primary_threat'].value_counts()
    sns.barplot(x=threat_counts.index, y=threat_counts.values)
    plt.title('Distribution of Primary Threats')
    plt.xlabel('Primary Threat')
    plt.ylabel('Number of Species')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'primary_threats_distribution.png'))
    plt.close()
    
    # 4. Correlation heatmap of numerical features
    plt.figure(figsize=(12, 10))
    numerical_cols = ['habitat_loss', 'temperature_change', 'pollution_level', 
                      'population_size', 'reproduction_rate', 'genetic_diversity',
                      'conservation_funding', 'vulnerability_score']
    corr = df[numerical_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # 5. Geographic distribution (simple scatter plot)
    plt.figure(figsize=(14, 8))
    plt.scatter(df['longitude'], df['latitude'], c=df['vulnerability_score'], 
                cmap='YlOrRd', alpha=0.7)
    plt.colorbar(label='Vulnerability Score')
    plt.title('Geographic Distribution of Species')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'geographic_distribution.png'))
    plt.close()

def create_geospatial_visualization(df):
    """Create a geospatial visualization of species distribution."""
    if not GEOPANDAS_AVAILABLE:
        print("Skipping geospatial visualization: geopandas not installed")
        return
        
    try:
        # Create a GeoDataFrame
        from shapely.geometry import Point
        geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry)
        
        # Get world map
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(15, 10))
        world.plot(ax=ax, color='lightgray')
        
        # Plot points with color based on conservation status
        status_colors = {
            'Critically Endangered': 'darkred',
            'Endangered': 'red',
            'Vulnerable': 'orange',
            'Near Threatened': 'yellow',
            'Least Concern': 'green'
        }
        
        for status, color in status_colors.items():
            subset = gdf[gdf['conservation_status'] == status]
            subset.plot(ax=ax, color=color, markersize=30, label=status, alpha=0.7)
        
        plt.title('Global Distribution of Species by Conservation Status')
        plt.legend()
        
        # Save the plot
        plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'plots')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        plt.savefig(os.path.join(plots_dir, 'species_geospatial_distribution.png'))
        plt.close()
        
        print("Geospatial visualization created successfully")
    except Exception as e:
        print(f"Could not create geospatial visualization: {e}")
        print("Continuing without geospatial visualization...")

def preprocess_data(df, task_type="classification"):
    """
    Preprocess data for machine learning.
    
    Args:
        df: DataFrame with species data
        task_type: Type of ML task - "classification" or "regression"
        
    Returns:
        dict: Dictionary with preprocessed data
    """
    print(f"Preprocessing data for {task_type}...")
    
    # Handle missing values
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Define features
    numerical_features = [
        'habitat_loss', 'temperature_change', 'pollution_level', 
        'population_size', 'reproduction_rate', 'genetic_diversity',
        'conservation_funding'
    ]
    
    categorical_features = [
        'species_group', 'habitat', 'region', 'population_trend', 
        'primary_threat', 'protected_area'
    ]
    
    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Define target variable based on task type
    if task_type == "classification":
        # Convert conservation status to numerical values
        status_map = {
            'Critically Endangered': 4,
            'Endangered': 3,
            'Vulnerable': 2,
            'Near Threatened': 1,
            'Least Concern': 0
        }
        
        target = df['conservation_status'].map(status_map)
    else:  # regression
        # Use vulnerability score as target
        target = df['vulnerability_score']
    
    # Split data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        df[numerical_features + categorical_features], 
        target, 
        test_size=0.2, 
        random_state=42,
        stratify=target if task_type == "classification" else None
    )
    
    # Fit and transform the data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing
    feature_names = []
    
    # Add numerical feature names
    feature_names.extend(numerical_features)
    
    # Add one-hot encoded feature names
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    categorical_feature_names = ohe.get_feature_names_out(categorical_features)
    feature_names.extend(categorical_feature_names)
    
    result = {
        'X_train': X_train_processed,
        'X_test': X_test_processed,
        'y_train': y_train.values,
        'y_test': y_test.values,
        'feature_names': feature_names,
        'preprocessor': preprocessor
    }
    
    # Add status_map only for classification
    if task_type == "classification":
        result['status_map'] = status_map
    
    return result

def save_data(df, filename):
    """Save data to CSV file."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    file_path = os.path.join(data_dir, filename)
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")
    return file_path

def load_data(filename):
    """Load data from CSV file."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    file_path = os.path.join(data_dir, filename)
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(f"Loaded data from {file_path}")
        return df
    else:
        print(f"File {file_path} not found. Generating synthetic data...")
        df = generate_synthetic_data(n_samples=200)  # Generate more samples for better analysis
        save_data(df, filename)
        return df

def generate_conservation_report(df, model_results=None):
    """
    Generate a comprehensive conservation report based on the data and model results.
    This would be used by conservation professionals to make informed decisions.
    
    Args:
        df: DataFrame with species data
        model_results: Optional dictionary with model evaluation results
        
    Returns:
        None (saves report to file)
    """
    print("Generating conservation report...")
    
    # Create reports directory
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'reports')
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
    
    # Generate report filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(reports_dir, f"conservation_report_{timestamp}.txt")
    
    with open(report_file, 'w') as f:
        # Report header
        f.write("=" * 80 + "\n")
        f.write("ENDANGERED SPECIES CONSERVATION ASSESSMENT REPORT\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # 1. Executive Summary
        f.write("1. EXECUTIVE SUMMARY\n")
        f.write("-" * 80 + "\n")
        
        # Count species by conservation status
        status_counts = df['conservation_status'].value_counts()
        total_species = len(df)
        
        f.write(f"Total species assessed: {total_species}\n\n")
        f.write("Conservation Status Distribution:\n")
        
        for status, count in status_counts.items():
            percentage = (count / total_species) * 100
            f.write(f"- {status}: {count} species ({percentage:.1f}%)\n")
        
        # Calculate species at risk (CR, EN, VU)
        at_risk = df[df['conservation_status'].isin(['Critically Endangered', 'Endangered', 'Vulnerable'])].shape[0]
        at_risk_pct = (at_risk / total_species) * 100
        f.write(f"\nSpecies at risk (CR, EN, VU): {at_risk} ({at_risk_pct:.1f}%)\n\n")
        
        # 2. Threat Analysis
        f.write("\n2. THREAT ANALYSIS\n")
        f.write("-" * 80 + "\n")
        
        # Primary threats
        threat_counts = df['primary_threat'].value_counts()
        f.write("Primary Threats:\n")
        for threat, count in threat_counts.items():
            percentage = (count / total_species) * 100
            f.write(f"- {threat}: {count} species ({percentage:.1f}%)\n")
        
        # Analyze threats by conservation status
        f.write("\nThreats by Conservation Status (top threat for each status):\n")
        for status in df['conservation_status'].unique():
            subset = df[df['conservation_status'] == status]
            top_threat = subset['primary_threat'].value_counts().index[0]
            count = subset['primary_threat'].value_counts().iloc[0]
            percentage = (count / len(subset)) * 100
            f.write(f"- {status}: {top_threat} ({percentage:.1f}% of {status} species)\n")
        
        # 3. Geographic Analysis
        f.write("\n3. GEOGRAPHIC ANALYSIS\n")
        f.write("-" * 80 + "\n")
        
        # Species by region
        region_counts = df['region'].value_counts()
        f.write("Species Distribution by Region:\n")
        for region, count in region_counts.items():
            percentage = (count / total_species) * 100
            f.write(f"- {region}: {count} species ({percentage:.1f}%)\n")
        
        # At-risk species by region
        f.write("\nAt-Risk Species by Region:\n")
        at_risk_df = df[df['conservation_status'].isin(['Critically Endangered', 'Endangered', 'Vulnerable'])]
        region_at_risk = at_risk_df['region'].value_counts()
        
        for region, count in region_at_risk.items():
            total_in_region = region_counts[region]
            percentage = (count / total_in_region) * 100
            f.write(f"- {region}: {count} at-risk species ({percentage:.1f}% of species in this region)\n")
        
        # 4. Habitat Analysis
        f.write("\n4. HABITAT ANALYSIS\n")
        f.write("-" * 80 + "\n")
        
        # Species by habitat
        habitat_counts = df['habitat'].value_counts()
        f.write("Species Distribution by Habitat:\n")
        for habitat, count in habitat_counts.items():
            percentage = (count / total_species) * 100
            f.write(f"- {habitat}: {count} species ({percentage:.1f}%)\n")
        
        # Most threatened habitats
        f.write("\nMost Threatened Habitats (by % of at-risk species):\n")
        habitat_threat = {}
        
        for habitat in df['habitat'].unique():
            habitat_df = df[df['habitat'] == habitat]
            at_risk_in_habitat = habitat_df[habitat_df['conservation_status'].isin(
                ['Critically Endangered', 'Endangered', 'Vulnerable'])].shape[0]
            percentage = (at_risk_in_habitat / len(habitat_df)) * 100
            habitat_threat[habitat] = percentage
        
        for habitat, percentage in sorted(habitat_threat.items(), key=lambda x: x[1], reverse=True):
            count = habitat_counts[habitat]
            f.write(f"- {habitat}: {percentage:.1f}% at risk ({count} total species)\n")
        
        # 5. Conservation Recommendations
        f.write("\n5. CONSERVATION RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n")
        
        # Identify priority species (CR with low protection)
        priority_species = df[(df['conservation_status'] == 'Critically Endangered') & 
                             (df['protected_area'] == False)]
        
        f.write(f"Priority Species for Protection: {len(priority_species)} identified\n")
        f.write("These critically endangered species are not in protected areas and require immediate action.\n\n")
        
        # Top 5 priority species
        if len(priority_species) > 0:
            f.write("Top Priority Species:\n")
            for i, (_, species) in enumerate(priority_species.sort_values('vulnerability_score', ascending=False).head(5).iterrows()):
                f.write(f"{i+1}. {species['name']} (ID: {species['id']})\n")
                f.write(f"   - Species Group: {species['species_group']}\n")
                f.write(f"   - Region: {species['region']}\n")
                f.write(f"   - Habitat: {species['habitat']}\n")
                f.write(f"   - Primary Threat: {species['primary_threat']}\n")
                f.write(f"   - Vulnerability Score: {species['vulnerability_score']:.4f}\n\n")
        
        # Priority habitats
        f.write("Priority Habitats for Conservation:\n")
        for habitat, percentage in sorted(habitat_threat.items(), key=lambda x: x[1], reverse=True)[:3]:
            f.write(f"- {habitat}: {percentage:.1f}% of species at risk\n")
            
            # Get primary threats for this habitat
            habitat_df = df[df['habitat'] == habitat]
            habitat_threats = habitat_df['primary_threat'].value_counts().head(2)
            f.write(f"  Primary threats: {', '.join(habitat_threats.index)}\n")
        
        # 6. Model Insights (if available)
        if model_results:
            f.write("\n6. PREDICTIVE MODEL INSIGHTS\n")
            f.write("-" * 80 + "\n")
            
            # Classification model results
            if 'classification' in model_results:
                clf_results = model_results['classification']
                f.write(f"Classification Model: {clf_results['best_model']}\n")
                f.write(f"Accuracy: {clf_results['accuracy']:.4f}\n")
                
                f.write("\nKey Factors Determining Conservation Status:\n")
                for feature in clf_results['top_features']:
                    f.write(f"- {feature}\n")
            
            # Regression model results
            if 'regression' in model_results:
                reg_results = model_results['regression']
                f.write(f"\nVulnerability Prediction Model: {reg_results['best_model']}\n")
                f.write(f"R² Score: {reg_results['r2_score']:.4f}\n")
                
                f.write("\nKey Factors Determining Vulnerability Score:\n")
                for feature in reg_results['top_features']:
                    f.write(f"- {feature}\n")
        
        # 7. Conclusion
        f.write("\n7. CONCLUSION\n")
        f.write("-" * 80 + "\n")
        f.write("This report highlights the current conservation status of assessed species and identifies\n")
        f.write("priority areas for conservation action. The data suggests that immediate attention should\n")
        f.write(f"be focused on {priority_species.shape[0]} critically endangered species not currently in protected areas,\n")
        f.write(f"particularly in {', '.join([h for h, _ in sorted(habitat_threat.items(), key=lambda x: x[1], reverse=True)[:2]])} habitats.\n\n")
        
        f.write("The primary threats identified across all species are:\n")
        for threat, count in threat_counts.head(3).items():
            percentage = (count / total_species) * 100
            f.write(f"- {threat} ({percentage:.1f}% of species affected)\n")
        
        f.write("\nRecommended next steps include:\n")
        f.write("1. Establish protected areas for identified priority species\n")
        f.write("2. Develop targeted conservation plans for most threatened habitats\n")
        f.write("3. Address primary threats through policy and direct intervention\n")
        f.write("4. Continue monitoring population trends and update conservation status regularly\n")
        f.write("5. Allocate conservation funding based on vulnerability predictions\n")
    
    print(f"Conservation report generated and saved to {report_file}")
    return report_file
