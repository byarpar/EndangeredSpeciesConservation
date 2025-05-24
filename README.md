# Endangered Species Conservation Project

## Overview

This project provides a comprehensive machine learning and data analysis framework for conservation biologists, environmental consultants, and wildlife management professionals. It helps identify priority species for conservation, predict conservation status, estimate resource requirements, and generate detailed reports and visualizations for decision-making.

## Features

- **Machine Learning Models**: Predict conservation status and vulnerability scores using multiple algorithms
- **Conservation Priority Analysis**: Calculate priority scores based on multiple ecological factors
- **Resource Estimation**: Estimate conservation costs and staff requirements
- **Species Clustering**: Group species by ecological traits for targeted conservation approaches
- **Comprehensive Reporting**: Generate detailed HTML reports with visualizations and recommendations
- **Species Profiles**: Create individual profiles for high-priority species
- **Conservation Priority Maps**: Visualize species distribution across key ecological variables

## Installation

\`\`\`bash
# Clone the repository
git clone https://github.com/byarpar/EndangeredSpeciesConservation.git
cd endangered-species-conservation

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
\`\`\`

## Usage

### Running the Full Analysis

\`\`\`bash
python main.py
\`\`\`

This will:
1. Load or generate species data
2. Perform conservation analysis and calculate priority scores
3. Train and evaluate multiple machine learning models
4. Generate a comprehensive conservation report
5. Create profiles for high-priority species
6. Save all results and visualizations

### Using Individual Components

\`\`\`python
from src.data_processing import load_data
from src.conservation_analysis import calculate_conservation_priority_score, identify_priority_species

# Load data
data = load_data('your_data.csv')

# Calculate conservation priority scores
data_with_priority = calculate_conservation_priority_score(data)

# Identify priority species
priority_species = identify_priority_species(data_with_priority)
print(f"Found {len(priority_species)} priority species")
\`\`\`



## Outputs

The project generates several outputs:

1. **Conservation Report**: A comprehensive HTML report with analysis, visualizations, and recommendations
2. **Species Profiles**: Individual HTML profiles for high-priority species
3. **Priority Map**: Visualization of species distribution across key ecological variables
4. **Model Visualizations**: Performance metrics, feature importance, and comparison charts
5. **Analyzed Data**: CSV file with added conservation metrics and priority scores

## For Conservation Professionals

This tool is designed to support evidence-based decision-making in conservation:

- **Conservation Planners**: Identify priority species and estimate resource requirements
- **Field Biologists**: Generate species profiles with specific conservation recommendations
- **Environmental Consultants**: Create professional reports for clients and stakeholders
- **Wildlife Managers**: Optimize resource allocation across multiple species
- **Conservation NGOs**: Support funding proposals with data-driven analysis

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## License

This project is licensed under the MIT License - see the LICENSE file for details.
