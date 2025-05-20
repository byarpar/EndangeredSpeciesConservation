import os
import glob
import re
from datetime import datetime
import pandas as pd
import json

def extract_timestamp(filename):
    """Extract timestamp from conservation report filename."""
    match = re.search(r'conservation_report_(\d{8}_\d{6})\.html', filename)
    if match:
        timestamp_str = match.group(1)
        try:
            return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        except ValueError:
            return datetime.min
    return datetime.min

def extract_species_name(filename):
    """Extract species name from profile filename."""
    # Remove _profile.html and replace underscores with spaces
    name = os.path.basename(filename).replace('_profile.html', '').replace('_', ' ').title()
    return name

def get_species_data():
    """Get species data from the analyzed CSV if available."""
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(root_dir, 'data', 'species_data_analyzed.csv')
    
    if os.path.exists(data_file):
        try:
            df = pd.read_csv(data_file)
            
            # Count species by conservation status
            status_counts = df['conservation_status'].value_counts().to_dict() if 'conservation_status' in df.columns else {}
            
            # Get high priority species
            high_priority = len(df[df['conservation_priority_score'] >= 70]) if 'conservation_priority_score' in df.columns else 0
            
            # Get average conservation cost
            avg_cost = df['estimated_conservation_cost'].mean() if 'estimated_conservation_cost' in df.columns else 0
            
            # Get regions with most endangered species
            if 'region' in df.columns and 'conservation_status' in df.columns:
                endangered_mask = df['conservation_status'].isin(['Critically Endangered', 'Endangered'])
                regions = df[endangered_mask]['region'].value_counts().head(3).to_dict()
            else:
                regions = {}
            
            # Get primary threats
            if 'primary_threat' in df.columns:
                threats = df['primary_threat'].value_counts().head(5).to_dict()
            else:
                threats = {}
            
            return {
                'status_counts': status_counts,
                'high_priority': high_priority,
                'avg_cost': avg_cost,
                'regions': regions,
                'threats': threats,
                'total_species': len(df)
            }
        except Exception as e:
            print(f"Error reading species data: {e}")
    
    return {
        'status_counts': {},
        'high_priority': 0,
        'avg_cost': 0,
        'regions': {},
        'threats': {},
        'total_species': 0
    }

def create_dashboard():
    """Create an index.html dashboard for all reports and profiles."""
    # Project root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Directories for reports and profiles
    reports_dir = os.path.join(root_dir, 'reports')
    profiles_dir = os.path.join(root_dir, 'profiles')
    plots_dir = os.path.join(root_dir, 'data', 'plots')
    
    # Create directories if they don't exist
    for directory in [reports_dir, profiles_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Find all report files
    report_files = glob.glob(os.path.join(reports_dir, '*.html'))
    report_files.sort(key=extract_timestamp, reverse=True)
    
    # Find all profile files
    profile_files = glob.glob(os.path.join(profiles_dir, '*.html'))
    profile_files.sort()
    
    # Get species data
    species_data = get_species_data()
    
    # Find key visualization files
    priority_map = os.path.join(plots_dir, 'conservation_priority_map.png')
    status_breakdown = os.path.join(plots_dir, 'status_breakdown.png')
    priority_distribution = os.path.join(plots_dir, 'priority_distribution.png')
    threat_analysis = os.path.join(plots_dir, 'threat_analysis.png')
    feature_importance = glob.glob(os.path.join(plots_dir, 'feature_importance_*.png'))
    
    # Create the index.html file
    with open(os.path.join(root_dir, 'index.html'), 'w') as f:
        f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enviro Wise Conservation | Client Portal</title>
    <style>
        :root {
            --primary-color: #1e6f5c;
            --secondary-color: #289672;
            --accent-color: #29bb89;
            --light-color: #e6f4f1;
            --dark-color: #333;
            --danger-color: #d7191c;
            --warning-color: #fdae61;
            --success-color: #1a936f;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--dark-color);
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
        }
        
        .container {
            width: 90%;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background-color: var(--primary-color);
            color: white;
            padding: 1rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }
        
        .logo {
            font-size: 1.8rem;
            font-weight: bold;
            display: flex;
            align-items: center;
        }
        
        .logo-icon {
            margin-right: 10px;
            font-size: 2rem;
        }
        
        nav ul {
            display: flex;
            list-style: none;
            margin: 0;
            padding: 0;
        }
        
        nav ul li {
            margin-left: 20px;
        }
        
        nav ul li a {
            color: white;
            text-decoration: none;
            font-weight: 500;
            padding: 5px 10px;
            border-radius: 3px;
            transition: background-color 0.3s;
        }
        
        nav ul li a:hover {
            background-color: rgba(255,255,255,0.1);
        }
        
        .hero {
            background-color: var(--light-color);
            padding: 40px 0;
            text-align: center;
            margin-bottom: 30px;
        }
        
        .hero h1 {
            color: var(--primary-color);
            font-size: 2.5rem;
            margin-bottom: 15px;
        }
        
        .hero p {
            font-size: 1.2rem;
            max-width: 800px;
            margin: 0 auto;
            color: var(--dark-color);
        }
        
        h1, h2, h3 {
            color: var(--primary-color);
        }
        
        .dashboard-section {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            padding: 25px;
            margin-bottom: 30px;
        }
        
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }
        
        .section-header h2 {
            margin: 0;
        }
        
        .section-header .actions {
            display: flex;
        }
        
        .btn {
            display: inline-block;
            padding: 8px 15px;
            background-color: var(--primary-color);
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-weight: 500;
            margin-left: 10px;
            transition: background-color 0.3s;
        }
        
        .btn:hover {
            background-color: var(--secondary-color);
        }
        
        .btn-outline {
            background-color: transparent;
            border: 1px solid var(--primary-color);
            color: var(--primary-color);
        }
        
        .btn-outline:hover {
            background-color: var(--light-color);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            padding: 20px;
            text-align: center;
            transition: transform 0.3s;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        .stat-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: var(--primary-color);
            margin: 10px 0;
        }
        
        .stat-label {
            color: var(--dark-color);
            font-size: 1rem;
            font-weight: 500;
        }
        
        .stat-card.critical {
            border-top: 4px solid var(--danger-color);
        }
        
        .stat-card.endangered {
            border-top: 4px solid var(--warning-color);
        }
        
        .stat-card.priority {
            border-top: 4px solid var(--accent-color);
        }
        
        .stat-card.funding {
            border-top: 4px solid var(--success-color);
        }
        
        .file-list {
            list-style: none;
            padding: 0;
        }
        
        .file-item {
            padding: 15px;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background-color 0.3s;
        }
        
        .file-item:last-child {
            border-bottom: none;
        }
        
        .file-item:hover {
            background-color: var(--light-color);
        }
        
        .file-link {
            color: var(--primary-color);
            text-decoration: none;
            font-weight: 500;
            flex-grow: 1;
        }
        
        .file-link:hover {
            text-decoration: underline;
        }
        
        .file-meta {
            display: flex;
            align-items: center;
        }
        
        .file-date {
            color: #777;
            font-size: 0.9em;
            margin-right: 15px;
        }
        
        .tag {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.8em;
            font-weight: 500;
            color: white;
        }
        
        .tag-critical {
            background-color: var(--danger-color);
        }
        
        .tag-endangered {
            background-color: var(--warning-color);
        }
        
        .tag-vulnerable {
            background-color: var(--accent-color);
        }
        
        .visualizations {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .visualization-card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .visualization-card img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            margin-top: 15px;
        }
        
        .visualization-card h3 {
            margin-top: 0;
            color: var(--primary-color);
        }
        
        .two-columns {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
        }
        
        .recommendations {
            background-color: var(--light-color);
            border-left: 4px solid var(--primary-color);
            padding: 15px;
            margin-bottom: 20px;
        }
        
        .recommendations h3 {
            margin-top: 0;
            color: var(--primary-color);
        }
        
        .recommendations ul {
            padding-left: 20px;
        }
        
        .recommendations li {
            margin-bottom: 10px;
        }
        
        .resources-list {
            list-style: none;
            padding: 0;
        }
        
        .resources-list li {
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        
        .resources-list li:last-child {
            border-bottom: none;
        }
        
        .resources-list a {
            color: var(--primary-color);
            text-decoration: none;
            font-weight: 500;
        }
        
        .resources-list a:hover {
            text-decoration: underline;
        }
        
        .search-box {
            width: 100%;
            padding: 10px 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-bottom: 20px;
            font-size: 1rem;
        }
        
        footer {
            background-color: var(--primary-color);
            color: white;
            padding: 30px 0;
            margin-top: 50px;
        }
        
        .footer-content {
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }
        
        .footer-column {
            flex: 1;
            min-width: 200px;
            margin-bottom: 20px;
        }
        
        .footer-column h3 {
            color: white;
            margin-top: 0;
            font-size: 1.2rem;
        }
        
        .footer-column ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        
        .footer-column ul li {
            margin-bottom: 10px;
        }
        
        .footer-column a {
            color: var(--light-color);
            text-decoration: none;
        }
        
        .footer-column a:hover {
            text-decoration: underline;
        }
        
        .copyright {
            text-align: center;
            padding-top: 20px;
            margin-top: 20px;
            border-top: 1px solid rgba(255,255,255,0.1);
            font-size: 0.9rem;
        }
        
        @media (max-width: 768px) {
            .header-content {
                flex-direction: column;
                text-align: center;
            }
            
            nav ul {
                margin-top: 15px;
                justify-content: center;
            }
            
            nav ul li {
                margin: 0 10px;
            }
            
            .visualizations {
                grid-template-columns: 1fr;
            }
            
            .two-columns {
                grid-template-columns: 1fr;
            }
            
            .file-item {
                flex-direction: column;
                align-items: flex-start;
            }
            
            .file-meta {
                margin-top: 10px;
            }
        }
    </style>
</head>
<body>
    <header>
        <div class="header-content">
            <div class="logo">
                <span class="logo-icon">🌿</span> Enviro Wise Consultancy
            </div>
            <nav>
                <ul>
                    <li><a href="#dashboard">Dashboard</a></li>
                    <li><a href="#reports">Reports</a></li>
                    <li><a href="#profiles">Species Profiles</a></li>
                    <li><a href="#visualizations">Visualizations</a></li>
                    <li><a href="#contact">Contact</a></li>
                </ul>
            </nav>
        </div>
    </header>
    
    <section class="hero">
        <div class="container">
            <h1>Endangered Species Conservation Portal</h1>
            <p>A comprehensive analysis platform for conservation planning, species assessment, and resource allocation</p>
        </div>
    </section>
    
    <div class="container" id="dashboard">
        <div class="dashboard-section">
            <div class="section-header">
                <h2>Executive Summary</h2>
                <div class="actions">
                    <a href="#" class="btn btn-outline">Export Data</a>
                    <a href="#" class="btn">Share</a>
                </div>
            </div>
            
            <div class="stats-grid">''')
        
        # Add statistics based on species data
        critically_endangered = species_data['status_counts'].get('Critically Endangered', 0)
        endangered = species_data['status_counts'].get('Endangered', 0)
        
        f.write(f'''
                <div class="stat-card critical">
                    <div class="stat-label">Critically Endangered</div>
                    <div class="stat-value">{critically_endangered}</div>
                    <div>Species</div>
                </div>
                
                <div class="stat-card endangered">
                    <div class="stat-label">Endangered</div>
                    <div class="stat-value">{endangered}</div>
                    <div>Species</div>
                </div>
                
                <div class="stat-card priority">
                    <div class="stat-label">High Priority</div>
                    <div class="stat-value">{species_data['high_priority']}</div>
                    <div>Species</div>
                </div>
                
                <div class="stat-card funding">
                    <div class="stat-label">Avg. Conservation Cost</div>
                    <div class="stat-value">${int(species_data['avg_cost']):,}</div>
                    <div>Per Species</div>
                </div>
            </div>
            
            <div class="recommendations">
                <h3>Key Recommendations</h3>
                <ul>''')
        
        # Add recommendations based on data
        if critically_endangered > 0:
            f.write(f'''
                    <li><strong>Immediate Action Required:</strong> {critically_endangered} critically endangered species need urgent conservation interventions.</li>''')
        
        if species_data['high_priority'] > 0:
            f.write(f'''
                    <li><strong>Resource Allocation:</strong> Focus on {species_data['high_priority']} high-priority species with conservation scores above 70.</li>''')
        
        # Add region-specific recommendations if available
        if species_data['regions']:
            top_region = max(species_data['regions'].items(), key=lambda x: x[1])
            f.write(f'''
                    <li><strong>Geographic Focus:</strong> Prioritize conservation efforts in {top_region[0]} with {top_region[1]} endangered species.</li>''')
        
        # Add threat-specific recommendations if available
        if species_data['threats']:
            top_threat = max(species_data['threats'].items(), key=lambda x: x[1])
            f.write(f'''
                    <li><strong>Threat Mitigation:</strong> Develop strategies to address {top_threat[0]}, affecting {top_threat[1]} species.</li>''')
        
        f.write('''
                    <li><strong>Monitoring:</strong> Implement regular monitoring programs to track conservation outcomes and population trends.</li>
                </ul>
            </div>
        </div>
        
        <div class="dashboard-section" id="visualizations">
            <div class="section-header">
                <h2>Key Visualizations</h2>
                <div class="actions">
                    <a href="data/plots/" class="btn btn-outline">View All</a>
                </div>
            </div>
            
            <div class="visualizations">''')
        
        # Add key visualizations if they exist
        if os.path.exists(priority_map):
            f.write('''
                <div class="visualization-card">
                    <h3>Conservation Priority Map</h3>
                    <p>Distribution of species by habitat loss and population size, colored by conservation priority.</p>
                    <img src="data/plots/conservation_priority_map.png" alt="Conservation Priority Map">
                </div>''')
        
        if os.path.exists(status_breakdown):
            f.write('''
                <div class="visualization-card">
                    <h3>Conservation Status Breakdown</h3>
                    <p>Distribution of species across IUCN Red List categories.</p>
                    <img src="data/plots/status_breakdown.png" alt="Conservation Status Breakdown">
                </div>''')
        
        if os.path.exists(threat_analysis):
            f.write('''
                <div class="visualization-card">
                    <h3>Threat Analysis</h3>
                    <p>Analysis of key threats across different conservation statuses.</p>
                    <img src="data/plots/threat_analysis.png" alt="Threat Analysis">
                </div>''')
        
        if feature_importance and len(feature_importance) > 0:
            f.write(f'''
                <div class="visualization-card">
                    <h3>Key Factors Influencing Conservation</h3>
                    <p>Feature importance analysis showing the most influential factors for conservation status.</p>
                    <img src="{os.path.relpath(feature_importance[0], root_dir)}" alt="Feature Importance">
                </div>''')
        
        f.write('''
            </div>
        </div>
        
        <div class="two-columns">
            <div class="dashboard-section" id="reports">
                <div class="section-header">
                    <h2>Conservation Reports</h2>
                    <div class="actions">
                        <a href="#" class="btn btn-outline">Filter</a>
                    </div>
                </div>
                
                <input type="text" class="search-box" placeholder="Search reports...">
                
                <ul class="file-list">''')
        
        # Add report links
        if report_files:
            for report_file in report_files:
                filename = os.path.basename(report_file)
                timestamp = extract_timestamp(filename)
                date_str = timestamp.strftime('%Y-%m-%d') if timestamp != datetime.min else "Unknown date"
                
                f.write(f'''
                    <li class="file-item">
                        <a href="reports/{filename}" class="file-link">Conservation Assessment Report</a>
                        <div class="file-meta">
                            <span class="file-date">{date_str}</span>
                            <span class="tag tag-critical">Priority</span>
                        </div>
                    </li>''')
        else:
            f.write('''
                    <li class="file-item">No conservation reports found.</li>''')
        
        f.write('''
                </ul>
            </div>
            
            <div class="dashboard-section" id="profiles">
                <div class="section-header">
                    <h2>Species Profiles</h2>
                    <div class="actions">
                        <a href="#" class="btn btn-outline">Filter</a>
                    </div>
                </div>
                
                <input type="text" class="search-box" placeholder="Search species...">
                
                <ul class="file-list">''')
        
        # Add profile links
        if profile_files:
            for profile_file in profile_files:
                filename = os.path.basename(profile_file)
                species_name = extract_species_name(filename)
                
                # Assign a random tag for demonstration (in a real scenario, this would be based on actual data)
                tags = ['tag-critical', 'tag-endangered', 'tag-vulnerable']
                tag_index = hash(filename) % len(tags)
                tag_class = tags[tag_index]
                tag_text = tag_class.replace('tag-', '').title()
                
                f.write(f'''
                    <li class="file-item">
                        <a href="profiles/{filename}" class="file-link">{species_name}</a>
                        <div class="file-meta">
                            <span class="tag {tag_class}">{tag_text}</span>
                        </div>
                    </li>''')
        else:
            f.write('''
                    <li class="file-item">No species profiles found.</li>''')
        
        f.write('''
                </ul>
            </div>
        </div>
        
        <div class="dashboard-section">
            <div class="section-header">
                <h2>Resources for Conservation Professionals</h2>
            </div>
            
            <div class="two-columns">
                <div>
                    <h3>Conservation Planning Tools</h3>
                    <ul class="resources-list">
                        <li><a href="#">Species Conservation Assessment Guide</a></li>
                        <li><a href="#">Habitat Restoration Planning Template</a></li>
                        <li><a href="#">Conservation Funding Calculator</a></li>
                        <li><a href="#">Population Viability Analysis Tool</a></li>
                        <li><a href="#">Threat Mitigation Strategy Framework</a></li>
                    </ul>
                </div>
                
                <div>
                    <h3>External Resources</h3>
                    <ul class="resources-list">
                        <li><a href="https://www.iucnredlist.org/" target="_blank">IUCN Red List of Threatened Species</a></li>
                        <li><a href="https://www.gbif.org/" target="_blank">Global Biodiversity Information Facility</a></li>
                        <li><a href="https://www.protectedplanet.net/" target="_blank">Protected Planet Database</a></li>
                        <li><a href="https://www.conservationevidence.com/" target="_blank">Conservation Evidence</a></li>
                        <li><a href="https://www.worldwildlife.org/" target="_blank">World Wildlife Fund Resources</a></li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
    
    <footer id="contact">
        <div class="footer-content">
            <div class="footer-column">
                <h3>Enviro Wise Consultancy</h3>
                <p>Providing evidence-based conservation solutions for a sustainable future.</p>
                <p>123 Conservation Way<br>Biodiversity Park, EC 12345</p>
            </div>
            
            <div class="footer-column">
                <h3>Contact Us</h3>
                <ul>
                    <li>Email: byarpar0@gmail.com</li>
                    <li>Phone: (+95) 770386642</li>
                    <li>Hours: Mon-Fri, 9am-5pm</li>
                </ul>
            </div>
            
            <div class="footer-column">
                <h3>Quick Links</h3>
                <ul>
                    <li><a href="#dashboard">Dashboard</a></li>
                    <li><a href="#reports">Reports</a></li>
                    <li><a href="#profiles">Species Profiles</a></li>
                    <li><a href="#visualizations">Visualizations</a></li>
                </ul>
            </div>
            
            <div class="footer-column">
                <h3>Our Services</h3>
                <ul>
                    <li><a href="#">Species Conservation Planning</a></li>
                    <li><a href="#">Habitat Assessment</a></li>
                    <li><a href="#">Environmental Impact Analysis</a></li>
                    <li><a href="#">Conservation Training</a></li>
                </ul>
            </div>
        </div>
        
        <div class="copyright">
            <div class="container">
                <p>&copy; ''' + str(datetime.now().year) + ''' Enviro Wise Consultancy. All rights reserved.</p>
                <p>Generated on ''' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '''</p>
            </div>
        </div>
    </footer>
    
    <script>
        // Simple search functionality
        document.querySelectorAll('.search-box').forEach(searchBox => {
            searchBox.addEventListener('input', function() {
                const searchTerm = this.value.toLowerCase();
                const fileList = this.nextElementSibling;
                const fileItems = fileList.querySelectorAll('.file-item');
                
                fileItems.forEach(item => {
                    const fileLink = item.querySelector('.file-link');
                    const text = fileLink.textContent.toLowerCase();
                    
                    if (text.includes(searchTerm)) {
                        item.style.display = '';
                    } else {
                        item.style.display = 'none';
                    }
                });
            });
        });
    </script>
</body>
</html>''')
    
    print(f"Dashboard created: {os.path.join(root_dir, 'index.html')}")
    return os.path.join(root_dir, 'index.html')

if __name__ == "__main__":
    dashboard_path = create_dashboard()
    print(f"Open the dashboard in your browser: file://{dashboard_path}")
