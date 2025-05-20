# update_dashboard.py
import os
import webbrowser
from create_dashboard import create_dashboard

def main():
    """Create the dashboard and open it in the default web browser."""
    dashboard_path = create_dashboard()
    
    # Convert to file URL format
    file_url = f"file://{os.path.abspath(dashboard_path)}"
    
    print(f"Opening dashboard: {file_url}")
    webbrowser.open(file_url)

if __name__ == "__main__":
    main()
