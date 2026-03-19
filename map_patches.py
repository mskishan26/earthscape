import json
import csv
import folium
from folium import GeoJson
import numpy as np
from folium import plugins

def create_patch_map(geojson_file, split_csv_file=None, output_file='patch_map.html'):
    """
    Create an interactive map from GeoJSON file showing patches with their IDs.
    
    Args:
        geojson_file: Path to the GeoJSON file
        split_csv_file: Optional path to CSV file with split information
        output_file: Output HTML file for the map
    """
    
    # Load the GeoJSON data
    with open(geojson_file, 'r') as f:
        geojson_data = json.load(f)
    
    # Load split information if provided
    split_map = {}
    if split_csv_file:
        with open(split_csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                split_map[row['patch_id']] = row['split']
        
        # Add split information to GeoJSON features
        for feature in geojson_data['features']:
            patch_id = feature['properties']['patch_id']
            feature['properties']['split'] = split_map.get(patch_id, 'unknown')
    
    # Try to transform coordinates from EPSG:3089 to EPSG:4326 (lat/lon)
    try:
        from pyproj import Transformer
        
        # Create transformer from EPSG:3089 to EPSG:4326
        transformer = Transformer.from_crs("EPSG:3089", "EPSG:4326", always_xy=True)
        
        # Transform all coordinates in the GeoJSON
        for feature in geojson_data['features']:
            if feature['geometry']['type'] == 'Polygon':
                new_coords = []
                for ring in feature['geometry']['coordinates']:
                    new_ring = []
                    for coord in ring:
                        lon, lat = transformer.transform(coord[0], coord[1])
                        new_ring.append([lon, lat])
                    new_coords.append(new_ring)
                feature['geometry']['coordinates'] = new_coords
        
        # Calculate bounds from all features
        all_lats = []
        all_lons = []
        for feature in geojson_data['features']:
            if feature['geometry']['type'] == 'Polygon':
                for ring in feature['geometry']['coordinates']:
                    for coord in ring:
                        all_lons.append(coord[0])
                        all_lats.append(coord[1])
        
        center_lat = np.mean(all_lats)
        center_lon = np.mean(all_lons)
        
    except ImportError:
        print("Warning: pyproj not installed. Map may not display correctly.")
        print("Install with: uv add pyproj")
        center_lat, center_lon = 0, 0
    
    # Create the map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Define colors for each split
    split_colors = {
        'train': '#2ecc71',  # Green
        'val': '#3498db',    # Blue
        'test': '#e74c3c',   # Red
        'unknown': '#95a5a6' # Gray
    }
    
    # Add a function to style the patches and show patch IDs
    def style_function(feature):
        split = feature['properties'].get('split', 'unknown')
        color = split_colors.get(split, '#95a5a6')
        return {
            'fillColor': color,
            'color': color,
            'weight': 2,
            'fillOpacity': 0.5,
        }
    
    def highlight_function(feature):
        return {
            'fillColor': '#ff7800',
            'color': '#ff7800',
            'weight': 3,
            'fillOpacity': 0.5,
        }
    
    # Add GeoJSON layer with tooltips
    tooltip_fields = ['patch_id', 'split'] if split_csv_file else ['patch_id']
    tooltip_aliases = ['Patch ID:', 'Split:'] if split_csv_file else ['Patch ID:']
    
    geojson_layer = GeoJson(
        geojson_data,
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True,
            sticky=False,
            labels=True,
            style="""
                background-color: #F0EFEF;
                border: 2px solid black;
                border-radius: 3px;
                box-shadow: 3px 3px 3px rgba(0, 0, 0, 0.4);
                font-size: 12px;
                font-weight: bold;
            """
        )
    )
    
    geojson_layer.add_to(m)
    
    # Fit map bounds to show all patches
    try:
        if all_lats and all_lons:
            bounds = [[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]]
            m.fit_bounds(bounds)
    except:
        pass
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add fullscreen button
    plugins.Fullscreen().add_to(m)
    
    # Add legend if splits are provided
    if split_csv_file:
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 180px; height: 140px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px; border-radius: 5px;
                    box-shadow: 3px 3px 5px rgba(0,0,0,0.3);">
            <p style="margin: 0 0 10px 0; font-weight: bold; text-align: center;">Split Legend</p>
            <p style="margin: 5px 0;"><i style="background: #2ecc71; width: 20px; height: 20px; 
                display: inline-block; border: 1px solid black;"></i> Train</p>
            <p style="margin: 5px 0;"><i style="background: #3498db; width: 20px; height: 20px; 
                display: inline-block; border: 1px solid black;"></i> Val</p>
            <p style="margin: 5px 0;"><i style="background: #e74c3c; width: 20px; height: 20px; 
                display: inline-block; border: 1px solid black;"></i> Test</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save the map
    m.save(output_file)
    print(f"Map saved to {output_file}")
    print(f"Total patches: {len(geojson_data['features'])}")
    print(f"Center: ({center_lat:.6f}, {center_lon:.6f})")
    
    if split_csv_file:
        train_count = sum(1 for f in geojson_data['features'] if f['properties'].get('split') == 'train')
        val_count = sum(1 for f in geojson_data['features'] if f['properties'].get('split') == 'val')
        test_count = sum(1 for f in geojson_data['features'] if f['properties'].get('split') == 'test')
        print(f"  Train: {train_count} | Val: {val_count} | Test: {test_count}")
    
    return m

# Alternative version using matplotlib for static maps
def create_static_map(geojson_file, output_file='patch_map_static.png'):
    """
    Create a static map using matplotlib and geopandas.
    
    Args:
        geojson_file: Path to the GeoJSON file
        output_file: Output PNG file for the map
    """
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
        
        # Load the GeoJSON
        gdf = gpd.read_file(geojson_file)
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        
        # Plot the patches
        gdf.plot(ax=ax, facecolor='lightblue', edgecolor='blue', linewidth=0.5, alpha=0.7)
        
        # Add patch IDs as labels (sample to avoid overcrowding)
        for idx, row in gdf.iterrows():
            if idx % 5 == 0:  # Label every 5th patch to avoid overcrowding
                centroid = row.geometry.centroid
                ax.annotate(
                    row['patch_id'], 
                    xy=(centroid.x, centroid.y), 
                    xytext=(3, 3), 
                    textcoords='offset points',
                    fontsize=6,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
                )
        
        ax.set_title('Patch Map with IDs', fontsize=16, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Static map saved to {output_file}")
        plt.show()
        
    except ImportError:
        print("For static maps, install: pip install geopandas matplotlib")

if __name__ == "__main__":
    # Usage
    geojson_file = "locations.geojson"
    split_csv_file = "split.csv"
    
    print("Creating interactive map with split colors...")
    create_patch_map(geojson_file, split_csv_file, 'interactive_patch_map.html')
    
    print("\nCreating static map...")
    create_static_map(geojson_file, 'static_patch_map.png')