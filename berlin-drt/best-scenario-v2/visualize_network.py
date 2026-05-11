import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import time
import os

def visualize_sumo_network(xml_file):
    if not os.path.exists(xml_file):
        print(f"Error: Could not find file '{xml_file}'")
        return

    print(f"Parsing {xml_file} (this may take a moment for large networks)...")
    start_time = time.time()
    
    lines = []
    # iterparse allows parsing large XML files without loading the whole tree into memory
    context = ET.iterparse(xml_file, events=('end',))
    
    for event, elem in context:
        if elem.tag == 'edge':
            # Skip internal edges inside intersections for a cleaner and faster plot
            if elem.get('function') == 'internal':
                elem.clear()
                continue
                
            for lane in elem.findall('lane'):
                shape_str = lane.get('shape')
                if shape_str:
                    # Parse shape string "x1,y1 x2,y2 ..." into list of tuples [(x1,y1), (x2,y2), ...]
                    points = []
                    for pt in shape_str.split():
                        try:
                            x, y = map(float, pt.split(','))
                            points.append((x, y))
                        except ValueError:
                            continue
                    
                    if points:
                        lines.append(points)
            
            # Free memory to keep RAM usage very low
            elem.clear()
            
    print(f"Successfully extracted {len(lines)} lanes in {time.time() - start_time:.2f} seconds.")
    
    if not lines:
        print("No valid geometry found to plot.")
        return

    print("Rendering plot...")
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # LineCollection is significantly faster for plotting thousands of lines than a standard loop
    lc = LineCollection(lines, colors='#1f77b4', linewidths=0.3, alpha=0.7)
    ax.add_collection(lc)
    
    ax.autoscale()
    ax.set_aspect('equal') # Ensure the map isn't stretched
    
    # Set dark theme styling for better visibility of thin lines
    plt.style.use('dark_background')
    ax.set_facecolor('#111111')
    lc.set_color('#00ffff') # Cyan lines on dark background looks great for networks
    
    plt.title(f'SUMO Network: {os.path.basename(xml_file)}', fontsize=16, pad=20)
    plt.xlabel('X Coordinate (meters)', fontsize=12)
    plt.ylabel('Y Coordinate (meters)', fontsize=12)
    plt.tight_layout()
    
    print("Opening plot window...")
    plt.savefig('network_visualization.png', dpi=300, bbox_inches='tight')
    print("Plot saved as network_visualization.png")

if __name__ == "__main__":
    net_file = "berlin.net.xml"
    visualize_sumo_network(net_file)
