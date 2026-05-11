import csv
import math
from pyproj import Transformer

def main():
    print("Initializing coordinate transformer for EPSG:4326 to EPSG:25833...")
    # always_xy=True ensures coordinates are handled as (lon, lat) regardless of pyproj version
    transformer = Transformer.from_crs("epsg:4326", "epsg:25833", always_xy=True)

    nodes = {}
    print("Reading nodes_osm.csv...")
    with open('nodes_osm.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row['id']
            lon = float(row['x'])
            lat = float(row['y'])
            # Project from WGS84 (lon, lat) to EPSG:25833
            x, y = transformer.transform(lon, lat)
            nodes[node_id] = {'x': x, 'y': y}
    
    print(f"Loaded {len(nodes)} nodes.")

    edges = []
    print("Reading edges_osm.csv...")
    with open('edges_osm.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        link_id = 1
        for row in reader:
            from_node = row['from_node']
            to_node = row['to_node']
            travel_time = float(row['travel_time'])
            
            if from_node not in nodes or to_node not in nodes:
                print(f"Warning: Edge {from_node}->{to_node} skips missing nodes.")
                continue
            
            x1, y1 = nodes[from_node]['x'], nodes[from_node]['y']
            x2, y2 = nodes[to_node]['x'], nodes[to_node]['y']
            
            # Calculate Euclidean distance on the projected CRS
            length = math.hypot(x2 - x1, y2 - y1)
            
            # Ensure travel time is at least a very small value to avoid division by zero
            tt = travel_time if travel_time > 0 else 1.0
            freespeed = length / tt
            
            edges.append({
                'id': str(link_id),
                'from': from_node,
                'to': to_node,
                'length': length,
                'freespeed': freespeed,
                'capacity': 600.0,
                'permlanes': 1.0,
                'oneway': 1,
                'modes': "ride,car,freight" # Used the ones from berlin_network.xml
            })
            link_id += 1
            
    print(f"Loaded {len(edges)} edges.")

    print("Writing test_network.xml...")
    with open('test_network.xml', 'w', encoding='utf-8') as f:
        # Standard MATSim Network Header
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE network SYSTEM "http://www.matsim.org/files/dtd/network_v2.dtd">\n')
        f.write('<network>\n\n')
        
        # Write Nodes
        f.write('<!-- ====================================================================== -->\n\n')
        f.write('\t<nodes>\n')
        for node_id, data in nodes.items():
            f.write(f'\t\t<node id="{node_id}" x="{data["x"]}" y="{data["y"]}" >\n\t\t</node>\n')
        f.write('\t</nodes>\n\n')
        
        # Write Links
        f.write('<!-- ====================================================================== -->\n\n')
        f.write('\t<links>\n')
        for edge in edges:
            f.write(f'\t\t<link id="{edge["id"]}" from="{edge["from"]}" to="{edge["to"]}" ')
            f.write(f'length="{edge["length"]}" freespeed="{edge["freespeed"]}" ')
            f.write(f'capacity="{edge["capacity"]}" permlanes="{edge["permlanes"]}" ')
            f.write(f'oneway="{edge["oneway"]}" modes="{edge["modes"]}" >\n')
            f.write('\t\t\t<attributes>\n')
            f.write(f'\t\t\t\t<attribute name="origid" class="java.lang.String">{edge["id"]}</attribute>\n')
            f.write('\t\t\t\t<attribute name="type" class="java.lang.String">tertiary</attribute>\n')
            f.write('\t\t\t</attributes>\n')
            f.write('\t\t</link>\n')
        f.write('\t</links>\n\n')
        
        f.write('<!-- ====================================================================== -->\n\n')
        f.write('</network>\n')
        
    print("Successfully generated test_network.xml")

if __name__ == '__main__':
    main()
