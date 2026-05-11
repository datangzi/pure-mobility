import xml.etree.ElementTree as ET
import csv
from pyproj import Transformer

MATSIM_NET = "berlin_network.xml"
SUMO_NET = "best-scenario-v2/berlin.net.xml"
OUTPUT_CSV = "network_mapping.csv"

# SUMO projection and offset
sumo_offset_x = -372355.98
sumo_offset_y = -5804722.35
transformer = Transformer.from_crs("EPSG:32633", "EPSG:31468", always_xy=True)

def sumo_to_matsim_coords(sumo_x, sumo_y):
    utm_x = sumo_x - sumo_offset_x
    utm_y = sumo_y - sumo_offset_y
    matsim_x, matsim_y = transformer.transform(utm_x, utm_y)
    return matsim_x, matsim_y

class SpatialIndex:
    def __init__(self, cell_size=500):
        self.cell_size = cell_size
        self.grid = {}
        self.points = []

    def add(self, id, x, y):
        cx = int(x / self.cell_size)
        cy = int(y / self.cell_size)
        if (cx, cy) not in self.grid:
            self.grid[(cx, cy)] = []
        self.grid[(cx, cy)].append((id, x, y))
        self.points.append((id, x, y))

    def nearest(self, x, y):
        cx = int(x / self.cell_size)
        cy = int(y / self.cell_size)
        
        min_dist = float('inf')
        nearest_id = None
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell = (cx + dx, cy + dy)
                if cell in self.grid:
                    for pid, px, py in self.grid[cell]:
                        dist = (px - x)**2 + (py - y)**2
                        if dist < min_dist:
                            min_dist = dist
                            nearest_id = pid
                            
        # If no neighbors in adjacent cells, fallback to linear search
        if nearest_id is None:
            for pid, px, py in self.points:
                dist = (px - x)**2 + (py - y)**2
                if dist < min_dist:
                    min_dist = dist
                    nearest_id = pid
        return nearest_id

def parse_matsim():
    print("Parsing MATSim network...")
    nodes = {}
    links = []
    
    context = ET.iterparse(MATSIM_NET, events=("end",))
    for event, elem in context:
        if elem.tag == "node":
            nodes[elem.attrib["id"]] = (float(elem.attrib["x"]), float(elem.attrib["y"]))
            elem.clear()
        elif elem.tag == "link":
            origid = ""
            for attr in elem.findall(".//attribute"):
                if attr.attrib.get("name") == "origid":
                    origid = attr.text
                    break
            links.append({
                "id": elem.attrib["id"],
                "from": elem.attrib["from"],
                "to": elem.attrib["to"],
                "origid": origid
            })
            elem.clear()
            
    return nodes, links

def parse_sumo():
    print("Parsing SUMO network...")
    junctions = []
    edges = []
    
    context = ET.iterparse(SUMO_NET, events=("end",))
    for event, elem in context:
        if elem.tag == "junction":
            if elem.attrib.get("type") != "internal" and not elem.attrib["id"].startswith(":"):
                junctions.append({
                    "id": elem.attrib["id"],
                    "x": float(elem.attrib["x"]),
                    "y": float(elem.attrib["y"])
                })
            elem.clear()
        elif elem.tag == "edge":
            if not elem.attrib["id"].startswith(":"):
                edges.append({
                    "id": elem.attrib["id"],
                    "from": elem.attrib.get("from", ""),
                    "to": elem.attrib.get("to", ""),
                    "shape": elem.attrib.get("shape", "")
                })
            elem.clear()
            
    return junctions, edges

def main():
    matsim_nodes, matsim_links = parse_matsim()
    sumo_junctions, sumo_edges = parse_sumo()
    
    # Store sumo junctions for easy lookup
    sumo_junc_dict = {j["id"]: (j["x"], j["y"]) for j in sumo_junctions}
    
    # 1. Map Nodes
    print("Mapping nodes...")
    node_index = SpatialIndex()
    for nid, (nx, ny) in matsim_nodes.items():
        node_index.add(nid, nx, ny)
        
    node_mapping = {}
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sumo_type", "sumo_id", "matsim_type", "matsim_id", "mapping_method"])
        
        mapped_nodes_exact = 0
        mapped_nodes_spatial = 0
        
        for j in sumo_junctions:
            sid = j["id"]
            if sid in matsim_nodes:
                mid = sid
                method = "Exact ID"
                mapped_nodes_exact += 1
            else:
                mx, my = sumo_to_matsim_coords(j["x"], j["y"])
                mid = node_index.nearest(mx, my)
                method = "Spatial Nearest"
                mapped_nodes_spatial += 1
            
            node_mapping[sid] = mid
            writer.writerow(["junction", sid, "node", mid, method])
            
        print(f"Nodes mapped: {mapped_nodes_exact} exact, {mapped_nodes_spatial} spatial fallback.")
        
        # Build MATSim edge lookups
        print("Building link lookups...")
        links_by_from_to = {}
        links_by_origid = {}
        link_index = SpatialIndex()
        
        for l in matsim_links:
            ft = (l["from"], l["to"])
            if ft not in links_by_from_to:
                links_by_from_to[ft] = []
            links_by_from_to[ft].append(l)
            
            if l["origid"]:
                if l["origid"] not in links_by_origid:
                    links_by_origid[l["origid"]] = []
                links_by_origid[l["origid"]].append(l)
                
            # Midpoint for spatial fallback
            if l["from"] in matsim_nodes and l["to"] in matsim_nodes:
                fx, fy = matsim_nodes[l["from"]]
                tx, ty = matsim_nodes[l["to"]]
                mx, my = (fx + tx) / 2.0, (fy + ty) / 2.0
                link_index.add(l["id"], mx, my)
                
        print("Mapping edges...")
        mapped_edges_topo = 0
        mapped_edges_origid = 0
        mapped_edges_spatial = 0
        
        for e in sumo_edges:
            sid = e["id"]
            s_from = e["from"]
            s_to = e["to"]
            
            m_from = node_mapping.get(s_from)
            m_to = node_mapping.get(s_to)
            
            mid = None
            method = ""
            
            # 1. Topo Match
            if m_from and m_to and (m_from, m_to) in links_by_from_to:
                candidates = links_by_from_to[(m_from, m_to)]
                # try to resolve tie by base id
                base_id = sid.split("#")[0].replace("-", "")
                best_cand = candidates[0]
                for c in candidates:
                    if c["origid"] == base_id:
                        best_cand = c
                        break
                mid = best_cand["id"]
                method = "Topological (from, to)"
                mapped_edges_topo += 1
                
            # 2. OrigID Match
            elif sid:
                base_id = sid.split("#")[0].replace("-", "")
                if base_id in links_by_origid:
                    mid = links_by_origid[base_id][0]["id"]
                    method = "Base OSM ID"
                    mapped_edges_origid += 1
                    
            # 3. Spatial Match
            if not mid:
                # Calculate sumo edge midpoint
                if e["shape"]:
                    coords = e["shape"].split()
                    mid_idx = len(coords) // 2
                    c = coords[mid_idx].split(",")
                    sx, sy = float(c[0]), float(c[1])
                else:
                    if s_from in sumo_junc_dict and s_to in sumo_junc_dict:
                        x1, y1 = sumo_junc_dict[s_from]
                        x2, y2 = sumo_junc_dict[s_to]
                        sx, sy = (x1+x2)/2.0, (y1+y2)/2.0
                    else:
                        sx, sy = 0, 0
                
                mx, my = sumo_to_matsim_coords(sx, sy)
                mid = link_index.nearest(mx, my)
                method = "Spatial Nearest"
                mapped_edges_spatial += 1
                
            writer.writerow(["edge", sid, "link", mid, method])
            
        print(f"Edges mapped: {mapped_edges_topo} topo, {mapped_edges_origid} origid, {mapped_edges_spatial} spatial fallback.")
        print(f"Mapping complete. Results written to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
