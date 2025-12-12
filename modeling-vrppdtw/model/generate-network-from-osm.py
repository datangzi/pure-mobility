import os
from typing import Tuple, Iterable
import re

import pandas as pd
import osmnx as ox


def build_graph(place: str):
    ox.settings.use_cache = True
    ox.settings.log_console = True
    custom = (
        '["highway"~"primary|secondary|primary_link|secondary_link"]'
    )
    return ox.graph_from_place(place, custom_filter=custom, network_type="drive")
    # return ox.graph_from_place(place, network_type="drive")


def _prune_isolated_nodes(G):
    """Remove isolated nodes using available OSMnx API or a networkx fallback."""
    try:
        # OSMnx >= 2.0
        return getattr(ox, 'graph').remove_isolated_nodes(G)
    except Exception:
        try:
            # OSMnx < 2.0
            return getattr(ox, 'utils_graph').remove_isolated_nodes(G)
        except Exception:
            # Fallback: networkx (remove nodes with degree 0)
            try:
                import networkx as nx
                iso = [n for n, d in G.degree() if d == 0]
                G.remove_nodes_from(iso)
            except Exception:
                pass
            return G


def filter_by_min_speed(G, min_kph: float = 5.0):
    """Filter graph to keep only edges with speed >= min_kph.

    Uses OSMnx's imputed 'speed_kph' attribute. Removes slower edges and prunes
    isolated nodes.
    """
    # Ensure speeds available
    G = ox.add_edge_speeds(G)

    to_remove = []
    for u, v, k, d in G.edges(keys=True, data=True):
        spd = d.get("speed_kph")
        try:
            keep = (float(spd) >= float(min_kph))
        except Exception:
            keep = False
        if not keep:
            to_remove.append((u, v, k))

    if to_remove:
        G.remove_edges_from(to_remove)

    # Drop isolated nodes (version-compatible)
    G = _prune_isolated_nodes(G)
    return G


def _to_int(value):
    """Best-effort conversion of OSM tag to int.

    - Lists/tuples: return max of parsed ints
    - Strings: extract first integer substring; if multiple, return max
    - Numbers: cast to int
    - Else: return None
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        nums = [ _to_int(v) for v in value ]
        nums = [n for n in nums if n is not None]
        return max(nums) if nums else None
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except Exception:
            return None
    if isinstance(value, str):
        # OSM can have formats like "2", "2;1", etc.
        parts = re.findall(r"\d+", value)
        if parts:
            try:
                return max(int(p) for p in parts)
            except Exception:
                return None
    return None


def filter_by_min_lanes_per_direction(G, min_lanes_per_dir: int = 2):
    """Filter graph to keep only edges with at least min lanes per direction.

    Heuristics per edge (directed):
    - If oneway: use 'lanes' (or lanes:forward/lanes:backward) for that edge.
      Remove if lanes < min.
    - If two-way and lanes:forward/backward available: require both >= min;
      otherwise remove (applies to each directed edge to avoid ambiguity).
    - If only total 'lanes' available on two-way: estimate per-dir as floor(lanes/2)
      and remove if < min.
    """
    to_remove = []
    for u, v, k, d in G.edges(keys=True, data=True):
        oneway = d.get('oneway')
        is_oneway = str(oneway).lower() in ('true', '1', 'yes') if oneway is not None else False

        lanes_total = _to_int(d.get('lanes'))
        lanes_fwd   = _to_int(d.get('lanes:forward'))
        lanes_back  = _to_int(d.get('lanes:backward'))

        keep = True
        if is_oneway:
            lanes_dir = lanes_total
            if lanes_dir is None:
                lanes_dir = lanes_fwd or lanes_back
            if lanes_dir is None:
                # Unknown: be conservative and drop
                keep = False
            else:
                keep = (lanes_dir >= min_lanes_per_dir)
        else:
            if lanes_fwd is not None or lanes_back is not None:
                # Require both directions to have sufficient lanes
                if lanes_fwd is None or lanes_back is None:
                    keep = False
                else:
                    keep = (lanes_fwd >= min_lanes_per_dir and lanes_back >= min_lanes_per_dir)
            elif lanes_total is not None:
                # Estimate per-direction lanes
                per_dir = lanes_total // 2
                keep = (per_dir >= min_lanes_per_dir)
            else:
                keep = False

        if not keep:
            to_remove.append((u, v, k))

    if to_remove:
        G.remove_edges_from(to_remove)
    G = _prune_isolated_nodes(G)
    return G


def _env_bool(name: str, default: bool) -> bool:
    """Parse boolean-like env var. Accepts 1/0, true/false, yes/no, on/off."""
    val = os.getenv(name)
    if val is None:
        return default
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return default


def export_nodes_edges(
    G,
    nodes_csv: str = "nodes_osm.csv",
    edges_csv: str = "edges_osm.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Export nodes and edges of G to CSV.

    Nodes CSV columns: id, x, y, street  (x=lon, y=lat)
    Edges CSV columns: from_node, to_node, travel_time, cost
      - travel_time in seconds (via OSMnx speed and length)
      - cost default 0.0 (placeholder)
    """

    # Ensure edge speeds (km/h) and travel times (seconds) exist
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    # --- nodes_osm.csv ---
    def _edge_names_for_node(nid) -> str:
        names = []
        try:
            for _, _, _, ed in G.edges(nid, keys=True, data=True):
                nm = ed.get("name")
                if nm is None:
                    continue
                if isinstance(nm, (list, tuple, set)):
                    for v in nm:
                        if v is not None:
                            names.append(str(v))
                else:
                    names.append(str(nm))
        except Exception:
            pass
        # Deduplicate preserving order
        seen = set()
        uniq = []
        for s in names:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        return ";".join(uniq)

    nodes_rows = []
    for nid, data in G.nodes(data=True):
        # nid is the OSM node id (int); x=lon, y=lat
        x = data.get("x")
        y = data.get("y")
        nodes_rows.append({"id": nid, "x": x, "y": y})
    nodes_df = pd.DataFrame(nodes_rows)
    nodes_df.to_csv(nodes_csv, index=False)

    # --- edges_osm.csv ---
    edges_rows = []
    for u, v, k, data in G.edges(keys=True, data=True):
        tt = data.get("travel_time")
        # Cast to int seconds for downstream compatibility
        tt_int = int(tt) if tt is not None else None
        edges_rows.append({
            "from_node": u,
            "to_node": v,
            "travel_time": tt_int
        })
    edges_df = pd.DataFrame(edges_rows)
    edges_df.to_csv(edges_csv, index=False)

    return nodes_df, edges_df


def main():
    place = "Charlottenburg, Berlin, Germany"
    print(f"Downloading drivable network for: {place}")
    G = build_graph(place)
    print(f"Raw graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Optionally keep only edges with speed >= MIN_SPEED_KPH
    if _env_bool('ENABLE_SPEED_FILTER', False):
        try:
            env_min = os.getenv('MIN_SPEED_KPH')
            min_kph = float(env_min) if env_min else 50.0
        except Exception:
            min_kph = 50.0
        G = filter_by_min_speed(G, min_kph=min_kph)
        print(f"Speed-filtered (>= {min_kph} kph): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    else:
        print("Speed filter disabled (ENABLE_SPEED_FILTER=false)")

    # Optionally keep only edges with at least MIN_LANES_PER_DIR lanes per direction
    if _env_bool('ENABLE_LANES_FILTER', False):
        try:
            env_lanes = os.getenv('MIN_LANES_PER_DIR')
            min_lanes = int(env_lanes) if env_lanes else 2
        except Exception:
            min_lanes = 2
        G = filter_by_min_lanes_per_direction(G, min_lanes_per_dir=min_lanes)
        print(f"Lane-filtered (>= {min_lanes} per dir): {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    else:
        print("Lane filter disabled (ENABLE_LANES_FILTER=false)")
    nodes_df, edges_df = export_nodes_edges(G)
    print(
        f"Wrote {len(nodes_df)} nodes to 'nodes_osm.csv' and "
        f"{len(edges_df)} edges to 'edges_osm.csv'."
    )


if __name__ == "__main__":
    main()
