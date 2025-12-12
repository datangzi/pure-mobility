import os
import pandas as pd
import numpy as np
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import threading
import time
import math
import heapq
from collections import defaultdict
from itertools import product

# ================== Script Description ===================
' vrp with 1 vehicle and 1 passenger'
' network generated from osm'
' starting and service points selected interactively on map'
' minimizing the total travel time while solving the optimization problem using DP on a state-space-time network'

# ---------- Helpers ----------
def norm_node(x):
    """Return node id as a clean string without trailing .0 etc."""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

# =================== Define Road Network ===================
# path of nodes and edges csv files
current_dir = os.path.dirname(os.path.abspath(__file__))
nodes_path = os.path.join(current_dir, 'nodes_osm.csv')
edges_path = os.path.join(current_dir, 'edges_osm.csv')

def load_network_data():
    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"nodes file not found at {nodes_path}")
    if not os.path.exists(edges_path):
        raise FileNotFoundError(f"edges file not found at {edges_path}")
    
    try:
        # Read node data
        nodes_df = pd.read_csv(nodes_path)
        if 'id' not in nodes_df.columns:
            raise ValueError("nodes file must contain an 'id' column")

        # Normalize node ids to strings
        nodes_df['id'] = nodes_df['id'].apply(norm_node)
        nodes = nodes_df['id'].tolist()

        # Read street data
        edges_df = pd.read_csv(edges_path)
        required_cols = {'from_node', 'to_node', 'travel_time'}
        if not required_cols.issubset(edges_df.columns):
            miss = required_cols - set(edges_df.columns)
            raise ValueError(f"edges file missing columns: {miss}")

        # Normalize street endpoints to strings to match 'nodes'
        edges_df['from_node'] = edges_df['from_node'].apply(norm_node)
        edges_df['to_node']   = edges_df['to_node'].apply(norm_node)

        # Times/costs as numbers
        edges_df['travel_time'] = edges_df['travel_time'].astype(float).astype(int)

        edges = set()
        travel_times = {}

        for _, row in edges_df.iterrows():
            i, j = row['from_node'], row['to_node']  # strings
            edges.add((i, j))
            travel_times[(i, j)] = int(row['travel_time'])

        return nodes, edges, travel_times, nodes_df, edges_df

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load network data: {str(e)}")
        raise

# Load network data
try:
    nodes, edges, travel_times, nodes_df, edges_df = load_network_data()
    print("Network loaded successfully:")
except Exception as e:
    print(f"Error: {str(e)}")
    input("Press Enter to exit...")
    raise SystemExit

# ============= Select Starting and Service Points on Map ===============
def select_points_on_map(nodes_df, edges_df=None):
    'Interactive selection of 3 points by clicking on a plotted network.'
    'Order: START, PICKUP, DROPOFF'
    'Returns tuple of normalized node ids (as strings).'
    
    # Prepare coordinates
    xs = nodes_df['x'].to_numpy()
    ys = nodes_df['y'].to_numpy()
    ids = nodes_df['id'].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(
        "Click in order: START, PICKUP, DROPOFF",
        fontsize=10,
    )

    # Draw edges first for context, if available
    if edges_df is not None:
        id_to_xy = {row['id']: (row['x'], row['y']) for _, row in nodes_df.iterrows()}
        for _, r in edges_df.iterrows():
            u, v = r['from_node'], r['to_node']
            if u in id_to_xy and v in id_to_xy:
                x0, y0 = id_to_xy[u]
                x1, y1 = id_to_xy[v]
                ax.plot([x0, x1], [y0, y1], color='lightgray', linewidth=0.5, alpha=0.6)

    # Draw nodes
    ax.scatter(xs, ys, s=5, c='black', alpha=0.7)
    ax.set_xlabel('Longitude (x)')
    ax.set_ylabel('Latitude (y)')
    ax.set_aspect('equal', adjustable='datalim')

    # Capture clicks
    want = 3
    clicks = plt.ginput(want, timeout=0)  # wait for 3 clicks

    if len(clicks) < want:
        plt.close(fig)
        raise RuntimeError("Not enough points selected. Please click 3 points.")

    # For each click, find nearest node by Euclidean distance in lon/lat
    sel_ids = []
    ptsx = np.array([c[0] for c in clicks])
    ptsy = np.array([c[1] for c in clicks])
    for cx, cy in zip(ptsx, ptsy):
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        idx = int(np.argmin(d2))
        sel_ids.append(norm_node(ids[idx]))
        # mark selected
        ax.scatter([xs[idx]], [ys[idx]], s=40, c='red')

    fig.canvas.draw()
    plt.pause(0.5)
    plt.close(fig)

    (
        START,
        PICKUP,
        DROPOFF,
    ) = sel_ids

    return START, PICKUP, DROPOFF

try:
    (
        START,
        PICKUP,
        DROPOFF,
    ) = select_points_on_map(nodes_df, edges_df)
    print(f"Map selection complete. Vehicle starts at: {START}, Passenger will be picked up at: {PICKUP}, dropped off at: {DROPOFF}")

except Exception as e:
    messagebox.showerror("Error while selecting starting and service points", str(e))
    raise SystemExit


# ====================== TIME HORIZON ESTIMATION ======================
# Build adjacency for Dijkstra on travel_time

def _build_adj(travel_times_dict):
    adj = {}
    for (u, v), w in travel_times_dict.items():
        adj.setdefault(u, []).append((v, int(w)))
    return adj

def _shortest_time(u, v, adj):
    if u == v:
        return 0
    dist = {u: 0}
    pq = [(0, u)]
    seen = set()
    while pq:
        d, x = heapq.heappop(pq)
        if x == v:
            return d
        if x in seen:
            continue
        seen.add(x)
        for y, w in adj.get(x, []):
            nd = d + w
            if y not in dist or nd < dist[y]:
                dist[y] = nd
                heapq.heappush(pq, (nd, y))
    return float('inf')

def _route_time(start, seq, adj):
    t = 0
    prev = start
    for node in seq:
        d = _shortest_time(prev, node, adj)
        if not np.isfinite(d):
            raise ValueError(f"No path from {prev} to {node} in network")
        t += int(d)
        prev = node
    return t

from itertools import permutations

def _precedence_sequences(p1, d1, p2, d2):
    nodes4 = [p1, d1, p2, d2]
    for perm in permutations(nodes4):
        if perm.index(p1) < perm.index(d1) and perm.index(p2) < perm.index(d2):
            yield perm

# Compute dynamic time horizon T based on selected points and network
adj = _build_adj(travel_times)

try:
    T_base = _route_time(START, [PICKUP, DROPOFF], adj)
    BUFFER_RATIO = 0.25  # 25% slack
    BUFFER_STATIC = 300  # +5 minutes in seconds
    T_seconds = int(T_base * (1 + BUFFER_RATIO)) + BUFFER_STATIC
    # Discretization step (seconds per step) - can override via env VRP_TIME_STEP
    try:
        TIME_STEP = int(os.getenv('VRP_TIME_STEP', '10'))
    except Exception:
        TIME_STEP = 10
    if TIME_STEP <= 0:
        TIME_STEP = 10
    T_steps = int(math.ceil(T_seconds / TIME_STEP))
    # Time parameters are in discrete steps
    times = range(0, T_steps + 1)
    print(f"Dynamic time horizon set to {T_seconds}s with step {TIME_STEP}s -> {T_steps} steps")
except Exception as e:
    # Fallback if routing fails: keep a conservative default
    print(f"Time horizon estimation failed: {e}. Falling back to default T=3600s, step=10s")
    TIME_STEP = 10
    T_steps = int(math.ceil(3600 / TIME_STEP))
    times = range(0, T_steps + 1)

states = ['_', 'p']  # Passenger carrying state (capacity = 1)


# ====================== NETWORK REDUCTION ======================
# Reduce the network to the union of shortest paths between key points

def _shortest_path(u, v, adj):
    """Return list of nodes for shortest path u->v using travel time weights.
    Raises if unreachable.
    """
    if u == v:
        return [u]
    pq = [(0, u)]
    dist = {u: 0}
    prev = {}
    seen = set()
    while pq:
        d, x = heapq.heappop(pq)
        if x in seen:
            continue
        seen.add(x)
        if x == v:
            # reconstruct
            path = [v]
            while path[-1] != u:
                path.append(prev[path[-1]])
            path.reverse()
            return path
        for y, w in adj.get(x, []):
            nd = d + int(w)
            if y not in dist or nd < dist[y]:
                dist[y] = nd
                prev[y] = x
                heapq.heappush(pq, (nd, y))
    raise RuntimeError(f"Unreachable: {u} -> {v}")

key_nodes_list = [START, PICKUP, DROPOFF]
reduced_nodes = set()
reduced_edges = set()

for a, b in product(key_nodes_list, key_nodes_list):
    if a == b:
        continue
    try:
        path = _shortest_path(a, b, adj)
    except Exception:
        continue
    reduced_nodes.update(path)
    for uu, vv in zip(path[:-1], path[1:]):
        reduced_edges.add((uu, vv))

if reduced_edges:
    # Restrict problem to reduced subgraph
    nodes = [norm_node(n) for n in reduced_nodes]
    # ensure uniqueness and stable order
    nodes = sorted(set(nodes))

    # Filter edges, travel_times
    edges = set((norm_node(i), norm_node(j)) for (i, j) in reduced_edges if (i, j) in travel_times)
    travel_times = {(norm_node(i), norm_node(j)): int(travel_times[(i, j)]) for (i, j) in reduced_edges if (i, j) in travel_times}

print(f"Reduced network: {len(nodes)} nodes, {len(edges)} edges (from union of shortest paths)")

# ====================== STATE-SPACE-TIME NETWORK (STS) ======================
# Generate 3D vertices (i, t, w)
vertexs = [(i, t, w) for i in nodes for t in times for w in states]

# Transport arcs (move along physical edges)
arcsTransport = []
for (i, j) in edges:
    for t in times:
        required_steps = int(math.ceil(travel_times[(i, j)] / TIME_STEP))
        s = t + required_steps
        if s <= max(times):
            for w in states:
                arcsTransport.append((i, j, t, s, w, w))

# Service arcs (Pickup/Dropoff)
arcsService = (
    [(PICKUP, PICKUP, t, t, '_', 'p') for t in times] +   # Pickup
    [(DROPOFF, DROPOFF, t, t, 'p', '_') for t in times]   # Dropoff
)

# Waiting arcs
arcsWaiting = [(i, i, t, t + 1, w, w)
               for i in nodes
               for t in range(0, max(times))  # last waiting until T-1 -> T
               for w in states]

# Combine all arcs (no super-sink Omega needed for DP)
arcsSTS = arcsTransport + arcsService + arcsWaiting

# Fast arc-type membership sets
setTransport = set(arcsTransport)
setWaiting   = set(arcsWaiting)
setService   = set(arcsService)

# Travel time (arc cost in seconds)
tt = {}
for arc in arcsSTS:
    if arc in setTransport:
        i, j = arc[0], arc[1]
        tt[arc] = travel_times[(i, j)]
    elif arc in setWaiting:
        tt[arc] = (arc[3] - arc[2]) * TIME_STEP  # convert steps to seconds
    elif arc in setService:
        tt[arc] = 0
    else:
        tt[arc] = 0

print("STS network size summary:")
print(f"- Physical nodes: {len(nodes)}; Physical edges: {len(edges)}")
print(f"- Time steps: {len(times)}; Passenger states: {len(states)}")
print(f"- Arcs transport: {len(arcsTransport)}; waiting: {len(arcsWaiting)}; service: {len(arcsService)}")
print(f"- Total STS arcs: {len(arcsSTS)}")

# ====================== DP SHORTEST-PATH SOLVER ======================
def solve_vrppdtw_dp(START, DROPOFF, arcsSTS, arcsService, tt):
    """
    Solve 1-vehicle / 1-passenger VRPPDTW via shortest path on the STS network.
    - Start at (START, t=0, w='_')
    - Must perform one pickup at PICKUP and one dropoff at DROPOFF
      (this is enforced by the structure of service arcs)
    - Objective: minimize total travel time (sum of tt[arc])
    Returns:
        arcs_solution: list of arcs (i, j, t, s, w, w')
        best_cost: total travel time in seconds
    """
    # Build adjacency list from STS arcs
    adj = defaultdict(list)
    for arc in arcsSTS:
        i, j, t, s, w, wp = arc
        cost = tt[arc]
        adj[(i, t, w)].append(((j, s, wp), cost, arc))

    # Determine "post-dropoff" vertices reachable via dropoff service arcs
    dropoff_vertices = set()
    for arc in arcsService:
        i, j, t, s, w, wp = arc
        if i == DROPOFF and w == 'p' and wp == '_':
            dropoff_vertices.add((j, s, wp))
    if not dropoff_vertices:
        raise RuntimeError("No dropoff vertices found in arcsService")

    source = (START, 0, '_')
    INF = float('inf')
    dist = defaultdict(lambda: INF)
    dist[source] = 0.0
    prev = {}
    pq = [(0.0, source)]

    best_target = None
    best_cost = INF

    print("Starting DP shortest-path search on STS network...")

    # Dijkstra on STS network
    while pq:
        d, v = heapq.heappop(pq)
        # Skip outdated entries
        if d > dist[v] + 1e-9:
            continue

        # If v is a dropoff vertex, we can stop: first such vertex popped is optimal
        if v in dropoff_vertices:
            best_target = v
            best_cost = d
            break

        for (v2, cost, arc) in adj.get(v, []):
            nd = d + cost
            if nd + 1e-9 < dist[v2]:
                dist[v2] = nd
                prev[v2] = (v, arc)
                heapq.heappush(pq, (nd, v2))

    if best_target is None:
        raise RuntimeError("No feasible path from START to DROPOFF under current time horizon.")

    # Reconstruct path from best_target back to source
    arcs_solution = []
    cur = best_target
    while cur != source:
        pv, arc = prev[cur]
        arcs_solution.append(arc)
        cur = pv
    arcs_solution.reverse()

    return arcs_solution, best_cost

# ================ Results & PRINTING ================
def prettify_solution_df(df):
    """
    Ensure 'i' and 'j' are strings without .0 (for printing).
    """
    df = df.copy()
    df['i'] = df['i'].apply(norm_node)
    df['j'] = df['j'].apply(norm_node)
    return df

def print_sorted_solutions(df_v):
    dv = prettify_solution_df(df_v)
    # Create numeric sort keys (safe if IDs are digits; non-digits become NaN and sorted last)
    dv['_i_num'] = pd.to_numeric(dv['i'], errors='coerce')
    dv['_j_num'] = pd.to_numeric(dv['j'], errors='coerce')

    dv = dv.sort_values(by=['t', 's', '_i_num', '_j_num', 'i', 'j'])

    print("\nVehicle Path (Sorted):")
    print(dv[['i', 'j', 't', 's', 'w', "w'"]].to_string(index=False))

# ---- Run DP solver ----
solved = False
try:
    arcs_v, best_cost = solve_vrppdtw_dp(START, DROPOFF, arcsSTS, arcsService, tt)
    df_solution_v = pd.DataFrame(arcs_v, columns=["i", "j", "t", "s", "w", "w'"])
    print("DP optimization complete.")
    print(f"Best total travel time: {best_cost} seconds")
    print_sorted_solutions(df_solution_v)
    solved = True
except Exception as e:
    print(f"DP solver error: {e}")
    solved = False


# ================== ANIMATION ==================
def _nn(x):
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") else s

def classify_arc(row, setTransport, setWaiting, setService):
    tup = (row['i'], row['j'], row['t'], row['s'], row['w'], row["w'"])
    if tup in setTransport: return 'move'
    if tup in setWaiting:   return 'wait'
    if tup in setService:   return 'service'
    return 'other'

def build_segments(df, setTransport, setWaiting, setService, vehicle):
    df = df.copy()
    df['i'] = df['i'].apply(_nn)
    df['j'] = df['j'].apply(_nn)
    df = df.sort_values(['t', 's', 'i', 'j'])

    segs = []
    events = []  # list of (time:int, kind:str, payload:dict with 'p','node','veh')
    for _, r in df.iterrows():
        a_type = classify_arc(r, setTransport, setWaiting, setService)
        if a_type == 'service':
            if r['w'] == '_' and r["w'"] == 'p':
                events.append((int(r['t']), 'pickup', {'p': r["w'"], 'node': r['i'], 'veh': vehicle}))
            elif r['w'] == 'p' and r["w'"] == '_':
                events.append((int(r['t']), 'dropoff', {'p': r['w'], 'node': r['i'], 'veh': vehicle}))
        elif a_type in ('move', 'wait'):
            segs.append({'type': a_type, 'i': r['i'], 'j': r['j'], 't0': int(r['t']), 't1': int(r['s'])})
    return segs, events

def initial_node(df):
    if df.empty: return None
    r = df.sort_values(['t', 's']).iloc[0]
    return _nn(r['i'])

def pickup_node_from_df(df, pax):
    svc = df[(df['w'] == '_') & (df["w'"] == pax)]
    if svc.empty: return None
    return _nn(svc.iloc[0]['i'])

def create_layout_from_coords(nodes_df):
    """Create position dictionary from actual x,y coordinates in nodes dataframe"""
    pos = {}
    for _, row in nodes_df.iterrows():
        node_id = _nn(row['id'])
        pos[node_id] = (row['x'], row['y'])
    return pos

# Animation function with enhanced features
def animate_routes(df_v, edges,
                   setTransport, setWaiting, setService,
                   nodes_df,
                   FRAMES_PER_UNIT=12,
                   SLOW_FACTOR=1.0):

    COLOR_VEHICLE = '#2ecc71'   # green
    COLOR_CARRY   = '#f1c40f'   # yellow
    COLOR_PAX     = '#e74c3c'   # red
    COLOR_ROUTE   = '#3498db'   # blue
    COLOR_WAIT    = '#9b59b6'   # purple

    pos = create_layout_from_coords(nodes_df)
    xs = nodes_df['x'].to_numpy()
    ys = nodes_df['y'].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('Longitude (x)')
    ax.set_ylabel('Latitude (y)')

    # Try to use global edges_df if present (OSM edges)
    try:
        df_edges = edges_df if 'edges_df' in globals() else None
    except Exception:
        df_edges = None

    id_to_xy = {row['id']: (row['x'], row['y']) for _, row in nodes_df.iterrows()}

    # Draw edges
    if df_edges is not None:
        for _, r in df_edges.iterrows():
            u, v = _nn(r['from_node']), _nn(r['to_node'])
            if u in id_to_xy and v in id_to_xy:
                x0, y0 = id_to_xy[u]
                x1, y1 = id_to_xy[v]
                ax.plot([x0, x1], [y0, y1], color='lightgray', linewidth=0.5, alpha=0.6, zorder=1)
    else:
        for (u, v) in edges:
            u, v = _nn(u), _nn(v)
            if u in pos and v in pos:
                (x0, y0), (x1, y1) = pos[u], pos[v]
                ax.plot([x0, x1], [y0, y1], color='lightgray', linewidth=0.5, alpha=0.6, zorder=1)

    # Draw nodes
    ax.scatter(xs, ys, s=5, c='black', alpha=0.7, zorder=2)

    # Segments/events
    segs_v, ev_v = build_segments(df_v, setTransport, setWaiting, setService, vehicle='v')

    # Service events (integer times)
    service_events = defaultdict(list)
    for tt_, kind, pl in ev_v:
        service_events[int(tt_)].append((kind, pl, pl['veh']))

    # Frames and radius
    T_max = 0
    for seg in segs_v:
        T_max = max(T_max, seg['t1'])
    total_frames = max(1, T_max * FRAMES_PER_UNIT)
    span = max(xs.max() - xs.min(), ys.max() - ys.min())
    RADIUS = (span * 0.01) if span > 0 else 0.001

    # Initial positions
    s_node = initial_node(df_v)
    if s_node in pos:
        x0, y0 = pos[s_node]
    else:
        x0, y0 = xs.mean(), ys.mean()

    # Create vehicle and passenger
    veh = Circle((x0, y0), RADIUS, facecolor=COLOR_VEHICLE, edgecolor='none', zorder=5)
    ax.add_patch(veh)

    both = pd.concat([df_v], ignore_index=True)
    p_pick = pickup_node_from_df(both, 'p')
    if p_pick in pos:
        px, py = pos[p_pick]
    else:
        px, py = x0, y0
    pax = Circle((px, py), RADIUS, facecolor=COLOR_PAX, edgecolor='none', zorder=4)
    ax.add_patch(pax)
    pax_label = ax.text(px, py, 'P', fontsize=8, color='white', weight='bold',
                        ha='center', va='center', zorder=6)

    # Route highlighting
    route_lines = []
    for seg in segs_v:
        if seg['i'] in pos and seg['j'] in pos:
            x1, y1 = pos[seg['i']]
            x2, y2 = pos[seg['j']]
            color = COLOR_WAIT if seg['type'] == 'wait' else COLOR_ROUTE
            line, = ax.plot([x1, x2], [y1, y2], color=color, linewidth=2,
                            alpha=0.3, zorder=3,
                            linestyle='--' if seg['type'] == 'wait' else '-')
            route_lines.append(line)

    # Info overlays
    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    pax_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Instructions
    ax.text(0.02, 0.02,
            'SPACE: pause/resume\nR: reset\n+: faster\n-: slower',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Speed indicator
    speed_text = ax.text(0.98, 0.02, f'Speed: {1/max(SLOW_FACTOR, 1e-6):.1f}x',
                         transform=ax.transAxes, fontsize=9, ha='right',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_VEHICLE, label='Empty Vehicle'),
        Patch(facecolor=COLOR_CARRY,   label='Carrying Passenger'),
        Patch(facecolor=COLOR_PAX,     label='Passenger'),
        Patch(facecolor=COLOR_ROUTE,   label='Travel Route'),
        Patch(facecolor=COLOR_WAIT,    label='Waiting')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    riding = {'v': set()}
    current_segment = None
    animation_paused = False
    frame_idx = 0
    current_slow_factor = SLOW_FACTOR
    base_interval = 100  # ms

    def update_colors():
        veh.set_facecolor(COLOR_CARRY if riding['v'] else COLOR_VEHICLE)

    def seg_at_time(segs, t):
        for s in segs:
            if s['t0'] <= t < s['t1']:
                return s
        return None

    def get_segment_info(seg, t_cont):
        if not seg:
            return "Waiting at node", "-"
        seg_type = "Moving" if seg['type'] == 'move' else "Waiting"
        from_to = f"{seg['i']} → {seg['j']}"
        return seg_type, from_to

    def update_info_display(t_cont, seg, riding_passengers):
        time_info = f"Time: {t_cont:.1f}"
        seg_type, seg_info = get_segment_info(seg, t_cont)
        passenger_status = "Carrying: " + (", ".join(riding_passengers) if riding_passengers else "None")
        info_text.set_text(f"{time_info}\n{seg_type}: {seg_info}")
        pax_text.set_text(f"Passenger: {passenger_status}")

    def update(frame):
        nonlocal frame_idx, current_segment
        frame_idx = frame

        if animation_paused:
            return veh, pax, pax_label, info_text, pax_text, speed_text

        t_cont = frame_idx / FRAMES_PER_UNIT
        t_int = int(round(t_cont))

        # Handle service events
        if (frame_idx % FRAMES_PER_UNIT) == 0:
            for kind, pl, veh_id in service_events.get(t_int, []):
                if kind == 'pickup':
                    if pl['p'] == 'p':
                        pax.set_visible(False)
                        pax_label.set_visible(False)
                    riding[veh_id].add(pl['p'])
                    update_colors()
                elif kind == 'dropoff':
                    node = pl['node']
                    if node in pos:
                        pax.center = pos[node]
                        pax.set_visible(True)
                        pax_label.set_visible(True)
                        pax_label.set_position(pos[node])
                    if 'p' in riding[veh_id]:
                        riding[veh_id].discard('p')
                    update_colors()

        # Update vehicle position
        current_segment = seg_at_time(segs_v, t_cont)
        if current_segment is not None:
            (x_from, y_from) = pos[current_segment['i']]
            (x_to, y_to) = pos[current_segment['j']]
            if current_segment['type'] == 'move' and current_segment['t1'] > current_segment['t0']:
                alpha = (t_cont - current_segment['t0']) / (current_segment['t1'] - current_segment['t0'])
            else:
                alpha = 0.0
            veh.center = (x_from + (x_to - x_from) * alpha,
                          y_from + (y_to - y_from) * alpha)

        update_info_display(t_cont, current_segment, riding['v'])
        return veh, pax, pax_label, info_text, pax_text, speed_text

    def on_key_press(event):
        nonlocal animation_paused, current_slow_factor, frame_idx
        if event.key == ' ':
            animation_paused = not animation_paused
            if animation_paused:
                ani.event_source.stop()
            else:
                ani.event_source.start()
        elif event.key == 'r':
            frame_idx = 0
            if animation_paused:
                animation_paused = False
                ani.event_source.start()
        elif event.key == '+':
            current_slow_factor = max(0.05, current_slow_factor * 0.7)
            ani.event_source.interval = max(
                1,
                int(base_interval * current_slow_factor / FRAMES_PER_UNIT)
            )
            speed_text.set_text(f'Speed: {1/current_slow_factor:.1f}x')
        elif event.key == '-':
            current_slow_factor = min(5.0, current_slow_factor * 1.3)
            ani.event_source.interval = max(
                1,
                int(base_interval * current_slow_factor / FRAMES_PER_UNIT)
            )
            speed_text.set_text(f'Speed: {1/current_slow_factor:.1f}x')

    interval_ms = max(1, int(base_interval * current_slow_factor / FRAMES_PER_UNIT))
    ani = FuncAnimation(fig, update, frames=total_frames + 1,
                        interval=interval_ms, blit=False, repeat=False)

    fig.canvas.mpl_connect('key_press_event', on_key_press)

    plt.title('VRPPDTW Animation (DP-based solution)')
    plt.tight_layout()
    plt.show()

    return ani

# -------- run the animation UI if solved --------
if solved:
    key_nodes = {
        'START': START,
        'PICKUP': PICKUP,
        'DROPOFF': DROPOFF,
    }
    animation = animate_routes(df_solution_v, edges,
                               setTransport, setWaiting, setService,
                               nodes_df=nodes_df,
                               FRAMES_PER_UNIT=12, SLOW_FACTOR=0.1)
