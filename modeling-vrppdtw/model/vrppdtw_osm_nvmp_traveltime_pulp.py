import json
import os
import pulp
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
' vrp with n vehicle and m passengers, n_v and n_p configurable via json file'
' network generated from osm'
' starting and service points configurable via json file'
' minimizing the total travel time while solving the optimization problem'

# ---------- Helpers ----------
def norm_node(x):
    """Return node id as a clean string without trailing .0 etc."""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

# =================== Define Road Network ===================
print("Begin building net work...")
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
    exit()

# ============= Load VRP configuration and visualize nodes ===============
print("Begin loading vrp_config.json...")
config_path = os.path.join(current_dir, 'vrp_config.json')

def load_vrp_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"vrp_config.json not found at {path}")
    with open(path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)

    def _require_positive(value, label):
        if value is None:
            raise ValueError(f"Config missing '{label}'")
        value = int(value)
        if value <= 0:
            raise ValueError(f"{label} must be positive")
        return value

    n = _require_positive(data.get('n'), 'n')
    m = _require_positive(data.get('m'), 'm')

    vehicles_cfg = data.get('vehicles', [])
    passengers_cfg = data.get('passengers', [])
    if len(vehicles_cfg) != n:
        raise ValueError(f"Expected {n} vehicles entries but found {len(vehicles_cfg)}")
    if len(passengers_cfg) != m:
        raise ValueError(f"Expected {m} passengers entries but found {len(passengers_cfg)}")

    vehicles = []
    for idx, veh in enumerate(vehicles_cfg):
        vid = str(veh.get('id', idx + 1))
        if 'start_node' not in veh:
            raise ValueError(f"Vehicle {vid} missing 'start_node'")
        vehicles.append({
            'id': vid,
            'start_node': norm_node(veh['start_node'])
        })

    passengers = []
    for idx, pax in enumerate(passengers_cfg):
        pid = str(pax.get('id', idx + 1))
        if 'pickup_node' not in pax or 'dropoff_node' not in pax:
            raise ValueError(f"Passenger {pid} missing pickup/dropoff node")
        passengers.append({
            'id': pid,
            'pickup_node': norm_node(pax['pickup_node']),
            'dropoff_node': norm_node(pax['dropoff_node']),
            'state': f"p{pid}"
        })

    return {
        'scenario_name': data.get('scenario_name', 'vrp_scenario'),
        'n': n,
        'm': m,
        'vehicles': vehicles,
        'passengers': passengers,
    }

def show_vrp_config(nodes_df, edges_df, vehicles, passengers):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"Configured VRP - {len(vehicles)} vehicles / {len(passengers)} passengers", fontsize=11)

    xs = nodes_df['x'].to_numpy()
    ys = nodes_df['y'].to_numpy()
    ax.scatter(xs, ys, s=5, c='black', alpha=0.7, zorder=1)
    ax.set_xlabel('Longitude (x)')
    ax.set_ylabel('Latitude (y)')
    ax.set_aspect('equal', adjustable='datalim')

    id_to_xy = {row['id']: (row['x'], row['y']) for _, row in nodes_df.iterrows()}

    if edges_df is not None:
        for _, r in edges_df.iterrows():
            u, v = r['from_node'], r['to_node']
            if u in id_to_xy and v in id_to_xy:
                (x0, y0), (x1, y1) = id_to_xy[u], id_to_xy[v]
                ax.plot([x0, x1], [y0, y1], color='lightgray', linewidth=0.5, alpha=0.5, zorder=0)

    def annotate_node(node_id, label, color, marker='o'):
        if node_id not in id_to_xy:
            print(f"[Config Map] Node {node_id} not found in nodes dataframe.")
            return
        x, y = id_to_xy[node_id]
        ax.scatter([x], [y], s=70, marker=marker, c=color, edgecolors='black', linewidths=0.5, zorder=5)
        ax.annotate(label, (x, y), textcoords='offset points', xytext=(0, 8),
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, linewidth=0.5))

    for veh in vehicles:
        annotate_node(veh['start_node'], f"V{veh['id']} start", '#2ecc71', marker='s')
    for pax in passengers:
        annotate_node(pax['pickup_node'], f"P{pax['id']}", '#e67e22')
        annotate_node(pax['dropoff_node'], f"D{pax['id']}", '#3498db')

    handles = []
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ecc71', markeredgecolor='black', label='Vehicle start', markersize=8))
    handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#e67e22', markeredgecolor='black', label='Pickup', markersize=8))
    handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markeredgecolor='black', label='Dropoff', markersize=8))
    ax.legend(handles=handles, loc='upper right')
    plt.tight_layout()
    plt.show()

try:
    config = load_vrp_config(config_path)
    vehicles_cfg = config['vehicles']
    passengers_cfg = config['passengers']
    show_vrp_config(nodes_df, edges_df, vehicles_cfg, passengers_cfg)
    print(f"Loaded config with {config['n']} vehicles and {config['m']} passengers.")
    valid_node_ids = set(nodes_df['id'].tolist())
    missing_vehicle_nodes = [veh['start_node'] for veh in vehicles_cfg if veh['start_node'] not in valid_node_ids]
    if missing_vehicle_nodes:
        raise ValueError(f"Vehicle start nodes not found in network: {missing_vehicle_nodes}")
    missing_pickups = [p['pickup_node'] for p in passengers_cfg if p['pickup_node'] not in valid_node_ids]
    missing_dropoffs = [p['dropoff_node'] for p in passengers_cfg if p['dropoff_node'] not in valid_node_ids]
    if missing_pickups or missing_dropoffs:
        raise ValueError("Some pickup/dropoff nodes are not present in the network data.")
except Exception as e:
    messagebox.showerror("Error while loading configuration", str(e))
    exit()


# ====================== TIME HORIZON ESTIMATION ======================
    print("Begin estimating time horizon...")
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

# Compute dynamic time horizon T based on selected points and network
adj = _build_adj(travel_times)

def all_config_nodes(vehicles, passengers):
    nodes = set()
    nodes.update(v['start_node'] for v in vehicles)
    for pax in passengers:
        nodes.add(pax['pickup_node'])
        nodes.add(pax['dropoff_node'])
    return sorted(nodes)

key_nodes_list = all_config_nodes(vehicles_cfg, passengers_cfg)

try:
    if len(key_nodes_list) < 2:
        raise ValueError("Need at least two key nodes to estimate horizon")
    max_pair_time = 0
    for a, b in product(key_nodes_list, key_nodes_list):
        if a == b:
            continue
        d = _shortest_time(a, b, adj)
        if not np.isfinite(d):
            raise ValueError(f"No path between {a} and {b}")
        max_pair_time = max(max_pair_time, int(d))
    BUFFER_RATIO = 0.5  # more conservative buffer
    BUFFER_STATIC = 600
    seq_factor = max(1, len(passengers_cfg))
    T_base = max_pair_time * seq_factor
    try:
        TIME_STEP = int(os.getenv('VRP_TIME_STEP', '10'))
    except Exception:
        TIME_STEP = 10
    if TIME_STEP <= 0:
        TIME_STEP = 10
    T_seconds = int(T_base * (1 + BUFFER_RATIO)) + BUFFER_STATIC
    # Allow explicit override for experimentation
    try:
        override_T = os.getenv('VRP_T_SECONDS')
        if override_T:
            override_val = int(override_T)
            if override_val > 0:
                T_seconds = override_val
                print(f"[Horizon] Overridden via VRP_T_SECONDS={override_val}")
    except Exception:
        pass
    T_steps = int(math.ceil(T_seconds / TIME_STEP))
    times = range(0, T_steps + 1)
    print(f"Dynamic time horizon set to {T_seconds}s with step {TIME_STEP}s -> {T_steps} steps")
except Exception as e:
    print(f"Time horizon estimation failed: {e}. Falling back to default T=3600s, step=10s")
    TIME_STEP = 10
    T_seconds = 3600
    T_steps = int(math.ceil(3600 / TIME_STEP))
    times = range(0, T_steps + 1)

passenger_states = [p['state'] for p in passengers_cfg]
states = ['_'] + passenger_states
state_to_passenger = {p['state']: p for p in passengers_cfg}
vehicle_ids = [veh['id'] for veh in vehicles_cfg]
vehicle_start_nodes = {veh['id']: veh['start_node'] for veh in vehicles_cfg}


# ====================== NETWORK REDUCTION ======================
# To speed up modeling massively on large OSM graphs, reduce the network
# to the union of shortest paths between the key points (starts, pickups, dropoffs).
print("Begin reducing network...")

# Keep originals in case reduction fails
orig_nodes = list(nodes)
orig_edges = set(edges)
orig_travel_times = dict(travel_times)

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

reduced_nodes = set()
reduced_edges = set()
failed_pairs = []

for a, b in product(key_nodes_list, key_nodes_list):
    if a == b:
        continue
    try:
        path = _shortest_path(a, b, adj)
    except Exception:
        failed_pairs.append((a, b))
        continue
    reduced_nodes.update(path)
    for uu, vv in zip(path[:-1], path[1:]):
        reduced_edges.add((uu, vv))

reduced_nodes.update(key_nodes_list)

if failed_pairs:
    print(f"[Reduction] Skipped {len(failed_pairs)} unreachable key pairs; using full graph instead.")
    nodes, edges, travel_times = orig_nodes, orig_edges, orig_travel_times
elif reduced_edges:
    nodes = sorted(set(norm_node(n) for n in reduced_nodes))
    edges = set((norm_node(i), norm_node(j)) for (i, j) in reduced_edges if (i, j) in travel_times)
    travel_times = {(norm_node(i), norm_node(j)): int(travel_times[(i, j)]) for (i, j) in reduced_edges if (i, j) in travel_times}
else:
    print("[Reduction] No reduced edges found; using full graph.")
    nodes, edges, travel_times = orig_nodes, orig_edges, orig_travel_times

print(f"Reduced network: {len(nodes)} nodes, {len(edges)} edges (from union of shortest paths)")

# Rebuild adjacency on the graph actually used and sanity-check reachability
adj = _build_adj(travel_times)
def _check_reachability(all_nodes, start_nodes, targets, adj_map, horizon_seconds=None):
    for s in start_nodes:
        for t in targets:
            d = _shortest_time(s, t, adj_map)
            if not np.isfinite(d):
                raise ValueError(f"Unreachable in reduced graph: {s} -> {t}")
            if horizon_seconds is not None and d > horizon_seconds:
                raise ValueError(f"Path {s}->{t} ({d}s) exceeds time horizon {horizon_seconds}s")

try:
    vehicle_starts = [v['start_node'] for v in vehicles_cfg]
    target_nodes = [p['pickup_node'] for p in passengers_cfg] + [p['dropoff_node'] for p in passengers_cfg]
    _check_reachability(nodes, vehicle_starts, target_nodes, adj, horizon_seconds=T_seconds)
except Exception as e:
    print(f"[Reachability] {e}. Using full graph as fallback.")
    nodes, edges, travel_times = orig_nodes, orig_edges, orig_travel_times
    adj = _build_adj(travel_times)

# ====================== VRP Model with State-Space-Time Network ======================
print("Begin building VRP model...")
# Generate all possible 3-dimensional vertices (i, t, w) over PHYSICAL nodes only
vertexs = [(i, t, w) for i in nodes for t in times for w in states]

# Transport arcs
arcsTransport = []
for (i, j) in edges:
    for t in times:
        required_steps = int(math.ceil(travel_times[(i, j)] / TIME_STEP))
        s = t + required_steps
        if s <= max(times):
            for w in states:
                arcsTransport.append((i, j, t, s, w, w))

# Service arcs (Pickup/Dropoff)
arcsService = []
pickup_arcs_by_state = defaultdict(list)
dropoff_arcs_by_state = defaultdict(list)
for pax in passengers_cfg:
    state = pax['state']
    pick_node = pax['pickup_node']
    drop_node = pax['dropoff_node']
    pickup_arcs = [(pick_node, pick_node, t, t, '_', state) for t in times]
    dropoff_arcs = [(drop_node, drop_node, t, t, state, '_') for t in times]
    arcsService.extend(pickup_arcs)
    arcsService.extend(dropoff_arcs)
    pickup_arcs_by_state[state].extend(pickup_arcs)
    dropoff_arcs_by_state[state].extend(dropoff_arcs)

# Waiting arcs
arcsWaiting = [(i, i, t, t + 1, w, w)
               for i in nodes
               for t in range(0, max(times))  # last waiting until T-1 -> T
               for w in states]

# --- Super-sink for open routes ---
OMEGA = 'Omega'  # sink node label (not in 'nodes')

def endPenalty(i, t):
    # 0.0 => truly "stop anywhere". Customize to add repositioning cost if desired.
    return 0.0

# End arcs: from ANY (i, t, '_') to Omega, zero-time arc
arcsEnd = [(i, OMEGA, t, t, '_', '_') for i in nodes for t in times]

# Summarize all arcs
arcsSTS = arcsTransport + arcsService + arcsWaiting + arcsEnd

# Build in/out arc indices for faster constraint construction
from collections import defaultdict
out_arcs = defaultdict(list)
in_arcs  = defaultdict(list)
for arc in arcsSTS:
    out_arcs[(arc[0], arc[2], arc[4])].append(arc)
    in_arcs[(arc[1], arc[3], arc[5])].append(arc)

# Progress info: sizes and rough constraint counts
print("Model size summary:")
print(f"- Nodes: {len(nodes)}; Edges: {len(edges)}; Time steps: {len(times)}; States: {len(states)}")
print(f"- Arcs transport: {len(arcsTransport)}; waiting: {len(arcsWaiting)}; service: {len(arcsService)}; end: {len(arcsEnd)}")
print(f"- Total arcs (per-vehicle binaries): {len(arcsSTS)} (x{len(vehicle_ids)} vehicles => {len(vehicle_ids) * len(arcsSTS)} binaries)")
approx_flow_cons = len(nodes) * len(times) * len(states) * len(vehicle_ids)
print(f"- Approx. flow conservation constraints: ~{approx_flow_cons}")

# Fast arc-type membership sets
setTransport = set(arcsTransport)
setWaiting   = set(arcsWaiting)
setService   = set(arcsService)
setEnd       = set(arcsEnd)

# Travel time (s - t)
tt = {}

for arc in arcsSTS:
    if arc in setTransport:
        i, j = arc[0], arc[1]
        tt[arc] = travel_times[(i, j)]
    elif arc in setWaiting:
        tt[arc] = (arc[3] - arc[2]) * TIME_STEP  # convert steps to seconds
    elif arc in setService:
        tt[arc] = 0
    else:  # end arcs to Omega
        i, t = arc[0], arc[2]
        tt[arc] = 0

# Initialize model
model = pulp.LpProblem("VRP_OpenRoute", pulp.LpMinimize)
y_vars = {
    vid: pulp.LpVariable.dicts(f"y_{vid}", arcsSTS, cat='Binary')
    for vid in vehicle_ids
}

# Set objective function for minimizing travel time
model += pulp.lpSum(
    tt[arc] * y_vars[vid][arc]
    for vid in vehicle_ids
    for arc in arcsSTS
), "total_travel_time"

# Start constraints per vehicle
for veh in vehicles_cfg:
    vid = veh['id']
    start_vertex = (veh['start_node'], 0, '_')
    outgoing = [arc for arc in out_arcs.get(start_vertex, []) if arc[5] == '_']
    if not outgoing:
        raise ValueError(f"No feasible outgoing arcs from start vertex {start_vertex} for vehicle {vid}")
    model += pulp.lpSum(y_vars[vid][arc] for arc in outgoing) == 1, f"vehicle_{vid}_start"

# Open-route end constraints (exactly one end arc to Omega per vehicle)
for vid in vehicle_ids:
    model += pulp.lpSum(y_vars[vid][arc] for arc in arcsEnd) == 1, f"vehicle_{vid}_end"

# Flow balance (skip only the respective start vertices) using pre-indexed arcs + progress
print("Assembling flow conservation constraints...")
total_vertices = len(vertexs) * len(vehicle_ids)
done = 0
# Progress interval can be tuned via env var
try:
    PROG_INT = int(os.getenv('VRP_PROGRESS_INTERVAL', '5000'))
except Exception:
    PROG_INT = 5000
if PROG_INT <= 0:
    PROG_INT = 5000

start_constraints_ts = time.time()
for vid in vehicle_ids:
    start_vertex_key = (vehicle_start_nodes[vid], 0, '_')
    for vertex in vertexs:
        i, t, w = vertex
        if vertex == start_vertex_key:
            continue
        inflow_v = pulp.lpSum(y_vars[vid][arc] for arc in in_arcs.get((i, t, w), []))
        outflow_v = pulp.lpSum(y_vars[vid][arc] for arc in out_arcs.get((i, t, w), []))
        model += (inflow_v - outflow_v) == 0, f"flow_balance_vehicle_{vid}_vertex_{vertex}"
        done += 1
        if (done % PROG_INT) == 0:
            elapsed = max(1e-6, time.time() - start_constraints_ts)
            rate = done / elapsed
            remain = max(0, total_vertices - done)
            eta = remain / rate if rate > 0 else float('inf')
            pct = int(100 * done / max(1, total_vertices))
            print(f" - vehicle flow: {done}/{total_vertices} (~{pct}%) | elapsed {elapsed:.1f}s | ETA ~{eta:.1f}s")

flow_elapsed = time.time() - start_constraints_ts
print(f"Flow constraints assembled in {flow_elapsed:.1f}s.")

# Pickup and drop-off exactly once per passenger
for pax in passengers_cfg:
    state = pax['state']
    pickups = pickup_arcs_by_state.get(state, [])
    drops = dropoff_arcs_by_state.get(state, [])
    if not pickups or not drops:
        raise ValueError(f"No service arcs generated for passenger {pax['id']}")
    pickup_expr = pulp.lpSum(y_vars[vid][arc] for vid in vehicle_ids for arc in pickups)
    drop_expr = pulp.lpSum(y_vars[vid][arc] for vid in vehicle_ids for arc in drops)
    model += pickup_expr == 1, f"pickup_passenger_{pax['id']}"
    model += drop_expr == 1, f"dropoff_passenger_{pax['id']}"

# Before solving
print("Starting optimization...")

# Solve the optimization problem with progress monitor
def solve_with_progress(model):
    # Prefer CBC solver with console messages; optional time limit from env
    try:
        limit_env = os.getenv('VRP_CBC_MAXSECONDS')
        max_seconds = int(limit_env) if limit_env else None
    except Exception:
        max_seconds = None

    # If no explicit limit, derive a heuristic soft limit from model size
    if max_seconds is None:
        try:
            num_bins = len(vehicle_ids) * len(arcsSTS)
            approx_cons = len(nodes) * len(times) * len(states) * len(vehicle_ids)
            # Heuristic: 1s per 15k binaries plus 1s per 50k flow constraints
            est = int(max(60, min(900, (num_bins / 15000.0) + (approx_cons / 50000.0)) * 60))
            max_seconds = est
            print(f"[Solver] No time limit set. Using heuristic soft limit: {max_seconds}s (binaries={num_bins}, flow_cons~{approx_cons})")
        except Exception:
            max_seconds = None

    # Optional relative gap target via env (best-effort: pass to CBC as option)
    cbc_opts = []
    try:
        gap_env = os.getenv('VRP_SOLVER_FRACGAP')
        if gap_env:
            # CBC uses 'ratio' for relative MIP gap termination
            cbc_opts += ["-ratio", str(float(gap_env))]
            print(f"[Solver] Target relative gap set to {gap_env}")
    except Exception:
        pass

    solver = None
    if hasattr(pulp, 'PULP_CBC_CMD'):
        try:
            solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=max_seconds, options=cbc_opts)
        except TypeError:
            # Older PuLP versions may not accept 'options'; fall back without it
            solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=max_seconds)

    done = threading.Event()
    result_container = {}

    def _run():
        try:
            res = model.solve(solver) if solver is not None else model.solve()
            result_container['res'] = res
        finally:
            done.set()

    t0 = time.time()
    thr = threading.Thread(target=_run, daemon=True)
    thr.start()

    # Simple console progress: elapsed time and optional ETA if max_seconds provided
    last_print = 0
    while not done.is_set():
        time.sleep(1)
        elapsed = int(time.time() - t0)
        if elapsed != last_print:
            if max_seconds:
                pct = min(100, int(100 * elapsed / max_seconds))
                eta = max(0, max_seconds - elapsed)
                print(f"[Solving] Elapsed: {elapsed}s, Limit: {max_seconds}s, ~{pct}% of limit, ETA ~{eta}s to soft limit")
            else:
                print(f"[Solving] Elapsed: {elapsed}s (no time limit set)")
            last_print = elapsed

    thr.join()
    total = int(time.time() - t0)
    print(f"Solve finished in {total}s. Status: {pulp.LpStatus.get(model.status, model.status)}")
    return result_container.get('res')

solve_with_progress(model)

# After solving
print(f"Optimization complete - Status: {pulp.LpStatus[model.status]}")
try:
    print("Objective value:", pulp.value(model.objective))
except Exception:
    pass

# ================ Results ================
print("Begin extracting results...")
def prettify_solution_df(df, OMEGA='Omega'):
    """
    - Replace 'Omega' in column 'j' with the corresponding physical node 'i'
      (so the last row shows the actual node where the vehicle ends).
    - Ensure 'i' and 'j' are strings without .0.
    """
    df = df.copy()
    # Replace Omega with i (end arcs are (i, 'Omega', t, t, '_', '_'))
    mask_end = df['j'] == OMEGA
    df.loc[mask_end, 'j'] = df.loc[mask_end, 'i']

    # Normalize display (no 1.0)
    df['i'] = df['i'].apply(norm_node)
    df['j'] = df['j'].apply(norm_node)

    return df

def print_sorted_solutions(df_by_vehicle):
    for vid, df_v in df_by_vehicle.items():
        if df_v.empty:
            print(f"\nVehicle {vid} Path: (no arcs selected)")
            continue
        dv = prettify_solution_df(df_v, OMEGA=OMEGA)
        dv['_i_num'] = pd.to_numeric(dv['i'], errors='coerce')
        dv['_j_num'] = pd.to_numeric(dv['j'], errors='coerce')
        dv = dv.sort_values(by=['t', 's', '_i_num', '_j_num', 'i', 'j'])
        print(f"\nVehicle {vid} Path (Sorted):")
        print(dv[['i', 'j', 't', 's', 'w', "w'"]].to_string(index=False))

vehicle_solutions = {}
if model.status == pulp.LpStatusOptimal:
    for vid in vehicle_ids:
        arcs_v = [arc for arc in arcsSTS if pulp.value(y_vars[vid][arc]) == 1]
        vehicle_solutions[vid] = pd.DataFrame(arcs_v, columns=["i", "j", "t", "s", "w", "w'"])
    print_sorted_solutions(vehicle_solutions)
else:
    print("No optimal solution found. Check time discretization, connectivity, and service feasibility.")


# ================== ANIMATION ==================
print("Begin preparing animation...")
def _nn(x):
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") else s

def classify_arc(row, setTransport, setWaiting, setService):
    tup = (row['i'], row['j'], row['t'], row['s'], row['w'], row["w'"])
    if tup in setTransport: return 'move'
    if tup in setWaiting:   return 'wait'
    if tup in setService:   return 'service'
    return 'other'

def build_segments(df, setTransport, setWaiting, setService, vehicle, state_to_passenger, OMEGA='Omega'):
    df = df.copy()
    df['i'] = df['i'].apply(_nn)
    df['j'] = df['j'].apply(_nn)
    df = df[df['j'] != OMEGA].copy()
    df = df.sort_values(['t','s','i','j'])

    segs = []
    events = []  # list of (time:int, kind:str, payload:dict)
    for _, r in df.iterrows():
        a_type = classify_arc(r, setTransport, setWaiting, setService)
        if a_type == 'service':
            if r['w'] == '_' and r["w'"] in state_to_passenger:
                pax_state = r["w'"]
                pax = state_to_passenger[pax_state]
                events.append((int(r['t']), 'pickup', {'state': pax_state, 'passenger': pax['id'], 'node': r['i'], 'veh': vehicle}))
            elif r['w'] in state_to_passenger and r["w'"] == '_':
                pax_state = r['w']
                pax = state_to_passenger[pax_state]
                events.append((int(r['t']), 'dropoff', {'state': pax_state, 'passenger': pax['id'], 'node': r['i'], 'veh': vehicle}))
        elif a_type in ('move','wait'):
            segs.append({'type': a_type, 'i': r['i'], 'j': r['j'], 't0': int(r['t']), 't1': int(r['s'])})
    return segs, events

def initial_node(df):
    if df.empty: return None
    r = df.sort_values(['t','s']).iloc[0]
    return _nn(r['i'])

def create_layout_from_coords(nodes_df):
    """Create position dictionary from actual x,y coordinates in nodes dataframe"""
    pos = {}
    for _, row in nodes_df.iterrows():
        node_id = _nn(row['id'])
        pos[node_id] = (row['x'], row['y'])
    return pos

# Animation function with enhanced features
def animate_routes(vehicle_paths, edges,
                    setTransport, setWaiting, setService,
                    nodes_df,
                    state_to_passenger,
                    OMEGA='Omega', FRAMES_PER_UNIT=12,
                    SLOW_FACTOR=1.0):
    if not vehicle_paths:
        print("No routes to animate.")
        return None

    COLOR_WAIT = '#9b59b6'
    COLOR_CARRY = '#f1c40f'
    pos = create_layout_from_coords(nodes_df)
    xs = nodes_df['x'].to_numpy()
    ys = nodes_df['y'].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('Longitude (x)')
    ax.set_ylabel('Latitude (y)')

    id_to_xy = {row['id']: (row['x'], row['y']) for _, row in nodes_df.iterrows()}
    try:
        df_edges = edges_df if 'edges_df' in globals() else None
    except Exception:
        df_edges = None

    if df_edges is not None:
        for _, r in df_edges.iterrows():
            u, v = _nn(r['from_node']), _nn(r['to_node'])
            if u in id_to_xy and v in id_to_xy:
                x0, y0 = id_to_xy[u]
                x1, y1 = id_to_xy[v]
                ax.plot([x0, x1], [y0, y1], color='lightgray', linewidth=0.4, alpha=0.5, zorder=1)
    else:
        for (u, v) in edges:
            u, v = _nn(u), _nn(v)
            if u in pos and v in pos:
                (x0, y0), (x1, y1) = pos[u], pos[v]
                ax.plot([x0, x1], [y0, y1], color='lightgray', linewidth=0.4, alpha=0.5, zorder=1)

    ax.scatter(xs, ys, s=5, c='black', alpha=0.6, zorder=2)

    vehicle_order = sorted(vehicle_paths.keys(), key=lambda vid: str(vid))
    cmap = plt.cm.get_cmap('tab10', max(1, len(vehicle_order)))
    vehicle_colors = {vid: cmap(idx % cmap.N) for idx, vid in enumerate(vehicle_order)}

    segs_by_vehicle = {}
    service_events = defaultdict(list)
    total_time = 0
    for vid in vehicle_order:
        df_v = vehicle_paths[vid]
        segs, events = build_segments(df_v, setTransport, setWaiting, setService,
                                      vehicle=vid, state_to_passenger=state_to_passenger, OMEGA=OMEGA)
        segs_by_vehicle[vid] = segs
        for seg in segs:
            total_time = max(total_time, seg['t1'])
        for tt, kind, payload in events:
            service_events[int(tt)].append((kind, payload))

    total_frames = max(1, int(max(1, total_time) * FRAMES_PER_UNIT))
    span = max(xs.max() - xs.min(), ys.max() - ys.min())
    RADIUS = (span * 0.01) if span > 0 else 0.001

    vehicle_artists = {}
    vehicle_labels = {}
    last_positions = {}
    riding = {vid: set() for vid in vehicle_order}
    current_segments = {vid: None for vid in vehicle_order}

    for vid in vehicle_order:
        df_v = vehicle_paths[vid]
        start_node = initial_node(df_v) or vehicle_start_nodes.get(vid)
        start_pos = pos.get(start_node, (xs.mean(), ys.mean()))
        circle = Circle(start_pos, RADIUS, facecolor=vehicle_colors[vid], edgecolor='none', zorder=5, alpha=0.9)
        ax.add_patch(circle)
        label = ax.text(start_pos[0], start_pos[1], f"V{vid}", fontsize=8, color='white', weight='bold',
                        ha='center', va='center', zorder=6,
                        bbox=dict(boxstyle='circle,pad=0.2', facecolor=vehicle_colors[vid], alpha=0.9))
        vehicle_artists[vid] = circle
        vehicle_labels[vid] = label
        last_positions[vid] = start_pos

    route_lines = []
    for vid in vehicle_order:
        color = vehicle_colors[vid]
        for seg in segs_by_vehicle[vid]:
            if seg['i'] in pos and seg['j'] in pos:
                x1, y1 = pos[seg['i']]
                x2, y2 = pos[seg['j']]
                draw_color = COLOR_WAIT if seg['type'] == 'wait' else color
                line, = ax.plot([x1, x2], [y1, y2], color=draw_color, linewidth=2,
                                alpha=0.25, zorder=3, linestyle='--' if seg['type'] == 'wait' else '-')
                route_lines.append(line)

    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    animation_paused = False
    current_slow_factor = SLOW_FACTOR
    frame_idx = 0
    speed_text = ax.text(0.98, 0.02, f'Speed: {1/current_slow_factor:.1f}x',
                         transform=ax.transAxes, fontsize=9, ha='right',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    def update_vehicle_style(vid):
        circle = vehicle_artists[vid]
        if riding[vid]:
            circle.set_linewidth(2)
            circle.set_edgecolor(COLOR_CARRY)
        else:
            circle.set_linewidth(0)
            circle.set_edgecolor('none')

    def update_info_display(t_cont):
        lines = [f"Time: {t_cont:.1f}"]
        for vid in vehicle_order:
            seg_type, seg_info = get_segment_info(current_segments.get(vid), t_cont)
            passengers = ", ".join(sorted(riding[vid])) if riding[vid] else "None"
            lines.append(f"V{vid}: {seg_type} ({seg_info}) | Carrying: {passengers}")
        info_text.set_text("\n".join(lines))

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
            ani.event_source.interval = max(1, int(100 * current_slow_factor / FRAMES_PER_UNIT))
            speed_text.set_text(f'Speed: {1/current_slow_factor:.1f}x')
        elif event.key == '-':
            current_slow_factor = min(5.0, current_slow_factor * 1.3)
            ani.event_source.interval = max(1, int(100 * current_slow_factor / FRAMES_PER_UNIT))
            speed_text.set_text(f'Speed: {1/current_slow_factor:.1f}x')

    fig.canvas.mpl_connect('key_press_event', on_key_press)

    def update(frame):
        nonlocal frame_idx
        frame_idx = frame
        if animation_paused:
            return list(vehicle_artists.values()) + list(vehicle_labels.values()) + [info_text, speed_text]

        t_cont = frame_idx / FRAMES_PER_UNIT
        t_int = int(round(t_cont))

        if (frame_idx % FRAMES_PER_UNIT) == 0:
            for kind, payload in service_events.get(t_int, []):
                vid = payload['veh']
                if kind == 'pickup':
                    riding[vid].add(str(payload['passenger']))
                    update_vehicle_style(vid)
                elif kind == 'dropoff':
                    riding[vid].discard(str(payload['passenger']))
                    update_vehicle_style(vid)

        for vid in vehicle_order:
            seg = seg_at_time(segs_by_vehicle[vid], t_cont)
            current_segments[vid] = seg
            circle = vehicle_artists[vid]
            if seg is not None and seg['i'] in pos and seg['j'] in pos:
                (x_from, y_from) = pos[seg['i']]
                (x_to, y_to) = pos[seg['j']]
                duration = max(1e-6, seg['t1'] - seg['t0'])
                alpha = min(max((t_cont - seg['t0']) / duration, 0.0), 1.0) if seg['type'] == 'move' else 0.0
                circle.center = (x_from + (x_to - x_from) * alpha,
                                 y_from + (y_to - y_from) * alpha)
                last_positions[vid] = circle.center
            else:
                circle.center = last_positions[vid]
            vehicle_labels[vid].set_position(circle.center)

        update_info_display(t_cont)
        return list(vehicle_artists.values()) + list(vehicle_labels.values()) + [info_text, speed_text]

    base_interval = 100
    interval_ms = max(1, int(base_interval * current_slow_factor / FRAMES_PER_UNIT))
    ani = FuncAnimation(fig, update, frames=total_frames + 1, interval=interval_ms, blit=False, repeat=False)

    ax.text(0.02, 0.02, 'SPACE: pause/resume\nR: reset\n+: faster\n-: slower',
            transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.title('VRPPDTW Animation - Multi-Vehicle')
    plt.tight_layout()
    plt.show()

    return ani

def seg_at_time(segs, t):
    for s in segs:
        if s['t0'] <= t < s['t1']:
            return s
    return None

def get_segment_info(seg, _t_cont):
    if not seg:
        return "Idle", "-"
    seg_type = "Moving" if seg['type'] == 'move' else "Waiting"
    return seg_type, f"{seg['i']} -> {seg['j']}"

# -------- run the animation UI if solved --------
if model.status == pulp.LpStatusOptimal:
    try:
        animate_routes  # type: ignore
    except NameError:
        pass
    else:
        animation = animate_routes(vehicle_solutions, edges,
                        setTransport, setWaiting, setService,
                        nodes_df=nodes_df,
                        state_to_passenger=state_to_passenger,
                        OMEGA=OMEGA, FRAMES_PER_UNIT=12, SLOW_FACTOR=0.1)
