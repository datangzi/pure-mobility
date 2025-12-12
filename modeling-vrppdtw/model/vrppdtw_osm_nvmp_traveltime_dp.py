import os
import json
import math
import heapq
import time
from collections import defaultdict
from itertools import product

import pandas as pd
import numpy as np
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle


# ================== Script Description ===================
"""
VRP with multiple vehicles and multiple passengers.
- Network generated from OSM (nodes_osm.csv, edges_osm.csv)
- Start nodes, pickup and dropoff nodes for vehicles/passengers are read from vrp_config.json
- Objective: minimize total travel time of all vehicles (heuristically) using DP-based routing
"""


# ---------- Helpers ----------
def norm_node(x):
    """Return node id as a clean string without trailing .0 etc."""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


# =================== Define Road Network ===================
current_dir = os.path.dirname(os.path.abspath(__file__))
nodes_path = os.path.join(current_dir, 'nodes_osm.csv')
edges_path = os.path.join(current_dir, 'edges_osm.csv')
config_path = os.path.join(current_dir, 'vrp_config.json')


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

        # Normalize street endpoints to strings
        edges_df['from_node'] = edges_df['from_node'].apply(norm_node)
        edges_df['to_node'] = edges_df['to_node'].apply(norm_node)

        # Times/costs as numbers
        edges_df['travel_time'] = edges_df['travel_time'].astype(float).astype(int)

        edges = set()
        travel_times = {}

        for _, row in edges_df.iterrows():
            i, j = row['from_node'], row['to_node']
            edges.add((i, j))
            travel_times[(i, j)] = int(row['travel_time'])

        return nodes, edges, travel_times, nodes_df, edges_df

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load network data: {str(e)}")
        raise


def load_vrp_config(config_path, nodes_df):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"VRP config file not found at {config_path}")

    with open(config_path, 'r', encoding='utf-8') as fh:
        cfg = json.load(fh)

    scenario_name = cfg.get('scenario_name', 'vrp_scenario')
    vehicles_raw = cfg.get('vehicles', [])
    passengers_raw = cfg.get('passengers', [])

    if not vehicles_raw or not passengers_raw:
        raise ValueError("Config must contain non-empty 'vehicles' and 'passengers' lists.")

    # Build set of valid node ids from nodes_df
    valid_ids = set(nodes_df['id'].astype(str).map(norm_node))

    vehicles = []
    for v in vehicles_raw:
        v_id = v['id']
        start_node = norm_node(v['start_node'])
        if start_node not in valid_ids:
            raise ValueError(f"Vehicle {v_id} has start_node {start_node} not in nodes_osm.csv")
        vehicles.append({'id': int(v_id), 'start_node': start_node})

    passengers = []
    for p in passengers_raw:
        p_id = p['id']
        pu = norm_node(p['pickup_node'])
        do = norm_node(p['dropoff_node'])
        if pu not in valid_ids:
            raise ValueError(f"Passenger {p_id} has pickup_node {pu} not in nodes_osm.csv")
        if do not in valid_ids:
            raise ValueError(f"Passenger {p_id} has dropoff_node {do} not in nodes_osm.csv")
        passengers.append({'id': int(p_id), 'pickup_node': pu, 'dropoff_node': do})

    # Optional consistency check
    n_cfg = cfg.get('n', None)
    m_cfg = cfg.get('m', None)
    if n_cfg is not None and n_cfg != len(vehicles):
        print(f"Warning: config n={n_cfg} but found {len(vehicles)} vehicles entries.")
    if m_cfg is not None and m_cfg != len(passengers):
        print(f"Warning: config m={m_cfg} but found {len(passengers)} passengers entries.")

    return scenario_name, vehicles, passengers


def show_config_map(nodes_df, edges_df, vehicles, passengers, scenario_name="vrp_scenario"):
    """Show OSM network with vehicle starts, pickups, and dropoffs highlighted."""
    xs = nodes_df['x'].to_numpy()
    ys = nodes_df['y'].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"VRP Scenario: {scenario_name}", fontsize=12)

    # Draw edges lightly
    id_to_xy = {row['id']: (row['x'], row['y']) for _, row in nodes_df.iterrows()}
    for _, r in edges_df.iterrows():
        u, v = norm_node(r['from_node']), norm_node(r['to_node'])
        if u in id_to_xy and v in id_to_xy:
            x0, y0 = id_to_xy[u]
            x1, y1 = id_to_xy[v]
            ax.plot([x0, x1], [y0, y1], color='lightgray', linewidth=0.5, alpha=0.6)

    # Draw all nodes faintly
    ax.scatter(xs, ys, s=5, c='black', alpha=0.3, zorder=1)

    # Highlight vehicle starts
    for v in vehicles:
        nid = v['start_node']
        if nid in id_to_xy:
            x, y = id_to_xy[nid]
            ax.scatter([x], [y], s=50, c='blue', marker='s', zorder=3)
            ax.text(x, y, f"V{v['id']}", fontsize=8, color='blue',
                    weight='bold', ha='left', va='bottom')

    # Highlight passenger pickups and dropoffs
    for p in passengers:
        pu = p['pickup_node']
        do = p['dropoff_node']
        if pu in id_to_xy:
            x, y = id_to_xy[pu]
            ax.scatter([x], [y], s=50, c='green', marker='^', zorder=3)
            ax.text(x, y, f"P{p['id']}+", fontsize=8, color='green',
                    weight='bold', ha='left', va='bottom')
        if do in id_to_xy:
            x, y = id_to_xy[do]
            ax.scatter([x], [y], s=50, c='red', marker='v', zorder=3)
            ax.text(x, y, f"P{p['id']}-", fontsize=8, color='red',
                    weight='bold', ha='left', va='bottom')

    ax.set_xlabel('Longitude (x)')
    ax.set_ylabel('Latitude (y)')
    ax.set_aspect('equal', adjustable='datalim')
    plt.tight_layout()
    plt.show()


# ====================== GRAPH UTILITIES ======================
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


# ====================== TIME HORIZON ESTIMATION ======================
def estimate_time_horizon(vehicles, passengers, adj):
    """
    Estimate a global time horizon that is large enough for
    multi-passenger, sequential service.

    1) max_travel = max over v,p of shortest_time(start_v -> pickup_p -> dropoff_p)
    2) approximate passengers per vehicle = ceil(m / n)
    3) horizon ≈ max_travel * (2 * approx_pass_per_vehicle) * 1.2 + buffer
       (the factor 2 allows for extra detours, 1.2 is an extra slack)
    """
    max_travel = 0
    for v in vehicles:
        s = v['start_node']
        for p in passengers:
            pu = p['pickup_node']
            do = p['dropoff_node']
            t1 = _shortest_time(s, pu, adj)
            t2 = _shortest_time(pu, do, adj)
            if not math.isfinite(t1) or not math.isfinite(t2):
                continue
            total = int(t1 + t2)
            if total > max_travel:
                max_travel = total

    if max_travel <= 0 or not math.isfinite(max_travel):
        print("Time horizon estimation failed or zero. Falling back to default 3600s.")
        T_seconds = 3600
    else:
        approx_pass_per_vehicle = max(1, math.ceil(len(passengers) / max(1, len(vehicles))))
        chain_factor = max(2, 2 * approx_pass_per_vehicle)
        T_seconds = int(max_travel * chain_factor * 1.2) + 300
        T_seconds = max(T_seconds, max_travel + 600)

        print(
            f"Max single-passenger route: {max_travel}s; "
            f"approx passengers/veh: {approx_pass_per_vehicle}; "
            f"chain_factor: {chain_factor}; "
            f"T_seconds: {T_seconds}"
        )

    try:
        TIME_STEP = int(os.getenv('VRP_TIME_STEP', '10'))
    except Exception:
        TIME_STEP = 10
    if TIME_STEP <= 0:
        TIME_STEP = 10

    T_steps = int(math.ceil(T_seconds / TIME_STEP))
    times = range(0, T_steps + 1)
    print(f"Dynamic time horizon set to {T_seconds}s with step {TIME_STEP}s -> {T_steps} steps")
    return TIME_STEP, T_seconds, T_steps, times


# ====================== NETWORK REDUCTION ======================
def reduce_network(nodes, edges, travel_times, vehicles, passengers):
    """Reduce network to union of shortest paths between all key nodes."""
    print("Reducing network to union of shortest paths between all vehicle starts and passenger stops...")
    adj_full = _build_adj(travel_times)

    key_nodes = set()
    for v in vehicles:
        key_nodes.add(v['start_node'])
    for p in passengers:
        key_nodes.add(p['pickup_node'])
        key_nodes.add(p['dropoff_node'])

    reduced_nodes = set()
    reduced_edges = set()

    key_nodes_list = list(key_nodes)
    for a, b in product(key_nodes_list, key_nodes_list):
        if a == b:
            continue
        try:
            path = _shortest_path(a, b, adj_full)
        except Exception:
            continue
        reduced_nodes.update(path)
        for uu, vv in zip(path[:-1], path[1:]):
            reduced_edges.add((uu, vv))

    if not reduced_nodes:
        print("Warning: network reduction found no paths; using full network.")
        return nodes, edges, travel_times

    nodes_red = sorted({norm_node(n) for n in reduced_nodes})
    edges_red = set()
    travel_red = {}
    for (i, j) in reduced_edges:
        if (i, j) in travel_times:
            ii = norm_node(i)
            jj = norm_node(j)
            edges_red.add((ii, jj))
            travel_red[(ii, jj)] = int(travel_times[(i, j)])

    print(f"Reduced network: {len(nodes_red)} nodes, {len(edges_red)} edges")
    return nodes_red, edges_red, travel_red


# ====================== STS BASE NETWORK ======================
def build_sts_base(nodes, edges, times, TIME_STEP, travel_times):
    """Build state-space-time base arcs (transport + waiting), shared by all passengers."""
    print("Building state-space-time (STS) base network...")
    # 0 = before pickup, 1 = carrying passenger, 2 = after dropoff
    states = ['0', '1', '2']
    base_arcs_transport = []
    base_arcs_waiting = []
    base_tt = {}
    adj_base = defaultdict(list)
    max_time_step = max(times)

    # Transport arcs
    for (i, j) in edges:
        tt_ij = int(travel_times[(i, j)])
        steps = int(math.ceil(tt_ij / TIME_STEP))
        for t in times:
            s = t + steps
            if s > max_time_step:
                continue
            for w in states:
                arc = (i, j, t, s, w, w)
                base_arcs_transport.append(arc)
                base_tt[arc] = tt_ij
                adj_base[(i, t, w)].append(((j, s, w), tt_ij, arc))

    # Waiting arcs
    for i in nodes:
        for t in range(0, max_time_step):
            for w in states:
                arc = (i, i, t, t + 1, w, w)
                base_arcs_waiting.append(arc)
                base_tt[arc] = TIME_STEP
                adj_base[(i, t, w)].append(((i, t + 1, w), TIME_STEP, arc))

    print(f"STS base network summary:")
    print(f"- States: {states}")
    print(f"- Time steps: {len(times)}")
    print(f"- Transport arcs: {len(base_arcs_transport)}")
    print(f"- Waiting arcs: {len(base_arcs_waiting)}")

    return states, base_arcs_transport, base_arcs_waiting, base_tt, adj_base


# ====================== SINGLE-PASSENGER DP SOLVER ======================
def dp_single_passenger(start_node, start_time,
                        pickup_node, dropoff_node,
                        states, times,
                        base_tt, adj_base,
                        TIME_STEP):
    """
    Solve 1-vehicle / 1-passenger VRPPDTW via shortest path on the STS network.
    States:
      '0' = before pickup
      '1' = carrying passenger
      '2' = after dropoff
    Start: (start_node, start_time, '0')
    End:   (dropoff_node, t, '2') for some t
    Returns:
        best_cost: total travel time
        arcs_solution: list of arcs (i, j, t, s, w, w')
        service_arcs: set of all service arcs generated in this DP
    """
    # Build service arcs and adjacency for them
    arcs_service = []
    service_arcs_out = defaultdict(list)
    dropoff_vertices = set()

    for t in times:
        if t < start_time:
            continue
        # pickup: 0 -> 1
        arc_pu = (pickup_node, pickup_node, t, t, '0', '1')
        arcs_service.append(arc_pu)
        service_arcs_out[(pickup_node, t, '0')].append(((pickup_node, t, '1'), 0.0, arc_pu))

        # dropoff: 1 -> 2
        arc_do = (dropoff_node, dropoff_node, t, t, '1', '2')
        arcs_service.append(arc_do)
        service_arcs_out[(dropoff_node, t, '1')].append(((dropoff_node, t, '2'), 0.0, arc_do))

        dropoff_vertices.add((dropoff_node, t, '2'))

    if not dropoff_vertices:
        raise RuntimeError("No dropoff vertices created – time horizon may be too short.")

    source = (start_node, start_time, '0')
    INF = float('inf')
    dist = defaultdict(lambda: INF)
    dist[source] = 0.0
    prev = {}

    pq = [(0.0, source)]
    best_target = None
    best_cost = INF

    # Dijkstra on STS network (base arcs + service arcs)
    while pq:
        d, v = heapq.heappop(pq)
        if d > dist[v] + 1e-9:
            continue

        if v in dropoff_vertices:
            best_target = v
            best_cost = d
            break

        # Base arcs
        for (v2, cost, arc) in adj_base.get(v, []):
            nd = d + cost
            if nd + 1e-9 < dist[v2]:
                dist[v2] = nd
                prev[v2] = (v, arc)
                heapq.heappush(pq, (nd, v2))

        # Service arcs
        for (v2, cost, arc) in service_arcs_out.get(v, []):
            nd = d + cost
            if nd + 1e-9 < dist[v2]:
                dist[v2] = nd
                prev[v2] = (v, arc)
                heapq.heappush(pq, (nd, v2))

    if best_target is None:
        return INF, [], set(arcs_service)

    # Reconstruct path from best_target back to source
    arcs_solution = []
    cur = best_target
    while cur != source:
        pv, arc = prev[cur]
        arcs_solution.append(arc)
        cur = pv
    arcs_solution.reverse()

    return best_cost, arcs_solution, set(arcs_service)


# ====================== MULTI-VEHICLE HEURISTIC SOLVER ======================
def solve_multi_vrp_dp(vehicles, passengers,
                       states, times,
                       base_tt, adj_base,
                       TIME_STEP):
    """
    Heuristic multi-vehicle, multi-passenger solver:
    1) For each (vehicle, passenger) pair, compute the cost of serving that passenger alone.
    2) Assign each passenger to the vehicle with minimal individual cost.
    3) For each vehicle, build a sequential route visiting its assigned passengers in ascending id order,
       using DP for each passenger in sequence.
    """
    print("Computing vehicle-passenger cost matrix via DP...")
    costs = {}
    for p in passengers:
        for v in vehicles:
            c, _, _ = dp_single_passenger(
                start_node=v['start_node'],
                start_time=0,
                pickup_node=p['pickup_node'],
                dropoff_node=p['dropoff_node'],
                states=states,
                times=times,
                base_tt=base_tt,
                adj_base=adj_base,
                TIME_STEP=TIME_STEP
            )
            costs[(v['id'], p['id'])] = c

    assignments = {v['id']: [] for v in vehicles}
    unserved_passengers = []

    for p in passengers:
        best_v_id = None
        best_cost = math.inf
        for v in vehicles:
            c = costs.get((v['id'], p['id']), math.inf)
            if c < best_cost:
                best_cost = c
                best_v_id = v['id']
        if best_v_id is None or not math.isfinite(best_cost):
            print(f"Passenger {p['id']} unreachable by all vehicles.")
            unserved_passengers.append(p['id'])
        else:
            assignments[best_v_id].append(p)

    vehicle_routes = {v['id']: [] for v in vehicles}
    service_arcs_global = set()
    total_travel_time = 0.0

    print("Building sequential routes per vehicle via DP...")
    for v in vehicles:
        v_id = v['id']
        current_node = v['start_node']
        current_time = 0
        assigned_ps = sorted(assignments[v_id], key=lambda p: p['id'])

        for p in assigned_ps:
            c_p, arcs_p, service_arcs_p = dp_single_passenger(
                start_node=current_node,
                start_time=current_time,
                pickup_node=p['pickup_node'],
                dropoff_node=p['dropoff_node'],
                states=states,
                times=times,
                base_tt=base_tt,
                adj_base=adj_base,
                TIME_STEP=TIME_STEP
            )
            if not math.isfinite(c_p) or not arcs_p:
                print(f"Warning: vehicle {v_id} cannot serve passenger {p['id']} from current state.")
                unserved_passengers.append(p['id'])
                continue

            total_travel_time += c_p
            service_arcs_global |= service_arcs_p

            for arc in arcs_p:
                i, j, t, s, w, wp = arc
                vehicle_routes[v_id].append((v_id, p['id'], i, j, t, s, w, wp))

            last_arc = arcs_p[-1]
            current_node = last_arc[1]
            current_time = last_arc[3]

    return vehicle_routes, total_travel_time, assignments, unserved_passengers, service_arcs_global


# ====================== RESULTS & PRINTING ======================
def prettify_solution_df(df):
    """Normalize ids for printing."""
    df = df.copy()
    df['i'] = df['i'].apply(norm_node)
    df['j'] = df['j'].apply(norm_node)
    return df


def print_routes(df_all):
    if df_all.empty:
        print("No routes found.")
        return

    for veh_id, df_v in df_all.groupby('veh'):
        dv = prettify_solution_df(df_v)
        dv['_i_num'] = pd.to_numeric(dv['i'], errors='coerce')
        dv['_j_num'] = pd.to_numeric(dv['j'], errors='coerce')
        dv = dv.sort_values(by=['t', 's', '_i_num', '_j_num', 'i', 'j'])

        print(f"\nVehicle {veh_id} route:")
        print(dv[['veh', 'p', 'i', 'j', 't', 's', 'w', "w'"]].to_string(index=False))


# ================== ANIMATION (MULTI-VEHICLE) ==================
def _nn(x):
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") else s


def create_layout_from_coords(nodes_df):
    """Create position dictionary from actual x,y coordinates in nodes dataframe."""
    pos = {}
    for _, row in nodes_df.iterrows():
        node_id = _nn(row['id'])
        pos[node_id] = (row['x'], row['y'])
    return pos


def animate_multi_vehicle_routes(df_all, edges_df,
                                 set_transport, set_waiting, set_service,
                                 nodes_df,
                                 vehicles, passengers,
                                 FRAMES_PER_UNIT=12,
                                 SLOW_FACTOR=0.05):
    if df_all.empty:
        print("No routes to animate.")
        return None

    pos = create_layout_from_coords(nodes_df)
    xs = nodes_df['x'].to_numpy()
    ys = nodes_df['y'].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('Longitude (x)')
    ax.set_ylabel('Latitude (y)')

    id_to_xy = {row['id']: (row['x'], row['y']) for _, row in nodes_df.iterrows()}

    # Prepare df_all
    df_all = df_all.copy()
    df_all['i'] = df_all['i'].apply(_nn)
    df_all['j'] = df_all['j'].apply(_nn)

    # --- Route edges (for highlighting) ---
    route_edges = set()
    for _, r in df_all.iterrows():
        arc = (r['i'], r['j'], r['t'], r['s'], r['w'], r["w'"])
        if arc in set_transport:
            route_edges.add((r['i'], r['j']))

    # --- Draw all original map edges (thin) ---
    for _, r in edges_df.iterrows():
        u, v = norm_node(r['from_node']), norm_node(r['to_node'])
        if u in id_to_xy and v in id_to_xy:
            x0, y0 = id_to_xy[u]
            x1, y1 = id_to_xy[v]
            ax.plot([x0, x1], [y0, y1], color='lightgray', linewidth=0.5, alpha=0.6, zorder=1)

    # --- Highlight route edges (thick dark gray) ---
    for (u, v) in route_edges:
        if u in id_to_xy and v in id_to_xy:
            x0, y0 = id_to_xy[u]
            x1, y1 = id_to_xy[v]
            ax.plot([x0, x1], [y0, y1],
                    color='0.4', linewidth=2.5, alpha=0.9, zorder=2)

    # Draw nodes faintly on top
    ax.scatter(xs, ys, s=5, c='black', alpha=0.3, zorder=1)

    # --- Build segments per vehicle, determine time horizon ---
    segs_per_vehicle = {}
    T_max = 0

    for veh_id, df_v in df_all.groupby('veh'):
        segs = []
        df_v = df_v.sort_values(['t', 's', 'i', 'j'])
        for _, r in df_v.iterrows():
            arc = (r['i'], r['j'], r['t'], r['s'], r['w'], r["w'"])
            if arc in set_transport:
                a_type = 'move'
            elif arc in set_waiting:
                a_type = 'wait'
            elif arc in set_service:
                a_type = 'service'
            else:
                continue

            if a_type in ('move', 'wait'):
                seg = {
                    'type': a_type,
                    'i': r['i'],
                    'j': r['j'],
                    't0': int(r['t']),
                    't1': int(r['s'])
                }
                segs.append(seg)
                T_max = max(T_max, seg['t1'])
        segs_per_vehicle[veh_id] = segs

    if T_max <= 0:
        print("No time span for animation.")
        return None

    total_frames = max(1, T_max * FRAMES_PER_UNIT)
    span = max(xs.max() - xs.min(), ys.max() - ys.min())
    RADIUS = (span * 0.01) if span > 0 else 0.001
    label_dy = span * 0.02 if span > 0 else 0.02

    # --- Passenger markers and events ---
    pax_nodes = {p['id']: (p['pickup_node'], p['dropoff_node']) for p in passengers}

    passenger_patches = {}
    passenger_labels = {}
    passenger_status = {}
    for p in passengers:
        p_id = p['id']
        pu, _ = pax_nodes[p_id]
        if pu in pos:
            x, y = pos[pu]
        else:
            x, y = xs.mean(), ys.mean()
        patch = Circle((x, y), RADIUS * 0.6,
                       facecolor='red', edgecolor='black', linewidth=0.5,
                       zorder=4)
        ax.add_patch(patch)
        passenger_patches[p_id] = patch
        passenger_status[p_id] = 'waiting'
        lbl = ax.text(x, y + RADIUS * 1.0, f"P{p_id}",
                      fontsize=8, color='black',
                      ha='center', va='bottom', zorder=5)
        passenger_labels[p_id] = lbl

    passenger_events = defaultdict(list)
    for _, r in df_all.iterrows():
        arc = (r['i'], r['j'], r['t'], r['s'], r['w'], r["w'"])
        if arc not in set_service:
            continue
        t = int(r['t'])
        veh_id = int(r['veh'])
        p_id = int(r['p'])
        node = r['i']
        if r['w'] == '0' and r["w'"] == '1':
            passenger_events[t].append(('pickup', veh_id, p_id, node))
        elif r['w'] == '1' and r["w'"] == '2':
            passenger_events[t].append(('dropoff', veh_id, p_id, node))

    # --- Vehicle patches & labels ---
    vehicle_patches = {}
    vehicle_labels = {}
    vehicle_occupancy = {v['id']: 0 for v in vehicles}

    def set_vehicle_color(veh_id):
        patch = vehicle_patches[veh_id]
        if vehicle_occupancy[veh_id] > 0:
            patch.set_facecolor('0.7')  # gray
        else:
            patch.set_facecolor('white')
        patch.set_edgecolor('black')
        patch.set_linewidth(1.0)

    for v in vehicles:
        veh_id = v['id']
        segs = segs_per_vehicle.get(veh_id, [])
        if segs:
            start_node = segs[0]['i']
        else:
            start_node = v['start_node']
        if start_node in pos:
            x0, y0 = pos[start_node]
        else:
            x0, y0 = xs.mean(), ys.mean()
        patch = Circle((x0, y0), RADIUS,
                       facecolor='white',
                       edgecolor='black',
                       linewidth=1.0,
                       zorder=5)
        ax.add_patch(patch)
        vehicle_patches[veh_id] = patch
        set_vehicle_color(veh_id)
        lbl = ax.text(x0, y0 + label_dy, f"{veh_id}",
                      fontsize=9, color='black',
                      ha='center', va='bottom', zorder=6)
        vehicle_labels[veh_id] = lbl

    # --- Info text ---
    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    frame_idx = 0

    def seg_at_time(segs, t):
        for s in segs:
            if s['t0'] <= t < s['t1']:
                return s
        return None

    def update(frame):
        nonlocal frame_idx
        frame_idx = frame
        t_cont = frame_idx / FRAMES_PER_UNIT
        t_int = int(round(t_cont))

        # Handle passenger events at integer times
        if (frame_idx % FRAMES_PER_UNIT) == 0:
            for kind, veh_id, p_id, node in passenger_events.get(t_int, []):
                if kind == 'pickup':
                    passenger_status[p_id] = 'onboard'
                    patch = passenger_patches.get(p_id)
                    lbl = passenger_labels.get(p_id)
                    if patch is not None:
                        patch.set_visible(False)
                    if lbl is not None:
                        lbl.set_visible(False)
                    vehicle_occupancy[veh_id] += 1
                    set_vehicle_color(veh_id)
                elif kind == 'dropoff':
                    passenger_status[p_id] = 'dropped'
                    pu, do = pax_nodes[p_id]
                    node_drop = do
                    if node_drop in pos:
                        x, y = pos[node_drop]
                    else:
                        x, y = xs.mean(), ys.mean()
                    patch = passenger_patches.get(p_id)
                    lbl = passenger_labels.get(p_id)
                    if patch is not None:
                        patch.center = (x, y)
                        patch.set_visible(True)
                    if lbl is not None:
                        lbl.set_position((x, y + RADIUS * 1.0))
                        lbl.set_visible(True)
                    vehicle_occupancy[veh_id] = max(0, vehicle_occupancy[veh_id] - 1)
                    set_vehicle_color(veh_id)

        # Move each vehicle
        for v in vehicles:
            veh_id = v['id']
            patch = vehicle_patches.get(veh_id)
            label = vehicle_labels.get(veh_id)
            if patch is None or label is None:
                continue
            segs = segs_per_vehicle.get(veh_id, [])
            seg = seg_at_time(segs, t_cont)
            if seg and seg['i'] in pos and seg['j'] in pos:
                (x_from, y_from) = pos[seg['i']]
                (x_to, y_to) = pos[seg['j']]
                if seg['type'] == 'move' and seg['t1'] > seg['t0']:
                    alpha = (t_cont - seg['t0']) / (seg['t1'] - seg['t0'])
                else:
                    alpha = 0.0
                x = x_from + (x_to - x_from) * alpha
                y = y_from + (y_to - y_from) * alpha
                patch.center = (x, y)
                label.set_position((x, y + label_dy))

        info_text.set_text(f"t = {t_int}")
        artists = (
            list(vehicle_patches.values())
            + list(passenger_patches.values())
            + list(passenger_labels.values())
            + list(vehicle_labels.values())
            + [info_text]
        )
        return artists

    base_interval = 100  # ms
    interval_ms = max(1, int(base_interval * SLOW_FACTOR / FRAMES_PER_UNIT))

    ani = FuncAnimation(fig, update, frames=total_frames + 1,
                        interval=interval_ms, blit=False, repeat=False)

    plt.title('Multi-vehicle VRP Simulation (DP-based heuristic)')
    plt.tight_layout()
    plt.show()

    return ani


# ====================== MAIN FLOW ======================
def main():
    try:
        nodes, edges, travel_times, nodes_df, edges_df = load_network_data()
        print("Network loaded successfully.")
    except Exception as e:
        print(f"Error: {str(e)}")
        input("Press Enter to exit...")
        return

    try:
        scenario_name, vehicles, passengers = load_vrp_config(config_path, nodes_df)
        print(f"Config loaded: scenario '{scenario_name}', "
              f"{len(vehicles)} vehicles, {len(passengers)} passengers.")
    except Exception as e:
        print(f"Error loading VRP config: {e}")
        input("Press Enter to exit...")
        return

    try:
        show_config_map(nodes_df, edges_df, vehicles, passengers, scenario_name)
    except Exception as e:
        print(f"Warning: failed to display config map: {e}")

    nodes_red, edges_red, travel_red = reduce_network(
        nodes, edges, travel_times, vehicles, passengers
    )
    nodes = nodes_red
    edges = edges_red
    travel_times = travel_red

    adj_reduced = _build_adj(travel_times)
    TIME_STEP, T_seconds, T_steps, times = estimate_time_horizon(
        vehicles, passengers, adj_reduced
    )

    (states,
     base_arcs_transport,
     base_arcs_waiting,
     base_tt,
     adj_base) = build_sts_base(nodes, edges, times, TIME_STEP, travel_times)

    set_transport = set(base_arcs_transport)
    set_waiting = set(base_arcs_waiting)

    start_solve = time.time()
    try:
        (vehicle_routes,
         total_travel_time,
         assignments,
         unserved_passengers,
         service_arcs_global) = solve_multi_vrp_dp(
            vehicles, passengers,
            states, times,
            base_tt, adj_base,
            TIME_STEP
        )
        solve_time = time.time() - start_solve
        print(f"\nHeuristic DP-based VRP solution complete in {solve_time:.1f}s.")
        print(f"Total travel time (sum over vehicles): {total_travel_time:.1f} seconds")
        if unserved_passengers:
            print(f"Unserved passengers: {sorted(set(unserved_passengers))}")
        else:
            print("All passengers assigned and routed (heuristically).")
    except Exception as e:
        print(f"Error during VRP solving: {e}")
        input("Press Enter to exit...")
        return

    rows = []
    for veh_id, arcs in vehicle_routes.items():
        for (veh, p_id, i, j, t, s, w, wp) in arcs:
            rows.append({
                'veh': veh,
                'p': p_id,
                'i': i,
                'j': j,
                't': t,
                's': s,
                'w': w,
                "w'": wp
            })
    if rows:
        df_solution_all = pd.DataFrame(rows)
    else:
        df_solution_all = pd.DataFrame(
            columns=['veh', 'p', 'i', 'j', 't', 's', 'w', "w'"]
        )

    set_service = service_arcs_global if 'service_arcs_global' in locals() else set()

    print_routes(df_solution_all)

    try:
        animate_multi_vehicle_routes(df_solution_all, edges_df,
                                     set_transport, set_waiting, set_service,
                                     nodes_df,
                                     vehicles, passengers,
                                     FRAMES_PER_UNIT=12,
                                     SLOW_FACTOR=0.05)
    except Exception as e:
        print(f"Warning: failed to animate routes: {e}")

    input("Press Enter to exit...")


if __name__ == '__main__':
    main()
