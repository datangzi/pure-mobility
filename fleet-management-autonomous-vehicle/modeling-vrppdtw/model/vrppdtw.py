import os
import pulp
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle

# =================== DESCRIPTION ===================
# Open-route VRPPDTW variant:
# - Node IDs are normalized to strings (no 1.0).
# - Vehicles may end ANYWHERE via a super-sink Ω (Omega).
# - Output replaces 'Omega' with the actual last physical node.
# - Travel times cast to integers (prevent float-time mismatches).
# - Set-based arc-type lookups for speed.
# - Exactly one pickup AND one dropoff per passenger.

# ---------- Helpers ----------
def norm_node(x):
    """Return node id as a clean string without trailing .0 etc."""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

# =================== ROAD NETWORK DEFINITION ===================
def load_network_data():
    if not os.path.exists('nodes.csv'):
        raise FileNotFoundError("nodes.csv not found")
    if not os.path.exists('streets.csv'):
        raise FileNotFoundError("streets.csv not found")
    try:
        # Read node data
        nodes_df = pd.read_csv('nodes.csv')
        if 'id' not in nodes_df.columns:
            raise ValueError("nodes.csv must contain an 'id' column")

        # Normalize node ids to strings
        nodes_df['id'] = nodes_df['id'].apply(norm_node)
        spaces = nodes_df['id'].tolist()

        # Read street data
        streets_df = pd.read_csv('streets.csv')
        required_cols = {'from_node', 'to_node', 'travel_time'}
        if not required_cols.issubset(streets_df.columns):
            miss = required_cols - set(streets_df.columns)
            raise ValueError(f"streets.csv missing columns: {miss}")

        # Normalize street endpoints to strings to match 'spaces'
        streets_df['from_node'] = streets_df['from_node'].apply(norm_node)
        streets_df['to_node']   = streets_df['to_node'].apply(norm_node)

        # Times/costs as numbers
        streets_df['travel_time'] = streets_df['travel_time'].astype(float).astype(int)
        if 'cost' not in streets_df.columns:
            streets_df['cost'] = 0.0
        else:
            streets_df['cost'] = streets_df['cost'].astype(float)

        streets = set()
        travel_times = {}
        costs = {}

        print("Script started")  # First line after imports

        for _, row in streets_df.iterrows():
            i, j = row['from_node'], row['to_node']  # strings
            streets.add((i, j))
            travel_times[(i, j)] = int(row['travel_time'])
            costs[(i, j)]        = float(row['cost'])

        return spaces, streets, travel_times, costs, nodes_df

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load network data: {str(e)}")
        raise

# Load network data
try:
    spaces, streets, travel_times, costs, nodes_df = load_network_data()
    print("Network loaded successfully:")
    print("Nodes:", spaces)
    print("Streets:", streets)
    print("Travel times:", list(travel_times.items()))
    print("Travel costs:", list(costs.items()))
except Exception as e:
    print(f"Error: {str(e)}")
    input("Press Enter to exit...")
    exit()

# Time parameters
times = range(0, 11)  # t=0 to t=10 inclusive
states = ['_', 'p1', 'p2']  # Passenger carrying state (capacity = 1)

# After loading network data
print(f"Network loaded - Nodes: {spaces}, Streets: {streets}")

# ============= STARTING POINTS DEFINITION ===============
def define_starting_point():
    global START_v1, START_v2
    try:
        START_v1 = norm_node(start_entry_v1.get())
        START_v2 = norm_node(start_entry_v2.get())
        if START_v1 not in spaces or START_v2 not in spaces:
            raise ValueError("Starting point must be a defined physical node")
        root.destroy()
    except ValueError as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

# Initialize GUI for starting points only
root = tk.Tk()
root.title("Vehicle Starting Point Selection")

tk.Label(root, text=f"Available nodes: {spaces}", font=('Arial', 12)).pack(pady=5)

tk.Label(root, text="Starting point of v1:", font=('Arial', 12)).pack(pady=5)
start_entry_v1 = tk.Entry(root, font=('Arial', 12))  # User defines starting point of v1
start_entry_v1.pack()

tk.Label(root, text="Starting point of v2:", font=('Arial', 12)).pack(pady=5)
start_entry_v2 = tk.Entry(root, font=('Arial', 12))  # User defines starting point of v2
start_entry_v2.pack()

tk.Button(root, text="Confirm", command=define_starting_point,
          font=('Arial', 12), bg='#4CAF50', fg='white').pack(pady=10)

root.mainloop()

# After starting point selection
print(f"Starting point of v1: {START_v1}, Starting point of v2: {START_v2}")

# ============= SERVICE POINTS DEFINITION ===============
def define_service_point():
    global PICKUP_p1, PICKUP_p2, DROPOFF_p1, DROPOFF_p2
    try:
        PICKUP_p1  = norm_node(pickup_entry_p1.get())
        DROPOFF_p1 = norm_node(dropoff_entry_p1.get())
        PICKUP_p2  = norm_node(pickup_entry_p2.get())
        DROPOFF_p2 = norm_node(dropoff_entry_p2.get())

        if (PICKUP_p1 not in spaces or DROPOFF_p1 not in spaces or
            PICKUP_p2 not in spaces or DROPOFF_p2 not in spaces):
            raise ValueError("Service point must be a defined physical node")
        root.destroy()
    except ValueError as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

# Initialize GUI for service points only
root = tk.Tk()
root.title("Service Point Selection")

tk.Label(root, text=f"Available nodes: {spaces}", font=('Arial', 12)).pack(pady=5)

tk.Label(root, text="Pickup point for p1:", font=('Arial', 12)).pack(pady=5)
pickup_entry_p1 = tk.Entry(root, font=('Arial', 12))  # User defines pickup for p1
pickup_entry_p1.pack()

tk.Label(root, text="Pickup point for p2:", font=('Arial', 12)).pack(pady=5)
pickup_entry_p2 = tk.Entry(root, font=('Arial', 12))  # User defines pickup for p2
pickup_entry_p2.pack()

tk.Label(root, text="Dropoff point p1:", font=('Arial', 12)).pack(pady=5)
dropoff_entry_p1 = tk.Entry(root, font=('Arial', 12))  # User defines dropoff for p1
dropoff_entry_p1.pack()

tk.Label(root, text="Dropoff point p2:", font=('Arial', 12)).pack(pady=5)
dropoff_entry_p2 = tk.Entry(root, font=('Arial', 12))  # User defines dropoff for p2
dropoff_entry_p2.pack()

tk.Button(root, text="Start Simulation", command=define_service_point,
          font=('Arial', 12), bg='#4CAF50', fg='white').pack(pady=10)

root.mainloop()

# After service point selection
print(f"Service points set - Pickup for p1: {PICKUP_p1}, Dropoff for p1: {DROPOFF_p1}, Pickup for p2: {PICKUP_p2}, Dropoff for p2: {DROPOFF_p2}")

# ====================== OPTIMIZATION MODEL ======================
# Generate all possible 3-dimensional vertices (i, t, w) over PHYSICAL nodes only
vertexs = [(i, t, w) for i in spaces for t in times for w in states]

# Generate transport arcs
arcsTransport = []
for (i, j) in streets:
    for t in times:
        required_time = int(travel_times[(i, j)])
        s = t + required_time
        if s <= max(times):
            for w in states:
                arcsTransport.append((i, j, t, s, w, w))

# Service arcs (dynamic based on GUI input) — pickup/dropoff at zero time
arcsService = (
    [(PICKUP_p1, PICKUP_p1, t, t, '_', 'p1') for t in times] +   # Pickup p1
    [(DROPOFF_p1, DROPOFF_p1, t, t, 'p1', '_') for t in times] + # Dropoff p1
    [(PICKUP_p2, PICKUP_p2, t, t, '_', 'p2') for t in times] +   # Pickup p2
    [(DROPOFF_p2, DROPOFF_p2, t, t, 'p2', '_') for t in times]   # Dropoff p2
)

# Waiting arcs
arcsWaiting = [(i, i, t, t + 1, w, w)
               for i in spaces
               for t in range(0, max(times))  # last waiting until T-1 -> T
               for w in states]

# --- Super-sink for open routes ---
OMEGA = 'Omega'  # sink node label (not in 'spaces')

def endPenalty(i, t):
    # 0.0 => truly "stop anywhere". Customize to add repositioning cost if desired.
    return 0.0

# End arcs: from ANY (i, t, '_') to Omega, zero-time arc
arcsEnd = [(i, OMEGA, t, t, '_', '_') for i in spaces for t in times]

# Summarize all arcs
arcsSTS = arcsTransport + arcsService + arcsWaiting + arcsEnd

# Fast arc-type membership sets
setTransport = set(arcsTransport)
setWaiting   = set(arcsWaiting)
setService   = set(arcsService)
setEnd       = set(arcsEnd)

# Travel time (s - t) and costs
tt = {}
tc = {}

for arc in arcsSTS:
    if arc in setTransport:
        i, j = arc[0], arc[1]
        tt[arc] = travel_times[(i, j)]
        tc[arc] = costs[(i, j)]
    elif arc in setWaiting:
        tt[arc] = (arc[3] - arc[2])  # 1
        tc[arc] = 0.0
    elif arc in setService:
        tt[arc] = 0
        tc[arc] = 0.0
    else:  # end arcs to Omega
        i, t = arc[0], arc[2]
        tt[arc] = 0
        tc[arc] = float(endPenalty(i, t))

# Select optimization objective
def select_objective():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    choice = simpledialog.askinteger(
        "Optimization Objective",
        "Select optimization criterion:\n\n"
        "1 - Minimize Travel Time\n"
        "2 - Minimize Cost\n\n"
        "Enter your choice (1 or 2):",
        minvalue=1,
        maxvalue=2
    )

    root.destroy()
    return choice

# Get user's choice
try:
    objective_choice = select_objective()
    if objective_choice is None:  # User closed the window
        print("No objective selected - exiting")
        exit()
except Exception as e:
    messagebox.showerror("Error", f"Failed to get objective choice: {str(e)}")
    exit()

# Initialize model
model = pulp.LpProblem("VRP_OpenRoute", pulp.LpMinimize)
y_vars_v1 = pulp.LpVariable.dicts("y_v1", arcsSTS, cat='Binary')  # decision var for v1 using arcs or not
y_vars_v2 = pulp.LpVariable.dicts("y_v2", arcsSTS, cat='Binary')  # decision var for v2 using arcs or not

# Set objective function based on user choice
if objective_choice == 1:
    model += pulp.lpSum(tt[arc] * (y_vars_v1[arc] + y_vars_v2[arc]) for arc in arcsSTS), "total_travel_time"
    print("Optimizing for minimum travel time")
else:
    model += pulp.lpSum(tc[arc] * (y_vars_v1[arc] + y_vars_v2[arc]) for arc in arcsSTS), "total_cost"
    print("Optimizing for minimum cost")

# Start constraints (leave start at t=0 in empty state)
model += pulp.lpSum(y_vars_v1[arc] for arc in arcsSTS
                    if arc[0] == START_v1 and arc[2] == 0 and arc[4] == '_' and arc[5] == '_') == 1, "start_v1"
model += pulp.lpSum(y_vars_v2[arc] for arc in arcsSTS
                    if arc[0] == START_v2 and arc[2] == 0 and arc[4] == '_' and arc[5] == '_') == 1, "start_v2"

# Open-route end constraints (exactly one end arc to Omega per vehicle)
model += pulp.lpSum(y_vars_v1[arc] for arc in arcsEnd) == 1, "end_v1"
model += pulp.lpSum(y_vars_v2[arc] for arc in arcsEnd) == 1, "end_v2"

# Flow balance (skip only the respective start vertices)
for vertex in vertexs:
    i, t, w = vertex
    if not (i == START_v1 and t == 0 and w == '_'):
        inflow_v1 = pulp.lpSum(y_vars_v1[arc] for arc in arcsSTS if arc[1] == i and arc[3] == t and arc[5] == w)
        outflow_v1 = pulp.lpSum(y_vars_v1[arc] for arc in arcsSTS if arc[0] == i and arc[2] == t and arc[4] == w)
        model += (inflow_v1 - outflow_v1) == 0, f"flow_balance_v1_vertex_{vertex}"

for vertex in vertexs:
    i, t, w = vertex
    if not (i == START_v2 and t == 0 and w == '_'):
        inflow_v2 = pulp.lpSum(y_vars_v2[arc] for arc in arcsSTS if arc[1] == i and arc[3] == t and arc[5] == w)
        outflow_v2 = pulp.lpSum(y_vars_v2[arc] for arc in arcsSTS if arc[0] == i and arc[2] == t and arc[4] == w)
        model += (inflow_v2 - outflow_v2) == 0, f"flow_balance_v2_vertex_{vertex}"

# Pickup exactly once (p1 & p2)
model += (pulp.lpSum(y_vars_v1[arc] for arc in arcsSTS if arc in setService and arc[4] == '_'  and arc[5] == 'p1') +
          pulp.lpSum(y_vars_v2[arc] for arc in arcsSTS if arc in setService and arc[4] == '_'  and arc[5] == 'p1')) == 1, "pickup_p1"

model += (pulp.lpSum(y_vars_v1[arc] for arc in arcsSTS if arc in setService and arc[4] == '_'  and arc[5] == 'p2') +
          pulp.lpSum(y_vars_v2[arc] for arc in arcsSTS if arc in setService and arc[4] == '_'  and arc[5] == 'p2')) == 1, "pickup_p2"

# Drop-off exactly once (p1 & p2)
model += (pulp.lpSum(y_vars_v1[arc] for arc in arcsSTS if arc in setService and arc[4] == 'p1' and arc[5] == '_') +
          pulp.lpSum(y_vars_v2[arc] for arc in arcsSTS if arc in setService and arc[4] == 'p1' and arc[5] == '_')) == 1, "dropoff_p1"

model += (pulp.lpSum(y_vars_v1[arc] for arc in arcsSTS if arc in setService and arc[4] == 'p2' and arc[5] == '_') +
          pulp.lpSum(y_vars_v2[arc] for arc in arcsSTS if arc in setService and arc[4] == 'p2' and arc[5] == '_')) == 1, "dropoff_p2"

# Before solving
print("Starting optimization...")

# Solve the optimization problem
model.solve()

# After solving
print(f"Optimization complete - Status: {pulp.LpStatus[model.status]}")
try:
    print("Objective value:", pulp.value(model.objective))
except Exception:
    pass

# ================ Results ================
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

def print_sorted_solutions(df_v1, df_v2):
    dv1 = prettify_solution_df(df_v1, OMEGA=OMEGA)
    dv2 = prettify_solution_df(df_v2, OMEGA=OMEGA)

    # Create numeric sort keys (safe if IDs are digits; non-digits become NaN and sorted last)
    dv1['_i_num'] = pd.to_numeric(dv1['i'], errors='coerce')
    dv1['_j_num'] = pd.to_numeric(dv1['j'], errors='coerce')
    dv2['_i_num'] = pd.to_numeric(dv2['i'], errors='coerce')
    dv2['_j_num'] = pd.to_numeric(dv2['j'], errors='coerce')

    dv1 = dv1.sort_values(by=['t', 's', '_i_num', '_j_num', 'i', 'j'])
    dv2 = dv2.sort_values(by=['t', 's', '_i_num', '_j_num', 'i', 'j'])

    print("\nVehicle 1 Path (Sorted):")
    print(dv1[['i', 'j', 't', 's', 'w', "w'"]].to_string(index=False))

    print("\nVehicle 2 Path (Sorted):")
    print(dv2[['i', 'j', 't', 's', 'w', "w'"]].to_string(index=False))

if model.status == pulp.LpStatusOptimal:
    arcs_v1 = [arc for arc in arcsSTS if pulp.value(y_vars_v1[arc]) == 1]
    arcs_v2 = [arc for arc in arcsSTS if pulp.value(y_vars_v2[arc]) == 1]

    df_solution_v1 = pd.DataFrame(arcs_v1, columns=["i", "j", "t", "s", "w", "w'"])
    df_solution_v2 = pd.DataFrame(arcs_v2, columns=["i", "j", "t", "s", "w", "w'"])

    print_sorted_solutions(df_solution_v1, df_solution_v2)
else:
    print("No optimal solution found. Check time discretization, connectivity, and service feasibility.")


# ================== ANIMATION (Matplotlib with Start/Replay & 4x3 layout) ==================
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.widgets import Button
import numpy as np
import pandas as pd

def _nn(x):
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") else s

def layout_4x3(spaces, dx=2.0, dy=2.0):
    """
    Fixed 4x3 layout:
      Row 1 (top):    1   2   3   4
      Row 2 (middle): 10      (blank) (blank)    5
      Row 3 (bottom): 9   8   7   6
    Columns aligned: (1,10,9) | (2,8) | (3,7) | (4,5,6)
    """
    # Column indexes 0..3, Row indexes 2..0 (top=2, middle=1, bottom=0)
    col = {'1':0,'2':1,'3':2,'4':3,'10':0,'5':3,'9':0,'8':1,'7':2,'6':3}
    row = {'1':2,'2':2,'3':2,'4':2,'10':1,'5':1,'9':0,'8':0,'7':0,'6':0}

    # Center the grid roughly around (0,0)
    def to_xy(n):
        c = col.get(n, 0)
        r = row.get(n, 1)
        x = (c - 1.5) * dx
        y = (r - 1.0) * dy
        return (x, y)

    # Only place nodes that exist in 'spaces'
    pos = {}
    for n in map(_nn, spaces):
        pos[n] = to_xy(n)
    return pos

def classify_arc(row, setTransport, setWaiting, setService):
    tup = (row['i'], row['j'], row['t'], row['s'], row['w'], row["w'"])
    if tup in setTransport: return 'move'
    if tup in setWaiting:   return 'wait'
    if tup in setService:   return 'service'
    return 'other'

def build_segments(df, setTransport, setWaiting, setService, vehicle, OMEGA='Omega'):
    df = df.copy()
    df['i'] = df['i'].apply(_nn)
    df['j'] = df['j'].apply(_nn)
    df = df[df['j'] != OMEGA].copy()
    df = df.sort_values(['t','s','i','j'])

    segs = []
    events = []  # list of (time:int, kind:str, payload:dict with 'p','node','veh')
    for _, r in df.iterrows():
        a_type = classify_arc(r, setTransport, setWaiting, setService)
        if a_type == 'service':
            if r['w'] == '_' and r["w'"] in ('p1','p2'):
                events.append((int(r['t']), 'pickup', {'p': r["w'"], 'node': r['i'], 'veh': vehicle}))
            elif r['w'] in ('p1','p2') and r["w'"] == '_':
                events.append((int(r['t']), 'dropoff', {'p': r['w'], 'node': r['i'], 'veh': vehicle}))
        elif a_type in ('move','wait'):
            segs.append({
                'type': a_type,
                'i': r['i'],
                'j': r['j'],
                't0': int(r['t']),
                't1': int(r['s'])
            })
    return segs, events

def initial_node(df):
    if df.empty: return None
    r = df.sort_values(['t','s']).iloc[0]
    return _nn(r['i'])

def pickup_node_from_df(df, pax):
    svc = df[(df['w']=='_') & (df["w'"]==pax)]
    if svc.empty: return None
    return _nn(svc.iloc[0]['i'])

def animate_routes(df_v1, df_v2, spaces, streets,
                   setTransport, setWaiting, setService,
                   OMEGA='Omega', FRAMES_PER_UNIT=12,
                   SLOW_FACTOR=3,   # <-- 3x slower than before
                   NODE_RADIUS=0.18, VEH_RADIUS=0.25, PAX_RADIUS=0.18):

    # layout + figure
    pos = layout_4x3(spaces, dx=2.2, dy=2.2)
    fig, ax = plt.subplots(figsize=(7.5,7.5))
    plt.subplots_adjust(bottom=0.22)  # leave space for buttons
    ax.set_aspect('equal')
    ax.axis('off')

    # draw edges
    for (u,v) in streets:
        u, v = _nn(u), _nn(v)
        if u in pos and v in pos:
            x1,y1 = pos[u]; x2,y2 = pos[v]
            ax.plot([x1,x2],[y1,y2], lw=1.0, color='#CCCCCC', zorder=1)

    # draw nodes + labels
    for n,(x,y) in pos.items():
        ax.add_patch(Circle((x,y), NODE_RADIUS, facecolor='#f0f0f0', edgecolor='#999999', zorder=2))
        ax.text(x, y+0.45, n, ha='center', va='bottom', fontsize=10, color='#444444', zorder=3)

    # segments + events
    segs_v1, ev_v1 = build_segments(df_v1, setTransport, setWaiting, setService, vehicle='v1', OMEGA=OMEGA)
    segs_v2, ev_v2 = build_segments(df_v2, setTransport, setWaiting, setService, vehicle='v2', OMEGA=OMEGA)

    # time horizon
    T = 0
    for seg in (segs_v1 + segs_v2):
        T = max(T, seg['t1'])
    total_frames = max(1, T * FRAMES_PER_UNIT)

    # initial positions
    start1 = initial_node(df_v1)
    start2 = initial_node(df_v2)
    x1,y1 = pos[start1] if start1 in pos else (0,0)
    x2,y2 = pos[start2] if start2 in pos else (0,0)

    veh1 = Circle((x1,y1), VEH_RADIUS, facecolor='#0078d7', edgecolor='none', zorder=5)  # blue
    veh2 = Circle((x2,y2), VEH_RADIUS, facecolor='#d32f2f', edgecolor='none', zorder=5)  # red
    ax.add_patch(veh1); ax.add_patch(veh2)

    # passengers initial at their pickup nodes
    both = pd.concat([df_v1, df_v2], ignore_index=True)
    p1_pick = pickup_node_from_df(both, 'p1')
    p2_pick = pickup_node_from_df(both, 'p2')
    if p1_pick and p1_pick in pos: px,py = pos[p1_pick]
    else: px,py = (x1,y1)
    if p2_pick and p2_pick in pos: qx,qy = pos[p2_pick]
    else: qx,qy = (x2,y2)
    pax1 = Circle((px,py), PAX_RADIUS, facecolor='#2e7d32', edgecolor='none', zorder=4)  # p1 green
    pax2 = Circle((qx,qy), PAX_RADIUS, facecolor='#ef6c00', edgecolor='none', zorder=4)  # p2 orange
    ax.add_patch(pax1); ax.add_patch(pax2)

    riding = {'v1': None, 'v2': None}

    # helpers
    def seg_at_time(segs, t):
        for s in segs:
            if s['t0'] <= t < s['t1']:
                return s
        return None

    # group service events at integer times (already include 'veh' tag)
    from collections import defaultdict
    service_events = defaultdict(list)
    for tt, kind, pl in (ev_v1 + ev_v2):
        service_events[int(tt)].append((kind, pl, pl['veh']))

    # --- buttons ---
    ax_start = plt.axes([0.22, 0.07, 0.22, 0.09])   # [left, bottom, width, height]
    ax_replay = plt.axes([0.56, 0.07, 0.22, 0.09])
    btn_start = Button(ax_start, 'Start')
    btn_replay = Button(ax_replay, 'Replay')

    # animation control
    ani = None  # will hold FuncAnimation instance

    def reset_state():
        """Reset positions/visibility/riding to the initial state."""
        veh1.center = (x1,y1)
        veh2.center = (x2,y2)
        pax1.center = (px,py); pax1.set_visible(True)
        pax2.center = (qx,qy); pax2.set_visible(True)
        riding['v1'] = None
        riding['v2'] = None
        fig.canvas.draw_idle()

    def update(frame_idx):
        # 3× slower via interval below; time mapping unchanged
        t_cont = frame_idx / FRAMES_PER_UNIT
        t_int  = int(round(t_cont))

        # process zero-time service events at exact integer boundaries
        if (frame_idx % FRAMES_PER_UNIT) == 0:
            for kind, pl, veh in service_events.get(t_int, []):
                if kind == 'pickup':
                    if pl['p'] == 'p1':
                        pax1.set_visible(False)
                        riding[veh] = 'p1'
                    elif pl['p'] == 'p2':
                        pax2.set_visible(False)
                        riding[veh] = 'p2'
                elif kind == 'dropoff':
                    node = pl['node']
                    if node in pos:
                        if riding[veh] == 'p1':
                            pax1.center = pos[node]; pax1.set_visible(True)
                        if riding[veh] == 'p2':
                            pax2.center = pos[node]; pax2.set_visible(True)
                    riding[veh] = None

        # vehicle 1 position
        seg1 = seg_at_time(segs_v1, t_cont)
        if seg1 is not None:
            (x_from, y_from) = pos[seg1['i']]
            (x_to,   y_to)   = pos[seg1['j']]
            if seg1['type'] == 'move' and seg1['t1'] > seg1['t0']:
                alpha = (t_cont - seg1['t0']) / (seg1['t1'] - seg1['t0'])
            else:
                alpha = 0.0
            x = x_from + (x_to - x_from) * alpha
            y = y_from + (y_to - y_from) * alpha
            veh1.center = (x,y)
            if riding['v1'] == 'p1': pax1.center = (x,y)
            if riding['v1'] == 'p2': pax2.center = (x,y)

        # vehicle 2 position
        seg2 = seg_at_time(segs_v2, t_cont)
        if seg2 is not None:
            (x_from, y_from) = pos[seg2['i']]
            (x_to,   y_to)   = pos[seg2['j']]
            if seg2['type'] == 'move' and seg2['t1'] > seg2['t0']:
                alpha = (t_cont - seg2['t0']) / (seg2['t1'] - seg2['t0'])
            else:
                alpha = 0.0
            x = x_from + (x_to - x_from) * alpha
            y = y_from + (y_to - y_from) * alpha
            veh2.center = (x,y)
            if riding['v2'] == 'p1': pax1.center = (x,y)
            if riding['v2'] == 'p2': pax2.center = (x,y)

        return veh1, veh2, pax1, pax2

    # 3× slower than before:
    interval_ms = int(SLOW_FACTOR * 1000 / FRAMES_PER_UNIT)

    def start_clicked(event):
        nonlocal ani
        reset_state()
        if ani is not None:
            try: ani.event_source.stop()
            except Exception: pass
        ani = FuncAnimation(fig, update, frames=total_frames+1,
                            interval=interval_ms, blit=False, repeat=False)
        fig.canvas.draw_idle()

    def replay_clicked(event):
        # Just re-run from the beginning
        start_clicked(event)

    btn_start.on_clicked(start_clicked)
    btn_replay.on_clicked(replay_clicked)

    plt.title("VRPPDTW – Route Animation (v1 blue, v2 red • p1 green, p2 orange)")
    plt.show()

# -------- run the animation UI if solved --------
if model.status == pulp.LpStatusOptimal:
    animate_routes(df_solution_v1, df_solution_v2, spaces, streets,
                   setTransport, setWaiting, setService,
                   OMEGA=OMEGA, FRAMES_PER_UNIT=12, SLOW_FACTOR=3)
