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
# This an updated version based on vrppdtw_20250715.py
# In this version, one more vehicle and one more passenger was implemented
# User can also define the starting points of both vehicles v1, v2

# =================== ROAD NETWORK DEFINITION ===================
def load_network_data():
    if not os.path.exists('nodes.csv'):
        raise FileNotFoundError("nodes.csv not found")
    if not os.path.exists('streets.csv'):
        raise FileNotFoundError("streets.csv not found")
    try:
        # Read node data
        nodes_df = pd.read_csv('nodes.csv')
        spaces = nodes_df['id'].tolist()
        
        # Read street data
        streets_df = pd.read_csv('streets.csv')
        streets = set()
        travel_times = {}
        costs = {}  # New dictionary for costs
        
        print("Script started")  # First line after imports

        for _, row in streets_df.iterrows():
            streets.add((row['from_node'], row['to_node']))
            travel_times[(row['from_node'], row['to_node'])] = row['travel_time']
            costs[(row['from_node'], row['to_node'])] = row['cost']
            
        return spaces, streets, travel_times, costs, nodes_df
    
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load network data: {str(e)}")
        raise

# Load network data
try:
    spaces, streets, travel_times, costs, nodes_df = load_network_data()  # Modified to include costs
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
times = range(0, 11)  # t=0 to t=10
states = ['_', 'p1', 'p2']  # Passenger carrying state

# After loading network data
print(f"Network loaded - Nodes: {spaces}, Streets: {streets}")

# ============= STARTING POINTS DEFINITION ===============
def define_starting_point():
    global START_v1, START_v2
    try:
        START_v1 = int(start_entry_v1.get())
        START_v2 = int(start_entry_v2.get())
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
start_entry_v1 = tk.Entry(root, font=('Arial', 12)) # User defines starting point of v1
start_entry_v1.pack()

tk.Label(root, text="Starting point of v2:", font=('Arial', 12)).pack(pady=5)
start_entry_v2 = tk.Entry(root, font=('Arial', 12)) # User defines starting point of v2
start_entry_v2.pack()

tk.Button(root, text="Confirm", command=define_starting_point, 
          font=('Arial', 12), bg='#4CAF50', fg='white').pack(pady=10)

root.mainloop()

# After service point selection
print(f"Starting point of v1: {START_v1}, Starting point of v2: {START_v2}")

# ============= SERVICE POINTS DEFINITION ===============
def define_service_point():
    global PICKUP_p1, PICKUP_p2, DROPOFF_p1, DROPOFF_p2
    try:
        PICKUP_p1 = int(pickup_entry_p1.get())
        DROPOFF_p1 = int(dropoff_entry_p1.get())
        PICKUP_p2 = int(pickup_entry_p2.get())
        DROPOFF_p2 = int(dropoff_entry_p2.get())
        if PICKUP_p1 not in spaces or DROPOFF_p1 not in spaces or PICKUP_p2 not in spaces or DROPOFF_p2 not in spaces :
            raise ValueError("Service point must be a defined physical node")
        root.destroy()
    except ValueError as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

# Initialize GUI for service points only
root = tk.Tk()
root.title("Service Point Selection")

tk.Label(root, text=f"Available nodes: {spaces}", font=('Arial', 12)).pack(pady=5)

tk.Label(root, text="Pickup point for p1:", font=('Arial', 12)).pack(pady=5) 
pickup_entry_p1 = tk.Entry(root, font=('Arial', 12)) # User defines pickup for p1
pickup_entry_p1.pack()

tk.Label(root, text="Pickup point for p2:", font=('Arial', 12)).pack(pady=5) 
pickup_entry_p2 = tk.Entry(root, font=('Arial', 12)) # User defines pickup for p2
pickup_entry_p2.pack()

tk.Label(root, text="Dropoff point p1:", font=('Arial', 12)).pack(pady=5)
dropoff_entry_p1 = tk.Entry(root, font=('Arial', 12)) # User defines dropoff for p1
dropoff_entry_p1.pack()

tk.Label(root, text="Dropoff point p2:", font=('Arial', 12)).pack(pady=5)
dropoff_entry_p2 = tk.Entry(root, font=('Arial', 12)) # User defines dropoff for p2
dropoff_entry_p2.pack()

tk.Button(root, text="Start Simulation", command=define_service_point, 
          font=('Arial', 12), bg='#4CAF50', fg='white').pack(pady=10)

root.mainloop()

# After service point selection
print(f"Service points set - Pickup for p1: {PICKUP_p1}, Dropoff for p1: {DROPOFF_p1}, Pickup for p2: {PICKUP_p2}, Dropoff for p2: {DROPOFF_p2}")

# ====================== OPTIMIZATION MODEL ======================
# Generate all possible 3-dimensional vertexes
vertexs = [(i, t, w) for i in spaces for t in times for w in states]

# Generate arcs
arcsTransport = []
for i, j in streets:
    for t in times:
        required_time = travel_times[(i, j)]
        s = t + required_time
        if s <= max(times):
            for w in states:
                arcsTransport.append((i, j, t, s, w, w))

# Service arcs (dynamic based on GUI input)
arcsService = [
    (PICKUP_p1, PICKUP_p1, t, t, '_', 'p1') for t in times  # Pickup p1
] + [
    (DROPOFF_p1, DROPOFF_p1, t, t, 'p1', '_') for t in times  # Dropoff p2
] + [
    (PICKUP_p2, PICKUP_p2, t, t, '_', 'p2') for t in times  # Pickup p2
] + [
    (DROPOFF_p2, DROPOFF_p2, t, t, 'p2', '_') for t in times  # Dropoff p2
]

# Waiting arcs
arcsWaiting = [(i, i, t, t + 1, w, w) 
            for i in spaces 
            for t in range(0, 10) 
            for w in states]

# Summarize all arcs
arcsSTS = arcsTransport + arcsService + arcsWaiting

# Travel time (s - t for transport arcs) and costs
tt = {}  # Travel times
tc = {}  # Transportation costs (new)

for arc in arcsSTS:
    if arc in arcsTransport:
        i, j = arc[0], arc[1]
        tt[arc] = travel_times[(i, j)]  # Transport arcs time
        tc[arc] = costs[(i, j)]         # Transport arcs cost
    elif arc in arcsWaiting:
        tt[arc] = (arc[3] - arc[2])    # Waiting arcs time
        tc[arc] = 0                     # Waiting arcs cost (zero)
    else:  # Service arcs (pickup/dropoff)
        tt[arc] = 0
        tc[arc] = 0
        
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
model = pulp.LpProblem("VRP", pulp.LpMinimize)
y_vars_v1 = pulp.LpVariable.dicts("y_v1", arcsSTS, cat='Binary') # decision variable (0 or 1) for vehicle v1 using arcs or not
y_vars_v2 = pulp.LpVariable.dicts("y_v2", arcsSTS, cat='Binary') # decision variable (0 or 1) for vehicle v2 using arcs or not

# Set objective function based on user choice
if objective_choice == 1:
    model += pulp.lpSum(tt[arc] * (y_vars_v1[arc] + y_vars_v2[arc]) for arc in arcsSTS), "total_travel_time"
    print("Optimizing for minimum travel time")
else:
    model += pulp.lpSum(tc[arc] * (y_vars_v1[arc] + y_vars_v2[arc])  for arc in arcsSTS), "total_cost"
    print("Optimizing for minimum cost")

model += pulp.lpSum(y_vars_v1[arc] for arc in arcsSTS 
                if arc[0] == START_v1 and arc[2] == 0 and arc[4] == '_' and arc[5] == '_') == 1, "flow_balance_vehicle_v1_start" # meets (2) in paper for v1
model += pulp.lpSum(y_vars_v1[arc] for arc in arcsSTS 
                if arc[1] == 10 and arc[3] == 10 and arc[4] == '_' and arc[5] == '_') == 1, "flow_balance_vehicle_v1_end" # meets (3) in paper for v1
model += pulp.lpSum(y_vars_v2[arc] for arc in arcsSTS 
                if arc[0] == START_v2 and arc[2] == 0 and arc[4] == '_' and arc[5] == '_') == 1, "flow_balance_vehicle_v2_start" # meets (2) in paper for v2
model += pulp.lpSum(y_vars_v2[arc] for arc in arcsSTS 
                if arc[1] == 10 and arc[3] == 10 and arc[4] == '_' and arc[5] == '_') == 1, "flow_balance_vehicle_v2_end" # meets (3) in paper for v2

for vertex in vertexs:
    i, t, w = vertex
    if not ((i == START_v1 and t == 0 and w == '_') or (i == 10 and t == 10 and w == '_')): # already constrained over (2) and (3)
        inflow_v1 = pulp.lpSum(y_vars_v1[arc] for arc in arcsSTS 
                        if arc[1] == i and arc[3] == t and arc[5] == w)
        outflow_v1 = pulp.lpSum(y_vars_v1[arc] for arc in arcsSTS 
                        if arc[0] == i and arc[2] == t and arc[4] == w)
        model += (inflow_v1 - outflow_v1) == 0, f"flow_balance_v1_vertex_{vertex}" # meets (4) in paper for v1

for vertex in vertexs:
    i, t, w = vertex
    if not ((i == START_v2 and t == 0 and w == '_') or (i == 10 and t == 10 and w == '_')): # already constrained over (2) and (3)
        inflow_v2 = pulp.lpSum(y_vars_v2[arc] for arc in arcsSTS 
                        if arc[1] == i and arc[3] == t and arc[5] == w)
        outflow_v2 = pulp.lpSum(y_vars_v2[arc] for arc in arcsSTS 
                        if arc[0] == i and arc[2] == t and arc[4] == w)
        model += (inflow_v2 - outflow_v2) == 0, f"flow_balance_v2_vertex_{vertex}" # meets (4) in paper for v2

model += pulp.lpSum(y_vars_v1[arc] for arc in arcsSTS 
                if arc in arcsService and arc[4] == '_' and arc[5] == 'p1') + \
         pulp.lpSum(y_vars_v2[arc] for arc in arcsSTS 
                if arc in arcsService and arc[4] == '_' and arc[5] == 'p1') == 1, "pickup_p1" # meets (5) in paper for p1, no need to constraint dropoff because it's covered when (3) is satisfied

model += pulp.lpSum(y_vars_v1[arc] for arc in arcsSTS 
                if arc in arcsService and arc[4] == '_' and arc[5] == 'p2') + \
         pulp.lpSum(y_vars_v2[arc] for arc in arcsSTS 
                if arc in arcsService and arc[4] == '_' and arc[5] == 'p2') == 1, "pickup_p2" # meets (5) in paper for p2, no need to constraint dropoff because it's covered when (3) is satisfied

# Before solving
print("Starting optimization...")

# Solve the optimization problem
model.solve()

# After solving
print(f"Optimization complete - Status: {pulp.LpStatus[model.status]}")


# ================ Visualization ================
def print_sorted_solutions(df_v1, df_v2):
    # Sort Vehicle 1's path
    df_v1_sorted = df_v1.sort_values(by=['t', 's', 'i', 'j'])
    print("\nVehicle 1 Path (Sorted):")
    print(df_v1_sorted[['i', 'j', 't', 's', 'w', "w'"]].to_string(index=False))
    
    # Sort Vehicle 2's path
    df_v2_sorted = df_v2.sort_values(by=['t', 's', 'i', 'j'])
    print("\nVehicle 2 Path (Sorted):")
    print(df_v2_sorted[['i', 'j', 't', 's', 'w', "w'"]].to_string(index=False))

# After solving the model
if model.status == pulp.LpStatusOptimal:
    arcs_v1 = [arc for arc in arcsSTS if pulp.value(y_vars_v1[arc]) == 1]
    arcs_v2 = [arc for arc in arcsSTS if pulp.value(y_vars_v2[arc]) == 1]
    
    df_solution_v1 = pd.DataFrame(arcs_v1, columns=["i", "j", "t", "s", "w", "w'"])
    df_solution_v2 = pd.DataFrame(arcs_v2, columns=["i", "j", "t", "s", "w", "w'"])
    
    print_sorted_solutions(df_solution_v1, df_solution_v2)