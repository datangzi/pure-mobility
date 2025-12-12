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
from matplotlib.collections import LineCollection
import threading
import time
import math

# ---------- Helpers ----------
def norm_node(x):
    """Return node id as a clean string without trailing .0 etc."""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

# =================== ROAD NETWORK DEFINITION ===================
# Define the path to your data files (prefer OSM exports if present)
current_dir = os.path.dirname(os.path.abspath(__file__))
nodes_path = os.path.join(current_dir, 'nodes_osm.csv')
streets_path = os.path.join(current_dir, 'edges_osm.csv')

# Selection mode: set True to pick points by clicking on a plotted network
USE_MAP_SELECTION = True

# Globals for optional plotting
streets_df_global = None

def load_network_data():
    global streets_df_global
    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"nodes file not found at {nodes_path}")
    if not os.path.exists(streets_path):
        raise FileNotFoundError(f"streets file not found at {streets_path}")
    
    try:
        # Read node data
        nodes_df = pd.read_csv(nodes_path)
        if 'id' not in nodes_df.columns:
            raise ValueError("nodes file must contain an 'id' column")

        # Normalize node ids to strings
        nodes_df['id'] = nodes_df['id'].apply(norm_node)
        spaces = nodes_df['id'].tolist()

        # Read street data
        streets_df = pd.read_csv(streets_path)
        required_cols = {'from_node', 'to_node', 'travel_time'}
        if not required_cols.issubset(streets_df.columns):
            miss = required_cols - set(streets_df.columns)
            raise ValueError(f"streets file missing columns: {miss}")

        # Normalize street endpoints to strings to match 'spaces'
        streets_df['from_node'] = streets_df['from_node'].apply(norm_node)
        streets_df['to_node']   = streets_df['to_node'].apply(norm_node)

        # Times/costs as numbers
        streets_df['travel_time'] = streets_df['travel_time'].astype(float).astype(int)

        streets = set()
        travel_times = {}

        print("Script started")  # First line after imports

        for _, row in streets_df.iterrows():
            i, j = row['from_node'], row['to_node']  # strings
            streets.add((i, j))
            travel_times[(i, j)] = int(row['travel_time'])

        # Keep a copy for plotting
        streets_df_global = streets_df.copy()

        return spaces, streets, travel_times, nodes_df

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load network data: {str(e)}")
        raise

# Load network data
try:
    spaces, streets, travel_times, nodes_df = load_network_data()
    print("Network loaded successfully:")
except Exception as e:
    print(f"Error: {str(e)}")
    input("Press Enter to exit...")
    exit()

# =================== GENERATE MAP (zoom + hover) ===================
def show_interactive_map(nodes_df, streets_df=None):
    xs = nodes_df['x'].to_numpy()
    ys = nodes_df['y'].to_numpy()
    ids = nodes_df['id'].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('Longitude (x)')
    ax.set_ylabel('Latitude (y)')
    ax.set_title('OSM Network (scroll to zoom, hover to highlight)')

    # Draw edges for context if provided
    if streets_df is not None:
        id_to_xy = {row['id']: (row['x'], row['y']) for _, row in nodes_df.iterrows()}
        for _, r in streets_df.iterrows():
            u, v = norm_node(r['from_node']), norm_node(r['to_node'])
            if u in id_to_xy and v in id_to_xy:
                x0, y0 = id_to_xy[u]
                x1, y1 = id_to_xy[v]
                ax.plot([x0, x1], [y0, y1], color='lightgray', linewidth=0.5, alpha=0.6, zorder=1)

    # Scatter nodes
    ax.scatter(xs, ys, s=8, c='black', alpha=0.7, zorder=2)

    # Highlight marker and annotation (hidden by default)
    highlight, = ax.plot([], [], marker='o', markersize=8, markerfacecolor='yellow',
                         markeredgecolor='black', linestyle='None', zorder=3)
    annot = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def _nearest(x0, y0):
        d2 = (xs - x0) ** 2 + (ys - y0) ** 2
        idx = int(np.argmin(d2))
        return idx, float(d2[idx])

    def on_move(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            annot.set_visible(False)
            highlight.set_data([], [])
            fig.canvas.draw_idle()
            return
        x0, y0 = event.xdata, event.ydata
        idx, d2 = _nearest(x0, y0)
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        span = max(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]))
        tol = (span * 0.01) ** 2  # within ~1% of view span
        if d2 <= tol:
            x, y = xs[idx], ys[idx]
            highlight.set_data([x], [y])
            annot.xy = (x, y)
            annot.set_text(f"id: {ids[idx]}\nx: {x:.6f}\ny: {y:.6f}")
            annot.set_visible(True)
        else:
            annot.set_visible(False)
            highlight.set_data([], [])
        fig.canvas.draw_idle()

    def on_scroll(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        base = 1.2
        scale = 1 / base if event.button == 'up' else base
        cur_xlim = ax.get_xlim(); cur_ylim = ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        new_w = (cur_xlim[1] - cur_xlim[0]) * scale
        new_h = (cur_ylim[1] - cur_ylim[0]) * scale
        relx = (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
        rely = (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])
        ax.set_xlim([xdata - new_w * relx, xdata + new_w * (1 - relx)])
        ax.set_ylim([ydata - new_h * rely, ydata + new_h * (1 - rely)])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_move)
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    plt.show()


# Show the map
show_interactive_map(nodes_df, streets_df_global)
