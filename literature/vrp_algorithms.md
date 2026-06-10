# Vehicle Routing Problem (VRP) Solving Algorithms

This document provides a structured taxonomy of the main algorithms used to solve the **Vehicle Routing Problem (VRP)** and its variants (e.g., CVRP, VRPTW, VRPPDTW, PCVRP). It categorizes methods from traditional exact mathematical models to classical heuristics, metaheuristics, Neural Combinatorial Optimization (NCO), and emerging LLM-driven heuristic designs.

---

## Algorithm Taxonomy (Hierarchical Left-to-Right Flowchart)

```mermaid
graph LR
    Root(("Vehicle Routing Problem<br/>Solving Algorithms"))
    
    %% Main Categories
    Exact["Exact Methods"]
    Classical["Classical Heuristics"]
    Meta["Metaheuristics"]
    NCO_ML["Neural Combinatorial<br/>Optimization & ML"]

    Root --> Exact
    Root --> Classical
    Root --> Meta
    Root --> NCO_ML

    %% Exact Methods
    Exact --> BB["Branch and Bound"]
    Exact --> BC["Branch and Cut"]
    Exact --> BP["Branch and Price<br/>(Column Generation)"]
    Exact --> ILP["Integer Linear Programming & DP"]

    %% Classical Heuristics
    Classical --> Constructive["Constructive Heuristics"]
    Classical --> LocalSearch["Improvement Heuristics<br/>(Local Search)"]

    Constructive --> Savings["Clarke & Wright Savings"]
    Constructive --> NN["Nearest Neighbor"]
    Constructive --> Insertion["Greedy Insertion"]
    Constructive --> Sweep["Sweep Algorithm"]

    LocalSearch --> KOpt["2-opt / 3-opt / k-opt"]
    LocalSearch --> OrOpt["Or-opt"]
    LocalSearch --> Exchange["Relocate & Exchange"]

    %% Metaheuristics
    Meta --> Trajectory["Single-Solution (Trajectory)"]
    Meta --> Population["Population-Based (Evolutionary)"]

    Trajectory --> TS["Tabu Search (TS)"]
    Trajectory --> SA["Simulated Annealing (SA)"]
    Trajectory --> ILS["Iterated Local Search (ILS)"]
    Trajectory --> LNS["Large Neighborhood Search<br/>(LNS / ALNS)"]
    LNS --> RuinRecreate["Ruin and Recreate"]
    Trajectory --> GRASP["GRASP"]

    Population --> GA["Genetic Algorithms (GA / HGS)"]
    Population --> Memetic["Memetic Algorithms"]
    Population --> ACO["Ant Colony (ACO)"]
    Population --> PSO["Particle Swarm (PSO)"]

    %% NCO & ML
    NCO_ML --> DRL["Deep Reinforcement Learning (DRL)"]
    NCO_ML --> LLM["LLM-Driven Heuristic Design"]

    DRL --> DRL_Constructive["Constructive (e.g., Pointer Nets, Attention)"]
    DRL --> DRL_Improvement["Improvement (Neural Local Search, GNN)"]

    LLM --> AHD["Automatic Heuristic Design (AILS-AHD)"]
    LLM --> Operator["LLM Operator Discovery (VRPAgent)"]
```
