#!/usr/bin/env python3
"""Generate vrp_config.json with random start/pickup/dropoff nodes."""
import argparse
import json
import os
import random
import sys
from typing import List

import pandas as pd


def positive_int(value: int, label: str) -> int:
    if value is None:
        return value
    if value <= 0:
        raise ValueError(f"{label} must be positive.")
    return value


def prompt_for(label: str) -> int:
    while True:
        try:
            raw = input(f"Enter {label}: ").strip()
            value = int(raw)
            if value <= 0:
                raise ValueError
            return value
        except ValueError:
            print(f"Please enter a positive integer for {label}.")


def load_node_ids(nodes_path: str) -> List[int]:
    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"nodes file not found: {nodes_path}")
    df = pd.read_csv(nodes_path)
    if 'id' not in df.columns:
        raise ValueError("nodes file must contain an 'id' column")
    ids = df['id'].dropna().tolist()
    if not ids:
        raise ValueError("nodes file contains no node ids")
    return [int(x) for x in ids]


def pick_nodes(rng: random.Random, population: List[int], count: int) -> List[int]:
    if count <= len(population):
        return rng.sample(population, count)
    return [rng.choice(population) for _ in range(count)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate vrp_config.json entries")
    parser.add_argument('-n', '--vehicles', type=int, help='number of vehicles')
    parser.add_argument('-m', '--passengers', type=int, help='number of passengers')
    parser.add_argument('--seed', type=int, default=None, help='optional RNG seed')
    parser.add_argument('--scenario-name', default='vrp_scenario', help='name stored in config')
    parser.add_argument('--nodes-file', default='nodes_osm.csv', help='path to nodes CSV')
    parser.add_argument('--config-file', default='vrp_config.json', help='path to output config JSON')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        n = positive_int(args.vehicles, 'vehicles') if args.vehicles is not None else None
        m = positive_int(args.passengers, 'passengers') if args.passengers is not None else None
    except ValueError as exc:
        print(exc)
        sys.exit(1)

    if n is None:
        n = prompt_for('number of vehicles')
    if m is None:
        m = prompt_for('number of passengers')

    if n <= 0 or m <= 0:
        print('Both n and m must be positive integers.')
        sys.exit(1)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    nodes_path = os.path.join(base_dir, args.nodes_file)
    config_path = os.path.join(base_dir, args.config_file)

    node_ids = load_node_ids(nodes_path)
    rng = random.Random(args.seed)

    start_nodes = pick_nodes(rng, node_ids, n)
    pickup_nodes = pick_nodes(rng, node_ids, m)
    drop_nodes = pick_nodes(rng, node_ids, m)

    vehicles = [
        {'id': idx + 1, 'start_node': int(node)}
        for idx, node in enumerate(start_nodes)
    ]
    passengers = [
        {
            'id': idx + 1,
            'pickup_node': int(pickup),
            'dropoff_node': int(drop)
        }
        for idx, (pickup, drop) in enumerate(zip(pickup_nodes, drop_nodes))
    ]

    config = {
        'scenario_name': args.scenario_name,
        'n': n,
        'm': m,
        'passengers': passengers,
        'vehicles': vehicles,
    }

    with open(config_path, 'w', encoding='utf-8') as fh:
        json.dump(config, fh, indent=2)
        fh.write('\n')

    print(f"Configuration written to {config_path}")
    print(f"Vehicles: {n}, Passengers: {m}")


if __name__ == '__main__':
    main()
