import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

def compute_features(G, root_id):
    """
    Compute 5 structural + temporal features from a cascade graph.
    Returns a dictionary of feature values.
    """
    nodes = dict(G.nodes(data=True))
    n = G.number_of_nodes()

    # --- structural features ---

    # 1. cascade depth — longest path from root
    try:
        depths = nx.single_source_shortest_path_length(G, root_id)
        depth = max(depths.values()) if depths else 0
    except Exception:
        depth = 0

    # 2. max branching factor — max number of direct children any node has
    out_degrees = [d for _, d in G.out_degree()]
    branching   = max(out_degrees) if out_degrees else 0

    # 3. cascade size — total number of tweets (including root)
    size = n

    # --- temporal features ---

    t_values = [data['t_min'] for _, data in nodes.items()
                if isinstance(data.get('t_min'), (int, float))]

    # 4. growth rate — number of tweets in first 30 minutes
    growth_30m = sum(1 for t in t_values if t <= 30)

    

    # 6. burstiness — how unevenly spaced the replies are
    #    = (std - mean) / (std + mean) of inter-arrival times
    #    ranges from -1 (perfectly regular) to +1 (very bursty)
    if len(t_values) > 2:
        t_sorted       = sorted(t_values)
        inter_arrivals = [t_sorted[i+1] - t_sorted[i]
                          for i in range(len(t_sorted)-1)]
        mu  = np.mean(inter_arrivals)
        std = np.std(inter_arrivals)
        burstiness = (std - mu) / (std + mu) if (std + mu) > 0 else 0.0
    else:
        burstiness = 0.0

    return {
        'depth'       : depth,
        'branching'   : branching,
        'size'        : size,
        'growth_30m'  : growth_30m,
        'burstiness'  : round(burstiness, 4),
    }

