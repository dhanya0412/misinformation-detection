import numpy as np
import torch
from torch_geometric.data import Data

def build_pyg_data(G, label, root_id, all_node_embeddings, node_id_to_idx, cut_minutes=None):
    """
    Convert a networkx cascade graph to a PyG Data object.
    cut_minutes: if set, remove nodes with t_min > cut_minutes (early detection).
    Node features: BERT (768) + depth (1) + degree (1) + t_min_norm (1) = 771 dim
    """
    # --- apply time cut ---
    if cut_minutes is not None:
        valid_nodes = {n for n in G.nodes()
                       if G.nodes[n].get('t_min', 0) <= cut_minutes}
        valid_nodes.add(root_id)
        G = G.subgraph(valid_nodes).copy()

    nodes = list(G.nodes())
    if len(nodes) < 2:
        return None

    # --- node index mapping ---
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n_nodes     = len(nodes)

    # --- normalise t_min within this cascade ---
    t_mins  = np.array([G.nodes[n].get('t_min', 0.0) for n in nodes], dtype=np.float32)
    t_max   = t_mins.max() if t_mins.max() > 0 else 1.0
    t_norms = t_mins / t_max

    # --- normalise degree ---
    degrees  = np.array([G.degree(n) for n in nodes], dtype=np.float32)
    deg_max  = degrees.max() if degrees.max() > 0 else 1.0
    deg_norm = degrees / deg_max

    # --- normalise depth ---
    depths    = np.array([G.nodes[n].get('depth', 0) for n in nodes], dtype=np.float32)
    depth_max = depths.max() if depths.max() > 0 else 1.0
    dep_norm  = depths / depth_max

    # --- build feature matrix ---
    feat_list = []
    for i, n in enumerate(nodes):
        idx      = node_id_to_idx.get(str(n))
        bert_vec = all_node_embeddings[idx] if idx is not None \
                   else np.zeros(768, dtype=np.float32)
        extra    = np.array([dep_norm[i], deg_norm[i], t_norms[i]], dtype=np.float32)
        feat_list.append(np.concatenate([bert_vec, extra]))

    x = torch.tensor(np.stack(feat_list), dtype=torch.float)

    # --- top-down edge index ---
    td_edges = [(node_to_idx[u], node_to_idx[v])
                for u, v in G.edges()
                if u in node_to_idx and v in node_to_idx]

    if len(td_edges) == 0:
        return None

    td_src, td_dst = zip(*td_edges)
    td_edge_index  = torch.tensor([td_src, td_dst], dtype=torch.long)
    bu_edge_index  = torch.tensor([td_dst, td_src], dtype=torch.long)

    root_idx = node_to_idx.get(root_id, 0)

    return Data(
        x             = x,
        edge_index    = td_edge_index,
        BU_edge_index = bu_edge_index,
        y             = torch.tensor([label], dtype=torch.long),
        root_index    = torch.tensor([root_idx], dtype=torch.long),
        num_nodes     = n_nodes
    )