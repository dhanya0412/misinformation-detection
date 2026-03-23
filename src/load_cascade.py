import json
import os
import networkx as nx
from datetime import datetime

TWITTER_FMT = "%a %b %d %H:%M:%S +0000 %Y"

def parse_time(ts_str):
    """Convert Twitter date string to datetime object."""
    return datetime.strptime(ts_str, TWITTER_FMT)

def load_cascade(thread_dir):
    G = nx.DiGraph()
    label = 0

    # load source tweet
    src_folder = f"{thread_dir}/source-tweet"
    src_file   = os.listdir(src_folder)[0]
    with open(f"{src_folder}/{src_file}", encoding='utf-8') as f:
        src = json.load(f)

    root_id   = src['id_str']
    root_time = parse_time(src['created_at'])
    G.add_node(root_id, text=src.get('text',''), t_min=0.0, depth=0)

    # load reactions
    reaction_folder = f"{thread_dir}/reactions"
    if os.path.exists(reaction_folder):
        for fname in os.listdir(reaction_folder):
            with open(f"{reaction_folder}/{fname}", encoding='utf-8') as f:
                tw = json.load(f)
            if 'tweet' in tw:
                tw = tw['tweet']

            tw_id     = tw.get('id_str')
            parent_id = tw.get('in_reply_to_status_id_str')
            t_min     = (parse_time(tw['created_at']) - root_time).total_seconds() / 60

            if tw_id:
                G.add_node(tw_id,
                           text  = tw.get('text', ''),
                           t_min = round(t_min, 2))
            if tw_id and parent_id:
                G.add_edge(parent_id, tw_id)

    # --- fix disconnected nodes: attach orphans directly to root ---
    for node in list(G.nodes):
        if node == root_id:
            continue
        # if node has no path from root, connect it to root
        if not nx.has_path(G, root_id, node):
            G.add_edge(root_id, node)

    # compute depth via BFS from root
    try:
        depths = nx.single_source_shortest_path_length(G, root_id)
        nx.set_node_attributes(G, depths, 'depth')
    except Exception:
        nx.set_node_attributes(G, 0, 'depth')

    return G, label, root_id
def load_all_cascades(data_dir, events=None):
    """
    Loop over all PHEME events and load every cascade.

    Args:
        data_dir -- path to the pheme/ folder
        events   -- list of event names to load (None = all)

    Returns:
        list of (G, label, root_id, event_name) tuples
    """
    from tqdm import tqdm

    all_events = events or os.listdir(data_dir)
    all_cascades = []

    for event in all_events:
        event_path = os.path.join(data_dir, event)
        if not os.path.isdir(event_path):
            continue

        for label_name in ['rumours', 'non-rumours']:
            label_path = os.path.join(event_path, label_name)
            if not os.path.exists(label_path):
                continue

            threads = [t for t in os.listdir(label_path)
                       if os.path.isdir(os.path.join(label_path, t))]

            for thread in tqdm(threads, desc=f"{event}/{label_name}"):
                thread_dir = os.path.join(label_path, thread)
                try:
                    G, label, root_id = load_cascade(thread_dir)
                    all_cascades.append((G, label, root_id, event))
                except Exception as e:
                    print(f"  skipped {thread}: {e}")

    return all_cascades