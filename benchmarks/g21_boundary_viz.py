"""
g21_boundary_viz.py — Visualize the Structural Plateau (G21).

Loads a FAM model at the plateau operating point (adapter + NSTP,
vigilance=0.30), extracts stored prototypes and query assignments,
then projects to 2D via UMAP to reveal the Voronoi tessellation
of the concept space.

Output: outputs/g21_boundary_viz.png
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np

from fast_associative_memory import FastAssociativeMemory
from adapter import MetricAdapter
from nstp import NSTPController

CACHE_FILE = os.path.join(os.path.dirname(__file__), "cache", "g6_features.pt")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "g21_boundary_viz.png")


def load_adapter(path, device):
    adapter = MetricAdapter(input_dim=1024, output_dim=256, hidden_dim=512)
    adapter.load_state_dict(torch.load(path, weights_only=True))
    adapter.eval()
    return adapter.to(device)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load cached features ---
    if not os.path.exists(CACHE_FILE):
        print(f"ERROR: Run sweep_vigilance_floor.py first to generate {CACHE_FILE}")
        sys.exit(1)

    data = torch.load(CACHE_FILE, weights_only=True)
    features, labels = data["features"], data["labels"]
    print(f"Loaded {features.shape[0]} samples, {features.shape[1]}-d")

    # --- Build FAM at plateau operating point ---
    adapter = load_adapter("adapter_trained.pt", device)
    nstp = NSTPController(root_strategy="proximity")
    fam = FastAssociativeMemory(
        input_dim=1024, value_dim=1000,
        core_entries=10000, core_vigilance=0.30,
        adapter=adapter, nstp=nstp,
    ).to(device)

    # Ingest all samples
    perm = torch.randperm(features.size(0))
    for i in range(0, len(perm), 128):
        batch_idx = perm[i:i + 128]
        with torch.no_grad():
            fam.learn_local(features[batch_idx].to(device),
                            labels[batch_idx].to(device))

    # --- Extract prototypes ---
    occupied = fam.core_cam.occupied
    proto_slots = occupied.nonzero(as_tuple=True)[0]
    n_protos = len(proto_slots)
    proto_keys_raw = fam.core_cam.keys[proto_slots].float()  # (P, key_dim)
    proto_classes = fam.core_cam.values[proto_slots].float().argmax(dim=-1)  # (P,)
    print(f"Prototypes: {n_protos}")

    # --- Use all samples as queries ---
    n_queries = features.size(0)
    query_feats = features                # (Q, 1024)
    query_labels = labels                 # (Q,)

    # --- Project queries through adapter (same space as proto keys) ---
    with torch.no_grad():
        query_projected = fam._project(query_feats.to(device))  # (Q, adapter_dim)

    # --- Assign each query to its nearest prototype ---
    q_norm = F.normalize(query_projected.float(), dim=-1)          # (Q, D)
    p_norm = F.normalize(proto_keys_raw, dim=-1)                   # (P, D)
    sims = q_norm @ p_norm.T                                       # (Q, P)
    assignments = sims.argmax(dim=1).cpu()                         # (Q,) index into proto_slots

    queries_per_proto = torch.bincount(assignments, minlength=n_protos)
    mean_qpp = queries_per_proto[queries_per_proto > 0].float().mean().item()
    print(f"Mean queries-per-prototype: {mean_qpp:.1f}")
    print(f"Prototypes with 0 queries: "
          f"{(queries_per_proto == 0).sum().item()} / {n_protos}")

    # --- UMAP projection ---
    print("Running UMAP...")
    import umap

    # Combine prototypes and queries for joint embedding
    proto_np = proto_keys_raw.cpu().numpy()                         # (P, D)
    query_np = query_projected.cpu().numpy()                        # (Q, D)
    combined = np.vstack([proto_np, query_np])                      # (P+Q, D)

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine",
                        random_state=42)
    embedding = reducer.fit_transform(combined)
    proto_2d = embedding[:n_protos]       # (P, 2)
    query_2d = embedding[n_protos:]       # (Q, 2)

    # --- Plot ---
    print("Plotting...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(16, 12))

    # Color by the CLASS of the assigned prototype (basin of attraction).
    # This shows whether each query lands in a same-class basin.
    assign_classes = proto_classes[assignments].cpu().numpy()
    unique_classes = np.unique(assign_classes)
    n_classes = len(unique_classes)

    # Build a perceptually distinct palette for ~50 classes
    # Combine tab20b + tab20c for 40 distinct colors, extend with hsv
    base_colors = (list(plt.colormaps["tab20b"].colors)
                   + list(plt.colormaps["tab20c"].colors))
    if n_classes > len(base_colors):
        hsv = plt.colormaps["hsv"]
        extra = [hsv(i / (n_classes - len(base_colors)))[:3]
                 for i in range(n_classes - len(base_colors))]
        base_colors.extend(extra)
    # Normalize all to RGB triples
    palette = np.array([c[:3] for c in base_colors[:n_classes]])

    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    query_cidx = np.array([class_to_idx[c] for c in assign_classes])
    proto_cidx = np.array([class_to_idx[c]
                           for c in proto_classes.cpu().numpy()])

    # Queries: small dots colored by basin class
    ax.scatter(
        query_2d[:, 0], query_2d[:, 1],
        c=palette[query_cidx], s=4, alpha=0.4,
        edgecolors="none", rasterized=True,
    )

    # Prototypes: large X markers with black edge
    ax.scatter(
        proto_2d[:, 0], proto_2d[:, 1],
        c=palette[proto_cidx], s=100, alpha=0.95,
        marker="X", edgecolors="black", linewidths=0.6,
        zorder=10,
    )

    ax.set_title(
        f"G21: Structural Plateau — {n_protos} Prototypes, "
        f"{n_queries} Queries, {n_classes} Classes\n"
        f"Mean queries/proto: {mean_qpp:.1f}  |  "
        f"vigilance=0.30, adapter+NSTP\n"
        f"Each color = one class basin of attraction  |  "
        f"X = prototype centroid",
        fontsize=12,
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_xticks([])
    ax.set_yticks([])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUTPUT_FILE}")
    plt.close(fig)


if __name__ == "__main__":
    main()
