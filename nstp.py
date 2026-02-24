"""
nstp.py — Neural Spanning Tree Protocol for lateral inhibition.

Solves G6 (Granularity Collapse) by pruning sibling prototypes that are
highly similar to each other but represent different classes.  When FAM
retrieves top-K candidates, sibling pairs (e.g. Husky and Wolf) compete
and the weaker one is suppressed, preventing soft-vote confusion.

The algorithm builds a virtual spanning tree rooted at the query:

    1. Root = prototype with max sim to query.
    2. Path cost for each candidate: cost(n) = 1 - sim(n, root).
    3. Sibling detection: pairs (A, B) where sim(A, B) > threshold AND
       |cost(A) - cost(B)| < epsilon (similar depth from root).
    4. Lateral inhibition: suppress the sibling with lower query similarity.
"""

import torch
import torch.nn.functional as F


class NSTPController:
    """Post-retrieval lateral inhibition filter.

    Args:
        sibling_threshold: Minimum inter-prototype cosine similarity to
            consider two candidates as siblings (default 0.85).
        depth_epsilon: Maximum difference in path cost for a pair to be
            considered at the same depth (siblings vs. parent-child).
        root_strategy: ``'proximity'`` uses the top-1 candidate as root;
            ``'context'`` accepts an external root vector.
    """

    def __init__(self, sibling_threshold: float = 0.85,
                 depth_epsilon: float = 0.10,
                 root_strategy: str = "proximity"):
        self.sibling_threshold = sibling_threshold
        self.depth_epsilon = depth_epsilon
        self.root_strategy = root_strategy

    def prune(self, query: torch.Tensor, keys: torch.Tensor,
              values: torch.Tensor, sims: torch.Tensor,
              root_vec: torch.Tensor | None = None):
        """Apply lateral inhibition to a set of retrieved candidates.

        Args:
            query:  Query vector, shape ``(D,)`` or ``(1, D)``.
            keys:   Candidate key vectors, shape ``(K, D)``.
            values: Candidate value vectors (one-hot or soft), ``(K, V)``.
            sims:   Query-to-candidate similarities, shape ``(K,)``.
            root_vec: External root vector for ``'context'`` strategy,
                      shape ``(D,)``.  Ignored when strategy is
                      ``'proximity'``.

        Returns:
            Tuple ``(mask, sims_out)`` where:
            - ``mask``:     Bool tensor ``(K,)`` — True = keep, False = suppressed.
            - ``sims_out``: Similarity tensor ``(K,)`` with suppressed entries
                            zeroed (useful for downstream softmax weighting).
        """
        query = query.squeeze(0)  # ensure (D,)
        K, D = keys.shape

        if K <= 1:
            return torch.ones(K, dtype=torch.bool, device=keys.device), sims

        # --- Step 1: Identify root ---
        if self.root_strategy == "context" and root_vec is not None:
            root = F.normalize(root_vec.squeeze(0).unsqueeze(0), dim=-1)
        else:
            root_idx = sims.argmax()
            root = keys[root_idx].unsqueeze(0)  # (1, D)
        root = F.normalize(root, dim=-1)

        # --- Step 2: Path cost = 1 - sim(candidate, root) ---
        keys_norm = F.normalize(keys.float(), dim=-1)
        root_sims = (keys_norm @ root.T).squeeze(-1)   # (K,)
        path_cost = 1.0 - root_sims                     # (K,)

        # --- Step 3: Detect sibling conflicts ---
        # Pairwise cosine similarity between candidates
        pairwise_sim = keys_norm @ keys_norm.T           # (K, K)

        # Pairwise depth difference
        depth_diff = (path_cost.unsqueeze(0) - path_cost.unsqueeze(1)).abs()

        # Sibling mask: high inter-sim AND similar depth AND different entries
        eye_mask = torch.eye(K, device=keys.device, dtype=torch.bool)
        siblings = ((pairwise_sim > self.sibling_threshold)
                     & (depth_diff < self.depth_epsilon)
                     & ~eye_mask)

        # --- Step 4: Lateral inhibition ---
        # For each sibling pair, suppress the one with lower query similarity.
        # Process greedily: once a candidate is suppressed, it can't suppress others.
        keep = torch.ones(K, dtype=torch.bool, device=keys.device)

        if siblings.any():
            # Get class labels from values for class-aware suppression
            classes = values.float().argmax(dim=-1)  # (K,)

            # Only suppress siblings of DIFFERENT classes
            # (same-class siblings should reinforce, not compete)
            class_diff = classes.unsqueeze(0) != classes.unsqueeze(1)
            conflict = siblings & class_diff

            if conflict.any():
                # For each candidate, count how many conflicts it has
                # Suppress candidates that lose the most pairwise comparisons
                conflict_rows, conflict_cols = conflict.nonzero(as_tuple=True)

                # Process pairs: for each conflict, suppress the weaker one
                suppressed = set()
                # Sort pairs by the similarity of the weaker member (ascending)
                # so we suppress the weakest first
                for i, j in zip(conflict_rows.tolist(), conflict_cols.tolist()):
                    if i >= j:  # avoid double-processing
                        continue
                    if i in suppressed or j in suppressed:
                        continue
                    # Suppress the one with lower query similarity
                    if sims[i] < sims[j]:
                        suppressed.add(i)
                    else:
                        suppressed.add(j)

                for idx in suppressed:
                    keep[idx] = False

        sims_out = sims.clone()
        sims_out[~keep] = 0.0

        return keep, sims_out

    def prune_batch(self, queries: torch.Tensor, keys: torch.Tensor,
                    values: torch.Tensor, sims: torch.Tensor):
        """Batched version: applies prune() per query in the batch.

        Args:
            queries: ``(B, D)``
            keys:    ``(B, K, D)``
            values:  ``(B, K, V)``
            sims:    ``(B, K)``

        Returns:
            Tuple ``(masks, sims_out)`` — both ``(B, K)``.
        """
        B, K = sims.shape
        masks = torch.ones_like(sims, dtype=torch.bool)
        sims_out = sims.clone()

        for b in range(B):
            m, s = self.prune(queries[b], keys[b], values[b], sims[b])
            masks[b] = m
            sims_out[b] = s

        return masks, sims_out
