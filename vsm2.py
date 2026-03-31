import random
import time
from typing import Dict, List, Set, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph
import os

# ---- PyTorch 2.6+ compatibility patch for OGB/PyG processed files ----
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load





class AnchorFirstPrefixBoundMatcher:
    """
    Exact semantic subgraph matching with:
    1) anchor-first strategy
    2) only the anchor query vertex gets global candidates
    3) anchor candidate generation uses exact prefix-bound reporting
    4) all other query vertices are expanded locally during search

    Prefix-bound reporting:
      - degree filter
      - stage-1 prefix upper bound
      - optional stage-2 prefix upper bound
      - exact full-dot refinement on survivors

    Exactness:
      safe upper bound by Cauchy-Schwarz, so no false negatives.
    """

    def __init__(
        self,
        data_graph: Data,
        query_graph: Data,
        tau: float,
        m1: int = 16,
        m2: int = 32,
    ):
        if data_graph.x.size(1) != query_graph.x.size(1):
            raise ValueError("Feature dimensions do not match.")

        self.G = data_graph
        self.Q = query_graph
        self.tau = float(tau)

        self.num_g = self.G.x.size(0)
        self.num_q = self.Q.x.size(0)
        self.feat_dim = self.G.x.size(1)

        self.m1 = min(int(m1), self.feat_dim)
        self.m2 = min(int(m2), self.feat_dim)
        if self.m2 < self.m1:
            self.m2 = self.m1

        self.G_x = F.normalize(self.G.x.float(), p=2, dim=1)
        self.Q_x = F.normalize(self.Q.x.float(), p=2, dim=1)

        self.G_adj = self._build_adj(self.G.edge_index, self.num_g)
        self.Q_adj = self._build_adj(self.Q.edge_index, self.num_q)

        self.G_deg = [len(self.G_adj[v]) for v in range(self.num_g)]
        self.Q_deg = [len(self.Q_adj[u]) for u in range(self.num_q)]

        self.anchor_q = self._choose_anchor_query_node_by_uniqueness()

        t0 = time.perf_counter()
        self._prepare_prefix_bound_layout()
        t1 = time.perf_counter()
        print(f"[Time] Prepare prefix-bound layout: {t1 - t0:.4f} s")
        print(f"[Info] prefix lengths: m1={self.m1}, m2={self.m2}")

        t2 = time.perf_counter()
        self.initial_candidates = self._build_initial_candidates_for_anchor_only()
        t3 = time.perf_counter()
        print(f"[Time] Build anchor candidates (prefix-bound): {t3 - t2:.4f} s")

    @staticmethod
    def _build_adj(edge_index: torch.Tensor, num_nodes: int) -> List[Set[int]]:
        adj = [set() for _ in range(num_nodes)]
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for u, v in zip(src, dst):
            if u == v:
                continue
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def _choose_anchor_query_node_by_uniqueness(self) -> int:
        """
        Choose the query node with minimum summed similarity to other query nodes.
        Tie-break:
          1) larger degree first
          2) smaller id
        """
        sim_sums = []

        for q_u in range(self.num_q):
            s = 0.0
            for q_v in range(self.num_q):
                if q_v == q_u:
                    continue
                s += float(torch.dot(self.Q_x[q_u], self.Q_x[q_v]))
            sim_sums.append(s)

        anchor = min(
            range(self.num_q),
            key=lambda q_u: (sim_sums[q_u], -self.Q_deg[q_u], q_u)
        )

        print("[Anchor selection]")
        for q_u in range(self.num_q):
            print(f"  q{q_u}: sim_sum={sim_sums[q_u]:.6f}, deg={self.Q_deg[q_u]}")
        print(f"  chosen anchor = q{anchor}")

        return anchor

    def _prepare_prefix_bound_layout(self):
        """
        Prepare a fixed global dimension order and corresponding data layout.

        Global dimension order:
          descending variance over data vectors

        Stored tensors:
          G_ord          : data vectors reordered by the global order
          G_prefix1      : first m1 dims
          G_prefix2      : first m2 dims
          G_tailnorm1    : L2 norm of dims [m1:]
          G_tailnorm2    : L2 norm of dims [m2:]
          G_deg_t        : degree tensor
        """
        var = torch.var(self.G_x, dim=0, unbiased=False)
        self.global_dim_order = torch.argsort(var, descending=True)

        self.G_ord = self.G_x[:, self.global_dim_order].contiguous()
        self.G_prefix1 = self.G_ord[:, :self.m1].contiguous()
        self.G_prefix2 = self.G_ord[:, :self.m2].contiguous()

        if self.m1 < self.feat_dim:
            self.G_tailnorm1 = torch.linalg.norm(self.G_ord[:, self.m1:], dim=1)
        else:
            self.G_tailnorm1 = torch.zeros(self.num_g, dtype=self.G_x.dtype, device=self.G_x.device)

        if self.m2 < self.feat_dim:
            self.G_tailnorm2 = torch.linalg.norm(self.G_ord[:, self.m2:], dim=1)
        else:
            self.G_tailnorm2 = torch.zeros(self.num_g, dtype=self.G_x.dtype, device=self.G_x.device)

        self.G_deg_t = torch.tensor(self.G_deg, dtype=torch.long, device=self.G_x.device)

    def _anchor_exact_reporting_prefix_bound(self, q_u: int) -> Set[int]:
        """
        Exact threshold reporting for the anchor query vertex.

        Pipeline:
          1) degree filter
          2) stage-1 prefix upper bound
          3) stage-2 prefix upper bound (optional if m2 > m1)
          4) exact full-dot on survivors

        Upper bound:
          dot(q, x) <= dot(q_prefix, x_prefix) + ||q_tail|| * ||x_tail||
        """
        q_deg = self.Q_deg[q_u]
        q_ord = self.Q_x[q_u][self.global_dim_order]

        q_prefix1 = q_ord[:self.m1]
        q_tailnorm1 = float(torch.linalg.norm(q_ord[self.m1:])) if self.m1 < self.feat_dim else 0.0

        q_prefix2 = q_ord[:self.m2]
        q_tailnorm2 = float(torch.linalg.norm(q_ord[self.m2:])) if self.m2 < self.feat_dim else 0.0

        # ---- stage 0: degree filter ----
        active = self.G_deg_t >= q_deg
        active_idx = torch.nonzero(active, as_tuple=True)[0]

        print(f"[AnchorReport] degree survivors = {active_idx.numel()} / {self.num_g}")

        if active_idx.numel() == 0:
            return set()

        # ---- stage 1: prefix upper bound ----
        p1 = self.G_prefix1[active_idx] @ q_prefix1
        ub1 = p1 + q_tailnorm1 * self.G_tailnorm1[active_idx]
        keep1 = ub1 >= self.tau
        idx1 = active_idx[keep1]

        print(f"[AnchorReport] stage1 survivors = {idx1.numel()}")

        if idx1.numel() == 0:
            return set()

        # ---- stage 2: tighter prefix upper bound ----
        if self.m2 > self.m1:
            p2 = self.G_prefix2[idx1] @ q_prefix2
            ub2 = p2 + q_tailnorm2 * self.G_tailnorm2[idx1]
            keep2 = ub2 >= self.tau
            idx2 = idx1[keep2]
        else:
            idx2 = idx1

        print(f"[AnchorReport] stage2 survivors = {idx2.numel()}")

        if idx2.numel() == 0:
            return set()

        # ---- exact refinement ----
        full_scores = self.G_x[idx2] @ self.Q_x[q_u]
        keep_exact = full_scores >= self.tau
        final_idx = idx2[keep_exact]

        print(f"[AnchorReport] exact survivors = {final_idx.numel()}")

        return set(final_idx.tolist())

    def _build_initial_candidates_for_anchor_only(self) -> Dict[int, Set[int]]:
        candidates: Dict[int, Set[int]] = {q_u: set() for q_u in range(self.num_q)}
        candidates[self.anchor_q] = self._anchor_exact_reporting_prefix_bound(self.anchor_q)
        return candidates

    def _sim(self, q_u: int, g_v: int) -> float:
        return float(torch.dot(self.Q_x[q_u], self.G_x[g_v]))

    def _generate_local_candidates(
        self,
        q_u: int,
        mapping: Dict[int, int],
        used_g_nodes: Set[int],
    ) -> Set[int]:
        matched_neighbors = [q_n for q_n in self.Q_adj[q_u] if q_n in mapping]

        if len(matched_neighbors) == 0:
            return set()

        first_qn = matched_neighbors[0]
        candidate_region = set(self.G_adj[mapping[first_qn]])

        for q_n in matched_neighbors[1:]:
            candidate_region &= self.G_adj[mapping[q_n]]

        candidate_region -= used_g_nodes

        cur = set()
        for g_v in candidate_region:
            if self.G_deg[g_v] < self.Q_deg[q_u]:
                continue
            if self._sim(q_u, g_v) < self.tau:
                continue

            feasible = True
            for q_m, g_m in mapping.items():
                if q_m in self.Q_adj[q_u] and g_m not in self.G_adj[g_v]:
                    feasible = False
                    break

            if feasible:
                cur.add(g_v)

        return cur

    def _select_next_query_node(
        self,
        unmatched_q_nodes: Set[int],
        candidates: Dict[int, Set[int]],
        mapping: Dict[int, int],
    ) -> int:
        if self.anchor_q in unmatched_q_nodes:
            return self.anchor_q

        expandable = []
        for q_u in unmatched_q_nodes:
            matched_nbr_cnt = sum(1 for q_n in self.Q_adj[q_u] if q_n in mapping)
            if matched_nbr_cnt > 0:
                expandable.append(q_u)

        if expandable:
            return min(
                expandable,
                key=lambda q_u: (
                    len(candidates[q_u]) if len(candidates[q_u]) > 0 else float("inf"),
                    -sum(1 for q_n in self.Q_adj[q_u] if q_n in mapping),
                    -self.Q_deg[q_u],
                    q_u,
                )
            )

        return min(unmatched_q_nodes, key=lambda q_u: (-self.Q_deg[q_u], q_u))

    def _is_locally_feasible(
        self,
        q_u: int,
        g_v: int,
        mapping: Dict[int, int],
        used_g_nodes: Set[int],
    ) -> bool:
        if g_v in used_g_nodes:
            return False

        if self._sim(q_u, g_v) < self.tau:
            return False

        for q_w, g_w in mapping.items():
            if q_w in self.Q_adj[q_u] and g_w not in self.G_adj[g_v]:
                return False

        return True

    def _forward_check(
        self,
        q_u: int,
        g_v: int,
        mapping: Dict[int, int],
        used_g_nodes: Set[int],
        candidates: Dict[int, Set[int]],
        unmatched_q_nodes: Set[int],
    ) -> Dict[int, Set[int]] | None:
        new_candidates: Dict[int, Set[int]] = {}
        for q_w in range(self.num_q):
            new_candidates[q_w] = set(candidates[q_w])

        new_candidates[q_u] = {g_v}

        new_mapping = dict(mapping)
        new_mapping[q_u] = g_v

        new_used = set(used_g_nodes)
        new_used.add(g_v)

        for q_w in unmatched_q_nodes:
            if q_w == q_u:
                continue

            if len(new_candidates[q_w]) == 0:
                if any(q_n in new_mapping for q_n in self.Q_adj[q_w]):
                    new_candidates[q_w] = self._generate_local_candidates(
                        q_u=q_w,
                        mapping=new_mapping,
                        used_g_nodes=new_used,
                    )
                    if len(new_candidates[q_w]) == 0:
                        return None
                    continue

            cur = new_candidates[q_w]
            cur = cur - new_used

            if q_w in self.Q_adj[q_u]:
                cur = cur & self.G_adj[g_v]

            for q_m, g_m in new_mapping.items():
                if q_m == q_w:
                    continue
                if q_m in self.Q_adj[q_w]:
                    cur = cur & self.G_adj[g_m]

            if len(cur) == 0:
                return None

            new_candidates[q_w] = cur

        return new_candidates

    def find_all_matches(self, max_matches: int | None = None):
        if len(self.initial_candidates[self.anchor_q]) == 0:
            return []

        all_matches = []

        def backtrack(
            mapping: Dict[int, int],
            used_g_nodes: Set[int],
            candidates: Dict[int, Set[int]],
            unmatched_q_nodes: Set[int],
        ):
            if max_matches is not None and len(all_matches) >= max_matches:
                return

            if len(unmatched_q_nodes) == 0:
                node_similarities = {
                    q_u: self._sim(q_u, g_v)
                    for q_u, g_v in mapping.items()
                }
                sum_similarity = sum(node_similarities.values())
                avg_similarity = sum_similarity / len(node_similarities)

                all_matches.append({
                    "mapping": dict(mapping),
                    "node_similarities": node_similarities,
                    "sum_similarity": sum_similarity,
                    "avg_similarity": avg_similarity,
                })
                return

            q_u = self._select_next_query_node(
                unmatched_q_nodes=unmatched_q_nodes,
                candidates=candidates,
                mapping=mapping,
            )

            if q_u != self.anchor_q and len(candidates[q_u]) == 0:
                generated = self._generate_local_candidates(
                    q_u=q_u,
                    mapping=mapping,
                    used_g_nodes=used_g_nodes,
                )
                if len(generated) == 0:
                    return

                candidates = {k: set(v) for k, v in candidates.items()}
                candidates[q_u] = generated

            cand_list = sorted(
                candidates[q_u],
                key=lambda g_v: self._sim(q_u, g_v),
                reverse=True,
            )

            for g_v in cand_list:
                if not self._is_locally_feasible(q_u, g_v, mapping, used_g_nodes):
                    continue

                new_candidates = self._forward_check(
                    q_u=q_u,
                    g_v=g_v,
                    mapping=mapping,
                    used_g_nodes=used_g_nodes,
                    candidates=candidates,
                    unmatched_q_nodes=unmatched_q_nodes,
                )
                if new_candidates is None:
                    continue

                new_mapping = dict(mapping)
                new_mapping[q_u] = g_v

                new_used = set(used_g_nodes)
                new_used.add(g_v)

                new_unmatched = set(unmatched_q_nodes)
                new_unmatched.remove(q_u)

                backtrack(
                    mapping=new_mapping,
                    used_g_nodes=new_used,
                    candidates=new_candidates,
                    unmatched_q_nodes=new_unmatched,
                )

        backtrack(
            mapping={},
            used_g_nodes=set(),
            candidates={q_u: set(cands) for q_u, cands in self.initial_candidates.items()},
            unmatched_q_nodes=set(range(self.num_q)),
        )

        all_matches.sort(key=lambda x: x["avg_similarity"], reverse=True)
        return all_matches



def save_query(
    query_graph: Data,
    gt_mapping: dict,
    query_path: str,
    metadata: dict = None,
):
    os.makedirs(os.path.dirname(query_path), exist_ok=True)
    payload = {
        "x": query_graph.x.cpu(),
        "edge_index": query_graph.edge_index.cpu(),
        "gt_mapping": gt_mapping,
        "metadata": metadata or {},
    }
    torch.save(payload, query_path)



def load_query(query_path: str):
    payload = torch.load(query_path, weights_only=False)
    query_graph = Data(
        x=payload["x"],
        edge_index=payload["edge_index"],
    )
    gt_mapping = payload["gt_mapping"]
    metadata = payload.get("metadata", {})
    return query_graph, gt_mapping, metadata



def get_or_create_query(
    data_graph: Data,
    query_path: str,
    dataset_name: str,
    num_query_nodes: int = 4,
    noise_std: float = 0.0,
    seed: int = 42,
):
    if os.path.exists(query_path):
        print(f"Loading existing query from: {query_path}")
        query_graph, gt_mapping, metadata = load_query(query_path)
        created = False
    else:
        print(f"Query file not found. Sampling and saving to: {query_path}")
        query_graph, gt_mapping = sample_query_fast(
            data_graph=data_graph,
            num_query_nodes=num_query_nodes,
            noise_std=noise_std,
            seed=seed,
        )
        metadata = {
            "dataset_name": dataset_name,
            "num_query_nodes": num_query_nodes,
            "noise_std": noise_std,
            "seed": seed,
        }
        save_query(query_graph, gt_mapping, query_path, metadata=metadata)
        created = True

    return query_graph, gt_mapping, metadata, created



def is_nonzero_feature(x: torch.Tensor, eps: float = 1e-12) -> bool:
    return float(torch.norm(x)) > eps


def sample_query_fast(
    data_graph: Data,
    num_query_nodes: int = 4,
    noise_std: float = 0.0,
    seed: int = 42,
    num_hops: int = 2,
    max_tries: int = 20,
):
    random.seed(seed)
    torch.manual_seed(seed)

    num_nodes = data_graph.x.size(0)
    edge_index = data_graph.edge_index

    # ===== 在这里先筛掉零向量节点 =====
    valid_nodes = [i for i in range(num_nodes) if is_nonzero_feature(data_graph.x[i])]
    if len(valid_nodes) < num_query_nodes:
        raise RuntimeError("Not enough non-zero-feature nodes to sample a query graph.")

    for _ in range(max_tries):
        # ===== 起点只从 valid_nodes 里选 =====
        start = random.choice(valid_nodes)

        subset, _, _, _ = k_hop_subgraph(
            node_idx=start,
            num_hops=num_hops,
            edge_index=edge_index,
            relabel_nodes=False,
        )

        if subset.numel() < num_query_nodes:
            continue

        subset_list = subset.tolist()
        subset_set = set(subset_list)

        # 只保留局部子图中非零向量的节点
        subset_list = [u for u in subset_list if is_nonzero_feature(data_graph.x[u])]
        subset_set = set(subset_list)

        if len(subset_list) < num_query_nodes:
            continue

        # build local adjacency on original node ids
        local_adj = {u: set() for u in subset_list}
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for u, v in zip(src, dst):
            if u in subset_set and v in subset_set and u != v:
                local_adj[u].add(v)
                local_adj[v].add(u)

        if start not in local_adj or len(local_adj[start]) == 0:
            continue

        # BFS/expansion to guarantee connected chosen nodes
        chosen = []
        visited = set([start])
        queue = [start]

        while queue and len(chosen) < num_query_nodes:
            cur = queue.pop(0)
            chosen.append(cur)

            nbrs = list(local_adj[cur])
            random.shuffle(nbrs)
            for nb in nbrs:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

        if len(chosen) < num_query_nodes:
            continue

        chosen = chosen[:num_query_nodes]
        chosen_tensor = torch.tensor(chosen, dtype=torch.long)

        query_edge_index, _ = subgraph(
            subset=chosen_tensor,
            edge_index=edge_index,
            relabel_nodes=True,
        )

        if query_edge_index.size(1) == 0:
            continue

        query_x = data_graph.x[chosen_tensor].clone().float()
        if noise_std > 0:
            query_x = query_x + noise_std * torch.randn_like(query_x)

        query_graph = Data(x=query_x, edge_index=query_edge_index)
        gt_mapping = {new: old for new, old in enumerate(chosen)}
        return query_graph, gt_mapping

    raise RuntimeError("Failed to sample a connected query graph with non-zero node features.")




def pretty_print_graph_stats(name: str, graph):
    print(f"{name}:")
    if getattr(graph, "x", None) is not None:
        print(f"  num_nodes = {graph.x.size(0)}")
        print(f"  feat_dim  = {graph.x.size(1)}")
    else:
        if hasattr(graph, "num_nodes") and graph.num_nodes is not None:
            print(f"  num_nodes = {graph.num_nodes}")
        else:
            print("  num_nodes = Unknown")
        print("  feat_dim  = None")
    print(f"  num_edges = {graph.edge_index.size(1)}")


def pretty_print_matches(matches, max_show: int = 10):
    if not matches:
        print("No exact semantic match found.")
        return

    print(f"Found {len(matches)} match(es).")
    show_n = min(len(matches), max_show)

    for i in range(show_n):
        match_info = matches[i]
        mapping = match_info["mapping"]
        node_similarities = match_info["node_similarities"]

        print(f"\nMatch {i + 1}:")
        for q_u, g_v in sorted(mapping.items()):
            print(f"  q{q_u} -> g{g_v}, sim = {node_similarities[q_u]:.6f}")
        print(f"  sum similarity     = {match_info['sum_similarity']:.6f}")
        print(f"  average similarity = {match_info['avg_similarity']:.6f}")

    if len(matches) > max_show:
        print(f"\n... {len(matches) - max_show} more not shown")


def ground_truth_in_matches(
    gt_mapping: Dict[int, int],
    matches,
) -> bool:
    gt_items = sorted(gt_mapping.items())
    for m in matches:
        if sorted(m["mapping"].items()) == gt_items:
            return True
    return False



from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import AttributedGraphDataset   #JODIEDataset 4 \\DGraphFin 1 \\EllipticBitcoinTemporalDataset 1

#EllipticBitcoinDataset 1 \\PolBlogs 1  \\Twitch 6 \\GitHub 1 \\Actor 1 
#HeterophilousGraphDataset 5\\ AttributedGraphDataset 10
#WebKB 3 samll

def load_dataset(dataset_type: str, d_name: str, root: str):
    """
    Unified loader for:
      - ogb
      - planetoid
      - attributed

    Returns:
      dataset, data_graph, train_idx, valid_idx, test_idx
    """
    dataset_type = dataset_type.lower()

    if dataset_type == "ogb":
        from ogb.nodeproppred import PygNodePropPredDataset

        dataset = PygNodePropPredDataset(name=d_name, root=root)
        split_idx = dataset.get_idx_split()
        train_idx = split_idx["train"]
        valid_idx = split_idx["valid"]
        test_idx = split_idx["test"]
        data_graph = dataset[0]

    elif dataset_type == "planetoid":
        from torch_geometric.datasets import Planetoid

        dataset = Planetoid(root=root, name=d_name)
        data_graph = dataset[0]

        train_idx = data_graph.train_mask.nonzero(as_tuple=True)[0] \
            if hasattr(data_graph, "train_mask") and data_graph.train_mask is not None else None
        valid_idx = data_graph.val_mask.nonzero(as_tuple=True)[0] \
            if hasattr(data_graph, "val_mask") and data_graph.val_mask is not None else None
        test_idx = data_graph.test_mask.nonzero(as_tuple=True)[0] \
            if hasattr(data_graph, "test_mask") and data_graph.test_mask is not None else None

    elif dataset_type == "attributed":
        from torch_geometric.datasets import AttributedGraphDataset

        dataset = AttributedGraphDataset(root=root, name=d_name)
        data_graph = dataset[0]

        # usually no official split is provided
        train_idx = None
        valid_idx = None
        test_idx = None

    else:
        raise ValueError(
            f"Unknown dataset_type: {dataset_type}. "
            f"Expected one of ['ogb', 'planetoid', 'attributed']."
        )

    return dataset, data_graph, train_idx, valid_idx, test_idx


def debug_ground_truth_candidates(matcher, gt_mapping):
    print("\n[Ground-truth diagnostic]")
    for q_u, g_v in sorted(gt_mapping.items()):
        sim = matcher._sim(q_u, g_v)
        q_deg = matcher.Q_deg[q_u]
        g_deg = matcher.G_deg[g_v]
        in_candidates = g_v in matcher.initial_candidates[q_u]
        print(
            f"q{q_u} -> g{g_v} | sim={sim:.6f} | "
            f"deg_q={q_deg} | deg_g={g_deg} | in_candidates={in_candidates}"
        )



def debug_anchor_candidates(matcher):
    print("\n[Anchor diagnostic]")
    print(f"anchor query node = q{matcher.anchor_q}")
    for q_u in range(matcher.num_q):
        print(f"  q{q_u}: initial_cand_size = {len(matcher.initial_candidates[q_u])}")




def main():
    # =========================================================
    # Config
    # =========================================================
    # dataset_type: "ogb", "planetoid", "attributed"
    dataset_type = "ogb"#"attributed"

    # Examples:
    #   OGB:        d_name = "ogbn-products", root = "dataset/ogb"
    #   Planetoid:  d_name = "Cora", root = "dataset/planetoid"
    #   Attributed: d_name = "Wiki", root = "dataset/attributed"
    d_name = "ogbn-products"#"CiteSeer" #"Cora" #"MAG"#"TWeibo" #"Facebook"#"PPI" #"BlogCatalog" #"PubMed" #"Wiki" #"ogbn-products"
    root = "dataset/ogb"

    tau = 0.9
    num_query_nodes = 5
    noise_std = 0.0
    seed = 42

    # Use None for no limit
    max_matches = None

    query_dir = "query"
    query_filename = f"{dataset_type}_{d_name}_q{num_query_nodes}_seed{seed}_noise{noise_std}.pt"
    query_path = os.path.join(query_dir, query_filename)

    total_start = time.perf_counter()

    # =========================================================
    # 1) Load dataset
    # =========================================================
    t0 = time.perf_counter()
    print("Loading dataset ...")

    dataset, data_graph, train_idx, valid_idx, test_idx = load_dataset(
        dataset_type=dataset_type,
        d_name=d_name,
        root=root,
    )

    t1 = time.perf_counter()

    print("Dataset loaded.")
    if train_idx is not None and valid_idx is not None and test_idx is not None:
        print(f"train/valid/test = {len(train_idx)}/{len(valid_idx)}/{len(test_idx)}")
    else:
        print("train/valid/test = N/A")

    pretty_print_graph_stats("Data graph", data_graph)
    print(f"[Time] Dataset loading: {t1 - t0:.4f} s")

    # Safety check
    if getattr(data_graph, "x", None) is None:
        raise RuntimeError(
            f"Dataset '{d_name}' does not provide node features in data_graph.x. "
            f"This baseline requires node vectors."
        )

    # =========================================================
    # 2) Prepare query graph
    # =========================================================
    t2 = time.perf_counter()
    print("\nPreparing query graph ...")

    query_graph, gt_mapping, metadata, created = get_or_create_query(
        data_graph=data_graph,
        query_path=query_path,
        dataset_name=d_name,
        num_query_nodes=num_query_nodes,
        noise_std=noise_std,
        seed=seed,
    )

    t3 = time.perf_counter()

    pretty_print_graph_stats("Query graph", query_graph)
    print("Ground-truth mapping (query node -> data node):")
    print(dict(sorted(gt_mapping.items())))
    if created:
        print(f"Query was newly sampled and saved to: {query_path}")
    else:
        print(f"Query was loaded from: {query_path}")
    print(f"[Time] Query preparation: {t3 - t2:.4f} s")

    # =========================================================
    # 3) Initialize matcher
    # =========================================================
    t4 = time.perf_counter()
    print("\nInitializing matcher ...")

    matcher = AnchorFirstPrefixBoundMatcher(
        data_graph=data_graph,
        query_graph=query_graph,
        tau=tau,
        m1=16,
        m2=32,
    )

    t5 = time.perf_counter()

    print("Initial candidate set sizes:")
    for u in range(matcher.num_q):
        print(f"  q{u}: {len(matcher.initial_candidates[u])}")
    print(f"[Time] Matcher initialization: {t5 - t4:.4f} s")

    debug_ground_truth_candidates(matcher, gt_mapping)

    debug_anchor_candidates(matcher)

    # =========================================================
    # 4) Run matching
    # =========================================================
    t6 = time.perf_counter()
    print("\nRunning exact semantic subgraph matching ...")

    matches = matcher.find_all_matches(max_matches=max_matches)

    t7 = time.perf_counter()

    print()
    pretty_print_matches(matches, max_show=20)

    print("\nGround-truth recovered?")
    print(ground_truth_in_matches(gt_mapping, matches))
    print(f"[Time] Matching search: {t7 - t6:.4f} s")

    # =========================================================
    # Runtime summary
    # =========================================================
    total_end = time.perf_counter()
    print("\n========== Runtime Summary ==========")
    print(f"Dataset loading        : {t1 - t0:.4f} s")
    print(f"Query preparation      : {t3 - t2:.4f} s")
    print(f"Matcher initialization : {t5 - t4:.4f} s")
    print(f"Matching search        : {t7 - t6:.4f} s")
    print(f"Total runtime          : {total_end - total_start:.4f} s")



if __name__ == "__main__":
    main()