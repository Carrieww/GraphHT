"""
Anchor-based subgraph sampler for uniform subgraph sampling.

This module implements an anchor-based sampling strategy that ensures
each subgraph instance is sampled with equal probability.
"""

import random
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from littleballoffur.sampler import Sampler


def extract_anchor_substructure(pattern: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract anchor substructure from pattern for enumeration.

    Args:
        pattern: Hypothesis pattern containing anchor information

    Returns:
        dict with 'nodes' and 'edges' keys representing the anchor substructure
    """
    anchor = pattern["anchor"]

    if anchor["type"] == "node":
        return {"nodes": [anchor["id"]], "edges": []}
    elif anchor["type"] == "edge":
        return {"nodes": anchor["ids"], "edges": [tuple(anchor["ids"])]}
    elif anchor["type"] == "subgraph":
        return {"nodes": anchor["ids"], "edges": pattern["subgraph"]["edges"]}
    else:
        raise ValueError(f"Unknown anchor type: {anchor['type']}")


class SubgraphSampler(Sampler):
    """
    Base class for anchor-based subgraph samplers.

    The sampling proceeds in two stages:
    1. Anchor Sampling: Select anchor a with probability proportional to c(a)
    2. Local Completion: Uniformly sample one instance among the c(a) completions

    This ensures each valid subgraph is sampled with equal probability.
    """

    def __init__(
        self,
        pattern: Dict[str, Any],
        number_of_subgraphs: int = 100,
        seed: int = 42,
    ):
        """
        Initialize the subgraph sampler.

        Args:
            pattern: Hypothesis pattern containing subgraph structure and anchor info
            number_of_subgraphs: Number of subgraphs to sample
            seed: Random seed
        """
        super().__init__()
        self.pattern = pattern
        self.number_of_subgraphs = number_of_subgraphs
        self.seed = seed
        self._set_seed()

        # Extract anchor information
        if "anchor" not in pattern:
            raise ValueError("Pattern must contain 'anchor' information.")
        self.anchor = pattern["anchor"]
        self.Pi = self.anchor["Pi"]
        self.anchor_substructure = extract_anchor_substructure(pattern)

        # Extract subgraph structure
        if "subgraph" not in pattern:
            raise ValueError("Pattern must contain 'subgraph' information.")
        self.subgraph_pattern = pattern["subgraph"]
        self.pattern_nodes = self.subgraph_pattern["nodes"]
        self.pattern_edges = self.subgraph_pattern["edges"]

    def _match_node_attributes(
        self, graph: nx.Graph, graph_node: int, pattern_node_info: Dict[str, Any]
    ) -> bool:
        """
        Check if a graph node matches pattern node attributes.

        Args:
            graph: The graph
            graph_node: Node ID in the graph
            pattern_node_info: Pattern node specification with type and attributes

        Returns:
            True if node matches, False otherwise
        """
        label = pattern_node_info.get("label")
        pattern_attrs = pattern_node_info.get("attribute", {})

        # Check node type (label)
        if graph.nodes[graph_node].get("label") != label:
            return False

        # Check node attributes
        for attr_key, attr_val in pattern_attrs.items():
            graph_val = graph.nodes[graph_node].get(attr_key)
            if pd.isna(graph_val):
                return False
            # Support vague match for string attributes
            if graph_val != attr_val and (
                attr_val not in graph_val if isinstance(graph_val, str) else True
            ):
                return False
        return True

    def _enumerate_anchors(self, graph: nx.Graph) -> List[Any]:
        """
        Enumerate all valid anchor positions in the graph.

        This is a base method that should be overridden by subclasses
        for specific anchor types.

        Args:
            graph: The graph to search in

        Returns:
            List of anchor identifiers
        """
        raise NotImplementedError("Subclasses must implement _enumerate_anchors")

    def _count_completions(self, graph: nx.Graph, anchor: Any) -> int:
        """
        Count the number of subgraph completions from a given anchor.

        This is a base method that should be overridden by subclasses.

        Args:
            graph: The graph to search in
            anchor: The anchor identifier

        Returns:
            Number of possible completions from this anchor
        """
        raise NotImplementedError("Subclasses must implement _count_completions")

    def _get_valid_completions(self, graph: nx.Graph, anchor: Any) -> List[Any]:
        """
        Get all valid completions from a given anchor.

        This is a base method that should be overridden by subclasses.
        This method is used to avoid duplicate computation between
        _count_completions and _sample_completion.

        Args:
            graph: The graph to search in
            anchor: The anchor identifier

        Returns:
            List of valid completion identifiers (e.g., node IDs for triangles)
        """
        raise NotImplementedError("Subclasses must implement _get_valid_completions")

    def _sample_completion(self, graph: nx.Graph, anchor: Any) -> Set[int]:
        """
        Sample one subgraph completion from a given anchor.

        This is a base method that should be overridden by subclasses.

        Args:
            graph: The graph to search in
            anchor: The anchor identifier

        Returns:
            Set of node IDs forming the sampled subgraph
        """
        raise NotImplementedError("Subclasses must implement _sample_completion")

    def _sample_completion_cached(
        self, anchor: Any, valid_completions: List[Any]
    ) -> Set[int]:
        """
        Sample one subgraph completion from cached valid completions.

        This method avoids recomputation when valid_completions are already known.
        Subclasses should override this method for their specific completion type.

        Args:
            anchor: The anchor identifier
            valid_completions: Pre-computed list of valid completion identifiers

        Returns:
            Set of node IDs forming the sampled subgraph
        """
        raise NotImplementedError("Subclasses must implement _sample_completion_cached")

    def sample(self, graph: nx.Graph) -> List[Set[int]]:
        """
        Sample subgraphs using anchor-based strategy.

        Args:
            graph: The graph to sample from

        Returns:
            A list of subgraphs, where each subgraph is represented as a set of node IDs
        """
        self._deploy_backend(graph)
        # self._check_number_of_nodes(graph)

        # Stage 1: Enumerate all anchors and compute c(a) for each
        anchors = self._enumerate_anchors(graph)
        if not anchors:
            # Return empty list if no anchors found
            return []

        # Cache valid completions to avoid recomputation in Stage 2
        completion_counts = {}
        completion_cache = {}  # Cache valid_completions for each anchor
        for anchor in anchors:
            valid_completions = self._get_valid_completions(graph, anchor)
            count = len(valid_completions)
            if count > 0:
                completion_counts[anchor] = count
                completion_cache[anchor] = valid_completions

        if not completion_counts:
            # Return empty list if no completions possible
            return []

        # Stage 2: Sample subgraphs
        sampled_subgraphs = []  # List of subgraphs, each subgraph is a set of nodes

        for _ in range(self.number_of_subgraphs):
            # Sample anchor with probability proportional to c(a)
            anchors_list = list(completion_counts.keys())
            weights = list(completion_counts.values())
            total_weight = sum(weights)

            if total_weight == 0:
                break

            probabilities = [w / total_weight for w in weights]
            selected_anchor_idx = np.random.choice(len(anchors_list), p=probabilities)
            selected_anchor = anchors_list[selected_anchor_idx]

            # Sample one completion from the selected anchor using cached completions
            subgraph_nodes = self._sample_completion_cached(
                selected_anchor, completion_cache[selected_anchor]
            )

            # Skip if no valid completion found (should not happen, but handle gracefully)
            if not subgraph_nodes:
                continue

            # Pi = self._compute_Pi(graph, subgraph_nodes)
            if self.Pi <= 0:
                continue

            if np.random.random() > 1 / self.Pi:
                continue

            # Add this triangle as a separate subgraph to the list
            sampled_subgraphs.append(subgraph_nodes)

        return sampled_subgraphs


class TriangleSubgraphSampler(SubgraphSampler):
    """
    Anchor-based sampler for triangle subgraphs.

    Uses edges as anchors. For each edge (u, v), finds all nodes w
    such that (u, w) and (v, w) are edges, forming triangle (u, v, w).
    """

    def __init__(
        self,
        pattern: Dict[str, Any],
        number_of_subgraphs: int = 100,
        seed: int = 42,
    ):
        """
        Initialize triangle subgraph sampler.

        Args:
            pattern: Hypothesis pattern with triangle subgraph structure
            number_of_subgraphs: Number of triangles to sample
            seed: Random seed
        """
        super().__init__(pattern, number_of_subgraphs, seed)

        # For triangle, anchor should be an edge
        if self.anchor["type"] != "edge":
            raise ValueError(
                "TriangleSubgraphSampler requires edge-type anchor. "
                f"Got: {self.anchor['label']}"
            )

        # Extract edge pattern nodes (first two nodes in triangle)
        if len(self.pattern_nodes) < 2:
            raise ValueError(
                "Triangle pattern must have at least 2 nodes for edge anchor."
            )

        # Find which pattern nodes correspond to the anchor edge
        anchor_ids = self.anchor["ids"]
        self.anchor_node_indices = []
        for i, node_info in enumerate(self.pattern_nodes):
            if node_info.get("id") in anchor_ids:
                self.anchor_node_indices.append(i)

        if len(self.anchor_node_indices) != 2:
            raise ValueError(
                f"Anchor edge must match exactly 2 pattern nodes. "
                f"Found: {len(self.anchor_node_indices)}"
            )

        # The third node (completion node) is the remaining one
        all_indices = set(range(len(self.pattern_nodes)))
        completion_indices = all_indices - set(self.anchor_node_indices)
        if len(completion_indices) != 1:
            raise ValueError(
                "Triangle pattern must have exactly 3 nodes. "
                f"Found: {len(self.pattern_nodes)}"
            )
        self.completion_node_index = list(completion_indices)[0]
        self.completion_node_pattern = self.pattern_nodes[self.completion_node_index]

        # Cache anchor node patterns to avoid repeated access in _enumerate_anchors
        self.anchor_node1_pattern = self.pattern_nodes[self.anchor_node_indices[0]]
        self.anchor_node2_pattern = self.pattern_nodes[self.anchor_node_indices[1]]

    def _enumerate_anchors(self, graph: nx.Graph) -> List[Tuple[int, int]]:
        """
        Enumerate all valid edge anchors in the graph.

        An edge (u, v) is valid if both nodes match the anchor pattern nodes.

        Args:
            graph: The graph to search in

        Returns:
            List of valid edge anchors as (u, v) tuples
        """
        anchors = []

        # Use cached anchor node patterns (set in __init__)
        # Enumerate all edges and check if they match anchor pattern
        seen_edges = set()  # Track seen edges to avoid duplicates

        for u, v in graph.edges():
            # Create canonical edge key for undirected graphs
            if graph.is_directed():
                edge_key = (u, v)
            else:
                edge_key = tuple(sorted([u, v]))

            if edge_key in seen_edges:
                continue

            # Check if (u, v) matches anchor pattern
            # For undirected graphs, we need to check both directions because:
            # 1. graph.edges() may return (u, v) or (v, u) for the same edge
            # 2. Anchor pattern has order: [n2, n1] means paper -> fos
            # 3. We need to match: (paper, fos) OR (fos, paper) depending on edge order

            # Forward: u matches first anchor node, v matches second anchor node
            matches_forward = self._match_node_attributes(
                graph, u, self.anchor_node1_pattern
            ) and self._match_node_attributes(graph, v, self.anchor_node2_pattern)

            # Reverse: v matches first anchor node, u matches second anchor node
            # Only needed for undirected graphs where edge direction is ambiguous
            matches_reverse = False
            if not graph.is_directed() and not matches_forward:
                # Only check reverse if forward didn't match (optimization)
                matches_reverse = self._match_node_attributes(
                    graph, v, self.anchor_node1_pattern
                ) and self._match_node_attributes(graph, u, self.anchor_node2_pattern)

            if matches_forward or matches_reverse:
                anchors.append((u, v))
                seen_edges.add(edge_key)

        return anchors

    def _get_valid_completions(
        self, graph: nx.Graph, anchor: Tuple[int, int]
    ) -> List[int]:
        """
        Get all valid triangle completions from an edge anchor.

        This is a helper method to avoid duplicate computation between
        _count_completions and _sample_completion.

        Args:
            graph: The graph to search in
            anchor: Edge anchor as (u, v) tuple

        Returns:
            List of node IDs that can complete the triangle
        """
        u, v = anchor

        # Find common neighbors that match completion node pattern
        u_neighbors = set(graph.neighbors(u))
        v_neighbors = set(graph.neighbors(v))
        common_neighbors = u_neighbors & v_neighbors

        # Filter by completion node pattern
        valid_completions = []
        for w in common_neighbors:
            if self._match_node_attributes(graph, w, self.completion_node_pattern):
                valid_completions.append(w)

        return valid_completions

    def _count_completions(self, graph: nx.Graph, anchor: Tuple[int, int]) -> int:
        """
        Count number of triangle completions from an edge anchor.

        For edge (u, v), finds all nodes w such that:
        - w matches the completion node pattern
        - (u, w) and (v, w) are edges in the graph

        Args:
            graph: The graph to search in
            anchor: Edge anchor as (u, v) tuple

        Returns:
            Number of possible triangle completions
        """
        valid_completions = self._get_valid_completions(graph, anchor)
        return len(valid_completions)

    def _sample_completion(self, graph: nx.Graph, anchor: Tuple[int, int]) -> Set[int]:
        """
        Sample one triangle completion from an edge anchor.

        Args:
            graph: The graph to search in
            anchor: Edge anchor as (u, v) tuple

        Returns:
            Set of node IDs forming the triangle {u, v, w}
        """
        valid_completions = self._get_valid_completions(graph, anchor)
        return self._sample_completion_cached(anchor, valid_completions)

    def _sample_completion_cached(
        self, anchor: Tuple[int, int], valid_completions: List[int]
    ) -> Set[int]:
        """
        Sample one triangle completion from cached valid completions.

        This method avoids recomputation when valid_completions are already known.

        Args:
            anchor: Edge anchor as (u, v) tuple
            valid_completions: Pre-computed list of valid completion nodes

        Returns:
            Set of node IDs forming the triangle {u, v, w}
        """
        u, v = anchor

        if not valid_completions:
            # Should not happen if _count_completions was called first
            # This indicates an inconsistency - return empty set to indicate failure
            # The caller should handle this case
            raise ValueError("No valid completions found for anchor.")

        # Uniformly sample one completion
        w = random.choice(valid_completions)

        return {u, v, w}
