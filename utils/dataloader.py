"""
Data loader class for graph datasets.

This module provides a DataLoader class that handles dataset loading,
preprocessing, and ground truth computation for hypothesis testing.
"""

import os
import time
from typing import Optional

import networkx as nx
import pandas as pd
from littleballoffur import RandomWalkSampler

from dataprep.citation_prep import citation_prep
from dataprep.movielens_prep import movielens_prep
from dataprep.yelp_prep import yelp_prep


class DataLoader:
    """
    Data loader class for graph datasets.

    Handles dataset loading, preprocessing, graph analysis, and ground truth computation.
    """

    def __init__(self, args):
        """
        Initialize the data loader.

        Args:
            args: Arguments object containing configuration
        """
        self.args = args
        self.graph = None
        self.preparation_time = None

    def load(self) -> nx.Graph:
        """
        Load and prepare the dataset.

        Returns:
            networkx.Graph: Prepared graph ready for hypothesis testing
        """
        time_dataset_prep = time.time()

        # Set dataset path
        self.args.dataset_path = os.path.join(
            os.getcwd(), "datasets", self.args.dataset
        )

        # Load the graph
        self.graph = self._load_dataset()

        # Dataset-specific preprocessing
        if self.args.dataset == "citation":
            self._preprocess_citation_dataset()

        # Log preparation time
        self.preparation_time = round((time.time() - time_dataset_prep), 2)
        print(
            f">>> Total time for dataset {self.args.dataset} preparation is {self.preparation_time}."
        )
        self.args.logger.info(
            f">>> Total time for dataset {self.args.dataset} preparation is {self.preparation_time}."
        )

        # Set graph characteristics
        self._set_graph_characteristics()

        return self.graph

    def _load_dataset(self) -> nx.Graph:
        """
        Dataset loader dispatcher.

        Loads the graph using dataset-specific preparators based on args.dataset.

        Returns:
            networkx.Graph: Loaded graph

        Raises:
            AssertionError: If hypothesis_pattern is None
            Exception: If dataset is not supported
        """
        assert (
            self.args.hypothesis_pattern is not None
        ), f"args.hypothesis_pattern should not be None."

        if self.args.dataset == "movielens":
            return movielens_prep(self.args)
        elif self.args.dataset == "yelp":
            return yelp_prep(self.args)
        elif self.args.dataset == "citation":
            return citation_prep(self.args)
        else:
            self.args.logger.error(f"Sorry, we don't support {self.args.dataset}.")
            raise Exception(f"Sorry, we don't support {self.args.dataset}.")

    def _preprocess_citation_dataset(self):
        """
        Preprocess citation dataset by sampling a subset for demonstration.

        Samples 1000 nodes using RandomWalkSampler and re-indexes to continuous integers.
        """
        original_num_nodes = self.graph.number_of_nodes()
        original_num_edges = self.graph.number_of_edges()
        self.args.logger.info(
            f"Original citation network has {original_num_nodes} nodes and {original_num_edges} edges."
        )
        print(
            f"Original citation network has {original_num_nodes} nodes and {original_num_edges} edges."
        )

        # Sample 1000 nodes using RandomWalkSampler for demonstration
        sample_size = 1000
        self.args.logger.info(
            f"Sampling {sample_size} nodes from citation network using RandomWalkSampler for demonstration..."
        )
        print(
            f"Sampling {sample_size} nodes from citation network using RandomWalkSampler for demonstration..."
        )

        sampler = RandomWalkSampler(
            number_of_nodes=sample_size, seed=int(self.args.seed)
        )
        self.graph = sampler.sample(self.graph)

        # Re-index nodes to ensure continuous integer indices (0, 1, 2, ...)
        # This is required by littleballoffur library
        self.graph = nx.relabel.convert_node_labels_to_integers(
            self.graph, first_label=0, ordering="default"
        )

        sampled_num_nodes = self.graph.number_of_nodes()
        sampled_num_edges = self.graph.number_of_edges()
        self.args.logger.info(
            f"Sampled graph has {sampled_num_nodes} nodes and {sampled_num_edges} edges."
        )
        print(
            f"Sampled graph has {sampled_num_nodes} nodes and {sampled_num_edges} edges."
        )

        # Analyze graph (use custom function if provided, otherwise use built-in method)
        self.analyze_graph(self.args.hypothesis_pattern)

    def _set_graph_characteristics(self):
        """
        Set graph characteristics on args object.

        Sets: num_nodes, num_edges, and logs graph properties.
        """
        self.args.num_nodes = self.graph.number_of_nodes()
        self.args.num_edges = self.graph.number_of_edges()
        is_directed = self.graph.is_directed()

        self.args.logger.info(
            f"{self.args.dataset} has {self.args.num_nodes} nodes and {self.args.num_edges} edges."
        )
        self.args.logger.info(f"{self.args.dataset} is directed: {is_directed}")
        self.args.logger.info(
            f"{self.args.dataset} is connected: {nx.is_connected(self.graph)}."
        )

        print(
            f"{self.args.dataset} graph type: {'Directed' if is_directed else 'Undirected'}"
        )
        print(f"{self.args.dataset} is connected: {nx.is_connected(self.graph)}")

    def get_graph(self) -> Optional[nx.Graph]:
        """
        Get the loaded graph.

        Returns:
            networkx.Graph or None: The loaded graph, or None if not loaded yet
        """
        return self.graph

    def get_preparation_time(self) -> Optional[float]:
        """
        Get the time taken for dataset preparation.

        Returns:
            float or None: Preparation time in seconds, or None if not loaded yet
        """
        return self.preparation_time

    def analyze_graph(self, hypothesis_pattern):
        """
        Analyze the graph by keeping only nodes and edges that match the hypothesis pattern.

        This method filters the graph to keep only nodes and edges that match
        the specified hypothesis pattern. It modifies self.graph in place.

        Args:
            hypothesis_pattern: The hypothesis pattern to analyze

        Returns:
            None (modifies self.graph in place)

        Raises:
            ValueError: If pattern is invalid or unsupported
        """
        if len(list(hypothesis_pattern)) != 1:
            raise ValueError("analyze_graph only supports one hypothesis pattern")

        # Extract pattern from hypothesis_pattern
        pattern_key = list(hypothesis_pattern.keys())[0]
        pattern = hypothesis_pattern[pattern_key]

        if pattern.get("type") != "subgraph":
            raise ValueError("analyze_graph only supports subgraph patterns")

        if "subgraph" not in pattern:
            raise ValueError("Subgraph pattern must contain 'subgraph' key")

        subgraph_pattern = pattern["subgraph"]
        pattern_nodes = subgraph_pattern["nodes"]
        pattern_edges = subgraph_pattern["edges"]

        # Helper function to check if a graph node matches a pattern node
        def node_matches_pattern(graph_node_data, pattern_node_info):
            """
            Check if a graph node matches a pattern node (label and attributes).

            Args:
                graph_node_data: Dictionary of graph node attributes
                pattern_node_info: Dictionary from pattern with 'label' and 'attribute' keys

            Returns:
                True if node matches, False otherwise
            """
            pattern_label = pattern_node_info.get("label")
            pattern_attrs = pattern_node_info.get("attribute", {})

            # Check node label
            if graph_node_data.get("label") != pattern_label:
                return False

            # Check node attributes
            for attr_key, attr_val in pattern_attrs.items():
                graph_val = graph_node_data.get(attr_key)
                if pd.isna(graph_val):
                    return False
                # Support vague match for string attributes (like citation dataset)
                if graph_val != attr_val and (
                    attr_val not in graph_val if isinstance(graph_val, str) else True
                ):
                    return False

            return True

        # Extract edge patterns: (from_label, to_label) pairs
        # Create a mapping from pattern node id to label
        node_id_to_label = {}
        for node_info in pattern_nodes:
            node_id = node_info.get("id")
            label = node_info.get("label")
            if node_id and label:
                node_id_to_label[node_id] = label

        # Extract allowed edge patterns
        allowed_edge_patterns = set()
        for edge_info in pattern_edges:
            from_id = edge_info.get("from")
            to_id = edge_info.get("to")
            if from_id in node_id_to_label and to_id in node_id_to_label:
                from_label = node_id_to_label[from_id]
                to_label = node_id_to_label[to_id]
                allowed_edge_patterns.add((from_label, to_label))
                # For undirected graphs, also allow reverse direction
                if not self.graph.is_directed():
                    allowed_edge_patterns.add((to_label, from_label))

        # Filter nodes: keep only nodes that match pattern nodes (label AND attributes)
        nodes_to_keep = []
        for node in self.graph.nodes():
            graph_node_data = self.graph.nodes[node]
            # Check if this node matches any pattern node
            for pattern_node_info in pattern_nodes:
                if node_matches_pattern(graph_node_data, pattern_node_info):
                    nodes_to_keep.append(node)
                    break  # Node matches at least one pattern, no need to check others

        # Create filtered subgraph with allowed nodes
        filtered_graph = self.graph.subgraph(nodes_to_keep).copy()

        # Filter edges: keep only edges that match allowed edge patterns
        edges_to_remove = []
        for u, v in filtered_graph.edges():
            u_label = filtered_graph.nodes[u].get("label")
            v_label = filtered_graph.nodes[v].get("label")
            edge_pattern = (u_label, v_label)
            if edge_pattern not in allowed_edge_patterns:
                edges_to_remove.append((u, v))

        # Remove edges that don't match the pattern
        filtered_graph.remove_edges_from(edges_to_remove)

        # Update self.graph with filtered graph
        self.graph = filtered_graph

        # Print analysis results
        print(f"Number of nodes in the graph: {self.graph.number_of_nodes()}")
        print(f"Number of edges in the graph: {self.graph.number_of_edges()}")
        print(
            f"Number of connected components in the graph: {nx.number_connected_components(self.graph)}"
        )
        print("--------------------------------")
