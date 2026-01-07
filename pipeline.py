"""
Hypothesis Testing Pipeline.

This module provides the main pipeline for graph sampling, instance extraction,
and hypothesis testing.
"""

import time
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx
from littleballoffur import (
    CommonNeighborAwareRandomWalkSampler,
    DegreeBasedSampler,
    ForestFireSampler,
    MetropolisHastingsRandomWalkSampler,
    NonBackTrackingRandomWalkSampler,
    PageRankBasedSampler,
    RandomEdgeSampler,
    RandomEdgeSamplerWithInduction,
    RandomNodeEdgeSampler,
    RandomNodeNeighborSampler,
    RandomNodeSampler,
    RandomWalkSampler,
    RandomWalkWithRestartSampler,
    ShortestPathSampler,
    SnowBallSampler,
)

from hypothesis_testing import hypothesis_testing
from instance_extraction import extract_attributes, extract_instances
from samplers.existing_graph_samplers import (
    CommonNeighborAwareRandomWalkSampler_new,
    CommunityStructureExpansionSampler_new,
    FrontierSampler_new,
    RandomEdgeSampler_new,
    RandomNodeEdgeSampler_new,
    RandomWalkWithRestartSampler_new,
    SnowBallSampler_new,
)
from samplers.phase import PHASE
from samplers.phase_opt import Opt_PHASE
from samplers.subgraph_sampler import TriangleSubgraphSampler


class HypothesisTestingPipeline:
    """
    Pipeline for graph sampling, instance extraction, and hypothesis testing.

    This class orchestrates the complete workflow:
    1. Graph sampling using various sampling methods
    2. Instance extraction from sampled graphs (optional)
    3. Attribute extraction from instances
    4. Hypothesis testing on extracted attributes
    5. Result aggregation across multiple samples
    """

    def __init__(self, args):
        """
        Initialize the pipeline with configuration.

        Args:
            args: Arguments object containing experiment configuration
        """
        self.args = args
        self.sampler_mapping = self._build_sampler_mapping()

    def _build_sampler_mapping(self) -> Dict:
        """Build mapping from sampling method names to sampler classes."""
        return {
            "RNNS": RandomNodeNeighborSampler,
            "SRW": RandomWalkSampler,
            "FFS": ForestFireSampler,
            "ShortestPathS": ShortestPathSampler,
            "MHRWS": MetropolisHastingsRandomWalkSampler,
            "CommunitySES": CommunityStructureExpansionSampler_new,
            "NBRW": NonBackTrackingRandomWalkSampler,
            "SBS": SnowBallSampler_new,
            "RW_Starter": RandomWalkWithRestartSampler_new,
            "FrontierS": FrontierSampler_new,
            "CNARW": CommonNeighborAwareRandomWalkSampler_new,
            "RNS": RandomNodeSampler,
            "DBS": DegreeBasedSampler,
            "PRBS": PageRankBasedSampler,
            "RES": RandomEdgeSampler_new,
            "RNES": RandomNodeEdgeSampler_new,
            "RES_Induction": RandomEdgeSamplerWithInduction,
            "PHASE": PHASE,
            "Opt_PHASE": Opt_PHASE,
            "TriangleS": TriangleSubgraphSampler,
        }

    def _create_sampler(self, seed: int):
        """
        Create a sampler instance based on the sampling method.

        Args:
            seed: Random seed for the sampler

        Returns:
            Sampler instance
        """
        if self.args.sampling_method not in self.sampler_mapping:
            raise ValueError(f"Invalid sampler type: {self.args.sampling_method}")

        sampler_class = self.sampler_mapping[self.args.sampling_method]

        # Handle special cases with custom parameters
        if self.args.sampling_method == "FFS":
            return sampler_class(number_of_nodes=self.args.ratio, seed=seed, p=0.4)
        elif self.args.sampling_method == "MHRWS":
            return sampler_class(number_of_nodes=self.args.ratio, seed=seed, alpha=0.5)
        elif self.args.sampling_method == "SBS":
            return sampler_class(number_of_nodes=self.args.ratio, seed=seed, k=200)
        elif self.args.sampling_method == "RW_Starter":
            return sampler_class(number_of_nodes=self.args.ratio, seed=seed, p=0.01)
        elif self.args.sampling_method in ["RES", "RNES", "RES_Induction"]:
            return sampler_class(number_of_edges=self.args.ratio, seed=seed)
        elif self.args.sampling_method == "PHASE":
            pattern_key = str(list(self.args.hypothesis_pattern.keys())[0])
            return PHASE(
                self.args.hypothesis_pattern[pattern_key]["path"],
                number_of_nodes=self.args.ratio,
                seed=seed,
            )
        elif self.args.sampling_method == "Opt_PHASE":
            pattern_key = str(list(self.args.hypothesis_pattern.keys())[0])
            return Opt_PHASE(
                self.args.hypothesis_pattern[pattern_key]["path"],
                number_of_nodes=self.args.ratio,
                seed=seed,
            )
        elif self.args.sampling_method == "TriangleS":
            pattern_key = str(list(self.args.hypothesis_pattern.keys())[0])
            pattern = self.args.hypothesis_pattern[pattern_key]
            return TriangleSubgraphSampler(
                pattern=pattern,
                number_of_subgraphs=self.args.ratio,
                seed=seed,
            )
        else:
            return sampler_class(number_of_nodes=self.args.ratio, seed=seed)

    def _sample(self, graph: nx.Graph, sampler, seed: int) -> Tuple:
        """
        Perform graph sampling.

        Args:
            graph: The graph to sample from
            sampler: Sampler instance
            seed: Random seed

        Returns:
            Tuple of (sampled_graph or instance_dict, new_graph if applicable)
        """
        if self.args.sampling_method == "TriangleS":
            instance_dict = sampler.sample(graph)
            return instance_dict, None
        else:
            if self.args.sampling_method in ["PHASE", "Opt_PHASE"]:
                new_graph = sampler.sample(graph, self.args)
            else:
                new_graph = sampler.sample(graph)

            num_nodes = new_graph.number_of_nodes()
            num_edges = new_graph.number_of_edges()

            self.args.logger.info(
                f"The sampled graph has {num_nodes} nodes and {num_edges} edges."
            )
            print(f"The sampled graph has {num_nodes} nodes and {num_edges} edges.")

            return None, new_graph

    def _extract_instances(self, new_graph: nx.Graph) -> Dict:
        """
        Extract instances from sampled graph.

        Args:
            new_graph: The sampled graph

        Returns:
            Dictionary of extracted instances
        """
        if self.args.sampling_method == "TriangleS":
            return {}
        return extract_instances(self.args, new_graph)

    def _run_single_sample(
        self, graph: nx.Graph, num_sample: int, time_dict: Dict
    ) -> Tuple[bool, float, float, float]:
        """
        Run a single sampling and hypothesis testing iteration.

        Args:
            graph: The graph to sample from
            num_sample: Sample iteration number
            time_dict: Dictionary to store time measurements

        Returns:
            Tuple of (accept, CI_lower, CI_upper, p_value)
        """
        time_one_sample_start = time.time()
        seed = int(self.args.seed) * num_sample

        # Create and run sampler
        sampler = self._create_sampler(seed)
        instance_dict, new_graph = self._sample(graph, sampler, seed)

        # Log and record sampling time
        sampling_time = round(time.time() - time_one_sample_start, 2)
        self.args.logger.info(
            f"Time for sampling once {self.args.ratio}: {sampling_time}."
        )
        time_dict["sampling"].append(sampling_time)

        # Extract instances if needed
        time_extraction_start = time.time()
        if instance_dict is None:
            instance_dict = self._extract_instances(new_graph)
            extraction_time = round(time.time() - time_extraction_start, 2)
            time_dict["extraction"].append(extraction_time)
        else:
            time_dict["extraction"].append(0)

        # Extract attributes and perform hypothesis testing
        attribute_dict = extract_attributes(self.args, instance_dict)
        _, accept, confidence_interval, p_value = hypothesis_testing(
            self.args, attribute_dict, new_graph if new_graph else graph
        )

        return accept, confidence_interval[0], confidence_interval[1], p_value

    def run(self, graph: nx.Graph) -> Tuple[List, float, float, float, Dict]:
        """
        Run the complete hypothesis testing pipeline.

        Args:
            graph: The graph to sample from and test

        Returns:
            Tuple of:
                - HT_result_list: List of accept/reject results
                - average_CI_lower: Average lower confidence interval
                - average_CI_upper: Average upper confidence interval
                - average_p_value: Average p-value
                - time_dict: Dictionary of time measurements
        """
        time_dict = defaultdict(list)
        CI_list = {"lower": [], "upper": []}
        p_value_list = []
        HT_result_list = []

        for num_sample in range(self.args.num_samples):
            accept, CI_lower, CI_upper, p_value = self._run_single_sample(
                graph, num_sample, time_dict
            )

            CI_list["lower"].append(CI_lower)
            CI_list["upper"].append(CI_upper)
            p_value_list.append(p_value)
            HT_result_list.append(accept)

        # Calculate averages
        average_CI_lower = (
            round(sum(CI_list["lower"]) / len(CI_list["lower"]), 2)
            if len(CI_list["lower"]) > 0
            else -1
        )

        average_CI_upper = (
            round(sum(CI_list["upper"]) / len(CI_list["upper"]), 2)
            if len(CI_list["upper"]) > 0
            else -1
        )

        average_p_value = (
            round(sum(p_value_list) / len(p_value_list), 2)
            if len(p_value_list) > 0
            else -1
        )

        return (
            HT_result_list,
            average_CI_lower,
            average_CI_upper,
            average_p_value,
            time_dict,
        )
