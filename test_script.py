import networkx as nx
import pandas as pd
from copy import deepcopy
from itertools import product
from collections import defaultdict



class PatternMatcher:
    def __init__(self, graph):
        self.graph = graph

    def match_node(self, graph_node_data, pattern_node_data):
        if graph_node_data.get("label") != pattern_node_data.get("label"):
            return False
        for attr_key, attr_val in pattern_node_data.items():
            if attr_key == "label":
                continue
            graph_val = graph_node_data.get(attr_key)
            if pd.isna(graph_val) or (graph_val != attr_val and (attr_val not in graph_val if isinstance(graph_val, str) else True)):
                return False
        return True

    def build_pattern_dag(self, pattern):
        dag = nx.DiGraph()
        for node_id, node_info in pattern["nodes"].items():
            dag.add_node(node_id, **node_info)
        for src, dst in pattern["edges"]:
            dag.add_edge(src, dst)
        return dag

    from itertools import product

    # def match_pattern_recursive(self, pattern_dag, current_pattern_node, current_graph_node, mapping, visited, rootFlag=True):
    #     mapping[current_pattern_node] = current_graph_node
    #     visited.add(current_graph_node)
    #
    #     successors = list(pattern_dag.successors(current_pattern_node))
    #     if not successors:
    #         # no more to match
    #         if rootFlag:
    #             yield mapping
    #         else:
    #             return [mapping]  # 返回一条路径，用于组合
    #
    #     all_branch_results = {}
    #
    #     for neighbor_pattern_node in successors:
    #         pattern_node_data = pattern_dag.nodes[neighbor_pattern_node]
    #         matches_for_this_branch = []
    #
    #         for neighbor_graph_node in self.graph.successors(current_graph_node):
    #             if neighbor_graph_node in visited:
    #                 continue
    #             if self.match_node(self.graph.nodes[neighbor_graph_node], pattern_node_data):
    #                 new_mapping = deepcopy(mapping)
    #                 new_visited = visited.copy()
    #                 sub_results = self.match_pattern_recursive(
    #                     pattern_dag, neighbor_pattern_node, neighbor_graph_node, new_mapping, new_visited, False
    #                 )
    #                 matches_for_this_branch.extend(sub_results)
    #
    #         all_branch_results[neighbor_pattern_node] = matches_for_this_branch
    #
    #     # 笛卡尔积合并所有分支
    #     all_combinations = product(*all_branch_results.values())
    #     for combo in all_combinations:
    #         final_mapping = deepcopy(mapping)
    #         for branch_mapping in combo:
    #             final_mapping.update(branch_mapping)
    #         if rootFlag:
    #             yield final_mapping
    #         else:
    #             yield final_mapping  # 被上层组合


    def match_pattern_recursive(self, pattern_dag, current_pattern_node, current_graph_node, mapping, visited, rootFlag = True):
        print(f">>> start pattern node {current_pattern_node} and graph node {current_graph_node}")
        mapping[current_pattern_node].append(current_graph_node)
        visited.add(current_graph_node)

        for neighbor_pattern_node in pattern_dag.successors(current_pattern_node):
            print(f"look for pattern {neighbor_pattern_node}")
            pattern_node_data = pattern_dag.nodes[neighbor_pattern_node]

            for neighbor_graph_node in self.graph.successors(current_graph_node):
                if neighbor_graph_node in visited:
                    continue
                if self.match_node(self.graph.nodes[neighbor_graph_node], pattern_node_data):
                    # print(mapping)
                    print(f"graph node {neighbor_graph_node} is matched, now mapping is {mapping}")
                    print("*** going to a recursive")
                    new_mapping = deepcopy(mapping)
                    new_visited = visited.copy()
                    new_mapping =  self.match_pattern_recursive(
                        pattern_dag, neighbor_pattern_node, neighbor_graph_node, new_mapping, new_visited, False
                    )
                    print("finished recursive!!! ***")
                    print(f"before update mapping {mapping}, newmapping: {new_mapping}")
                    mapping = new_mapping
                    print(f"resulting mapping {mapping}, newmapping: {new_mapping}")
        # ✅ yield 只在路径完整时发生
        if len(mapping) == len(pattern_dag.nodes) and rootFlag:
            print("完成啦！")
            yield mapping

    def find_meta_structures(self, pattern):
        pattern_dag = self.build_pattern_dag(pattern)
        pattern_root_id = pattern["meta_root"]
        root_node_info = pattern["nodes"][pattern_root_id]

        results = []

        for graph_node in self.graph.nodes:
            if self.match_node(self.graph.nodes[graph_node], root_node_info):
                print("✅ 匹配到 root:", graph_node)
                results.extend(self.match_pattern_recursive(
                    pattern_dag, pattern_root_id, graph_node, defaultdict(list), set()
                ))
                print(f"extended result: {results}")
            else:
                print("❌ 未匹配 root:", graph_node)

        return results

def test_pattern_matcher():
    G = nx.DiGraph()

    # 添加节点
    G.add_node(1, label="author", author_org="Microsoft Research")
    G.add_node(2, label="paper", n_citation=123)
    G.add_node(3, label="venue", venue_type="C")
    G.add_node(4, label="venue", venue_type="A")
    G.add_node(5, label="author", author_org="HKU")
    G.add_node(6, label="author", author_org="CityU")


    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(3, 6)
    G.add_edge(2, 6)

    # pattern
    pattern = {
        "nodes": {
            "n0": {"label": "author", "author_org": "Microsoft Research"},
            "n1": {"label": "paper"},
            "n2": {"label": "venue", "venue_type": "C"},
            "n3": {"label": "author"},
        },
        "edges": [["n0", "n2"], ["n0", "n1"], ["n1", "n3"], ["n2", "n3"]],
        "target": {"node": "n1", "attribute": "n_citation"},
        "meta_root": "n0"
    }

    matcher = PatternMatcher(G)
    matches = matcher.find_meta_structures(pattern)

    print("\n🔍 匹配结果:")
    for m in matches:
        print("Mapping:", m)
        combinations = [
            dict(zip(m.keys(), values))
            for values in product(*m.values())
        ]
        print(combinations)


if __name__ == "__main__":
    test_pattern_matcher()
