import networkx as nx
import numpy as np
from enum import Enum
from typing import Any
from collections.abc import Callable
from preprocess_data import directed_to_undirected


class Weight(Enum):
    N_PASSES = "n_passes"
    REL_PASSES = "rel_passes"
    REC_DISTANCE = "rec_distance"
    MAX_DISTANCE = "max_distance"  # NOTE: Currently unused


def process_graph_for_analysis(G: nx.Graph | nx.DiGraph, make_undirected: bool = False):
    # Do not make undirected by default
    # TODO: I am not sure what makes more sense for our analysis; Haka generally considered the graph as directed; so, put it behind a flag for now
    if make_undirected:
        G = directed_to_undirected(G)

    # Remove the Goalie as he is rarely involved in passes
    try:
        G.remove_node("Goalie")
    except nx.NetworkXError:
        pass

    # Remove loops because they conceptually do not make sense for our analyses
    for u in G.nodes:
        try:
            G.remove_edge(u, u)
        except nx.NetworkXError:
            pass

    # Now, after removing other stuff, normalise edge weights.
    total_passes = sum(w for (_, _, w) in G.edges.data("n_passes"))

    for u, v, n_passes in G.edges.data("n_passes"):
        # For convenience, add multiple "distance" measures to edge.
        G.edges[u, v][Weight.N_PASSES.value] = n_passes
        G.edges[u, v][Weight.REL_PASSES.value] = n_passes / total_passes
        # NOTE: Distance measures are arguably not very meaningful.
        G.edges[u, v][Weight.REC_DISTANCE.value] = total_passes / n_passes

    return G


# Topological Features
def connectivity(G: nx.Graph | nx.DiGraph) -> float:
    # NOTE: Probably uninteresting because constant over regular networks.
    return nx.average_node_connectivity(G)


def assortativity(G: nx.Graph | nx.DiGraph, weight: Weight = Weight.N_PASSES) -> float:
    # TODO: not sure if weight="distance" would make more sense here too
    return nx.degree_assortativity_coefficient(G, weight=weight.value)


def number_connected_components(G: nx.Graph | nx.DiGraph) -> int:
    # NOTE: Probably uninteresting because constant 1 over regular networks.
    if isinstance(G, nx.DiGraph):
        return nx.number_strongly_connected_components(G)
    elif isinstance(G, nx.Graph):
        return nx.number_connected_components(G)
    else:
        raise TypeError(f"Unsupported argument type {type(G)}.")


def clustering(G: nx.Graph | nx.DiGraph, weight: Weight = Weight.N_PASSES) -> float:
    return nx.average_clustering(G, weight=weight.value)


def centrality(G: nx.Graph | nx.DiGraph, u=None, weight: Weight = Weight.N_PASSES) -> float:
    return nx.closeness_centrality(G, u, distance=weight.value, wf_improved=False)


def betweenness_centrality(G: nx.Graph | nx.DiGraph, weight: Weight = Weight.REC_DISTANCE) -> dict:
    # TODO: Julian hat das zwar erwähnt, aber die Größe verwendet auch weight als distance.
    return nx.betweenness_centrality(G, weight=weight.value)


def eigenvector_centrality(G: nx.Graph | nx.DiGraph, weight: Weight = Weight.N_PASSES) -> dict:
    return nx.eigenvector_centrality(G, weight=weight.value)


def degree_mean(G: nx.Graph | nx.DiGraph, weight: Weight = Weight.N_PASSES):
    return np.mean(list(d for (_, d) in G.degree(weight=weight.value)))


def degree_std(G: nx.Graph | nx.DiGraph, weight: Weight = Weight.N_PASSES):
    return np.std(list(d for (_, d) in G.degree(weight=weight.value)))


# Disruption Features
def _disruption_feature(
    G: nx.Graph | nx.DiGraph,
    nodes: list[Any] | None,
    feature: Callable[[nx.Graph | nx.DiGraph], float | int],
) -> float:
    # TODO: All nodes are weighted equally for the disruption feature, in contrast to the Haka-Network paper’s use of the disruption frequency d_{u, t_2, g}.
    mean = 0.0

    if nodes is None:
        nodes = G.nodes

    for node in nodes:
        # TODO: Dies if feature(G_local) = nan, which can happen if, e.g., G_local is no longer connected and feature is assortativity.
        G_local = G.copy()
        G_local.remove_node(node)
        mean += feature(G_local) - feature(G)
        print(mean)

    return mean / len(nodes)


def disruption_connectivity(G: nx.Graph | nx.DiGraph, nodes: list[Any] | None = None) -> float:
    """Return average connectivity of `G' when `nodes' are removed.  Defaults to averaging over removing every node."""
    return _disruption_feature(G, nodes, connectivity)


def disruption_assortativity(G: nx.Graph | nx.DiGraph, nodes: list[Any] | None = None) -> float:
    return _disruption_feature(G, nodes, assortativity)


def disruption_number_connected_components(G: nx.Graph | nx.DiGraph, nodes: list[Any] | None = None) -> float:
    return _disruption_feature(G, nodes, number_connected_components)


def disruption_clustering(G: nx.Graph | nx.DiGraph, nodes: list[Any] | None = None) -> float:
    return _disruption_feature(G, nodes, number_connected_components)


def disruption_centrality(G: nx.Graph | nx.DiGraph):
    # TODO: Haka-Network Paper weighs centrality(G, u) by d_{u, t2, g}, the disruption frequency of node u
    # So right now, a more apt name would be "average_centrality".
    mean = 0.0
    for node in G.nodes:
        mean += centrality(G, node)

    return mean / G.number_of_nodes()
