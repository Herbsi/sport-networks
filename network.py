import networkx as nx
from typing import Any
from collections.abc import Callable


def connectivity(G: nx.Graph | nx.DiGraph) -> float:
    return nx.average_node_connectivity(G)


def assortativity(G: nx.Graph | nx.DiGraph) -> float:
    return nx.degree_assortativity_coefficient(G, weight="weight")


def number_connected_components(G: nx.Graph | nx.DiGraph) -> int:
    if isinstance(G, nx.DiGraph):
        return nx.number_strongly_connected_components(G)
    elif isinstance(G, nx.Graph):
        return nx.number_connected_components(G)
    else:
        raise TypeError(f"Unsupported argument type {type(G)}.")


def clustering(G: nx.Graph | nx.DiGraph) -> float:
    return nx.average_clustering(G, weight="weight")


def centrality(G: nx.Graph | nx.DiGraph, u=None) -> float:
    return nx.closeness_centrality(G, u, distance="weight")


# Disruption Features
def _disruption_feature(G: nx.Graph | nx.DiGraph, nodes: list[Any] | None, feature: Callable[[nx.Graph | nx.DiGraph], float | int]) -> float:
    # TODO: All nodes are weighted equally for the disruption feature, in contrast to the Haka-Network paper’s use of the disruption frequency d_{u, t_2, g}.
    mean = 0.0

    if nodes is None:
        nodes = G.nodes

    for node in nodes:
        # TODO: Dies if feature(G_local) = nan, which can happen if, e.g., G_local is no longer connected and feature is assortativity.
        G_local = G.copy()
        G_local.remove_node(node)
        mean += (feature(G_local) - feature(G))
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
