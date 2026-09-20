import pytest
import networkx as nx
from src.graph_store import GraphStore

def test_traverse_subgraph_path_ranking():
    # Instantiate GraphStore in memory without persistent file
    store = GraphStore()
    store.graph = nx.DiGraph()

    # Add edges:
    # Alpha -> Beta (1-hop from Alpha, 2 sources)
    # Beta -> Gamma (2-hop from Alpha, 1 source)
    # Alpha -> Delta (1-hop from Alpha, 1 source)
    store.graph.add_edge("Alpha", "Beta", relation="connects_to", sources=["doc1.pdf", "doc2.pdf"], pages=["doc1.pdf:p1", "doc2.pdf:p1"])
    store.graph.add_edge("Beta", "Gamma", relation="influences", sources=["doc1.pdf"], pages=["doc1.pdf:p2"])
    store.graph.add_edge("Alpha", "Delta", relation="supports", sources=["doc3.pdf"], pages=["doc3.pdf:p1"])

    # Traverse starting from Alpha with max_depth=2
    relations = store.traverse_subgraph(["Alpha"], max_depth=2)

    assert len(relations) == 3
    # First relation should be Alpha -> Beta because support=2 at depth 0: score = (2*2)/(0+1) = 4.0
    assert relations[0]["subject"] == "Alpha" and relations[0]["object"] == "Beta"
    assert relations[0]["score"] == 4.0

    # Alpha -> Delta has support=1 at depth 0: score = (1*2)/(0+1) = 2.0
    # Beta -> Gamma has support=1 at depth 1: score = (1*2)/(1+1) = 1.0
    assert relations[1]["object"] == "Delta"
    assert relations[2]["object"] == "Gamma"
    assert relations[1]["score"] > relations[2]["score"]

def test_traverse_subgraph_source_filtering():
    store = GraphStore()
    store.graph = nx.DiGraph()

    store.graph.add_edge("Entity1", "Entity2", relation="partners_with", sources=["public.pdf"], pages=["public.pdf:p1"])
    store.graph.add_edge("Entity1", "Entity3", relation="secret_meeting", sources=["classified.pdf"], pages=["classified.pdf:p1"])

    results = store.traverse_subgraph(["Entity1"], source_filter=["public.pdf"])

    assert len(results) == 1
    assert results[0]["object"] == "Entity2"
    assert "public.pdf" in results[0]["sources"]
