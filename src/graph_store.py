import os
import json
import threading
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
from pyvis.network import Network
from src import config
from src import llm

class GraphStore:
    def __init__(self):
        self.graph_path = os.path.join(config.GRAPH_DIR, "knowledge_graph.json")
        self.graph = nx.DiGraph()
        self._write_lock = threading.Lock()  # Serialises graph mutations during parallel ingestion
        self.load()

    def load(self):
        """Loads the graph from a JSON file if it exists."""
        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                self.graph = nx.DiGraph()
                for node in data.get("nodes", []):
                    self.graph.add_node(node["id"], **node.get("data", {}))
                for edge in data.get("edges", []):
                    self.graph.add_edge(edge["source"], edge["target"], **edge.get("data", {}))
                print(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")
            except Exception as e:
                print(f"Error loading graph, initializing empty graph: {e}")
                self.graph = nx.DiGraph()
        else:
            self.graph = nx.DiGraph()

    def save(self):
        """Saves the graph to a JSON file."""
        data = {
            "nodes": [{"id": node, "data": self.graph.nodes[node]} for node in self.graph.nodes],
            "edges": [{"source": u, "target": v, "data": self.graph.edges[u, v]} for u, v in self.graph.edges]
        }
        try:
            with open(self.graph_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving graph: {e}")

    def add_relations_from_chunk(self, chunk: dict):
        """
        Uses LLM to extract entities and relations from a text chunk,
        and adds them to the NetworkX graph.
        The LLM inference runs without any lock (concurrent-safe via Ollama's own queuing).
        Graph mutations are serialised with _write_lock to prevent race conditions.
        """
        text = chunk["text"]
        source = chunk["source"]
        page = chunk["page"]
        
        # Prepare extraction prompt
        prompt = config.ENTITY_EXTRACTION_PROMPT.format(text=text)
        
        try:
            # LLM call: runs concurrently across threads (no lock needed)
            triples = llm.generate_json(prompt, task="fast")
            if not isinstance(triples, list):
                return

            # Build the local mutations list without holding the lock
            mutations = []
            for triple in triples:
                if not isinstance(triple, dict):
                    continue
                subj = triple.get("subject")
                rel = triple.get("relation")
                obj = triple.get("object")
                if not subj or not rel or not obj:
                    continue
                subj = str(subj).strip().title()
                obj = str(obj).strip().title()
                rel = str(rel).strip().lower()
                mutations.append((subj, rel, obj))

            # Acquire lock only for the actual graph write
            with self._write_lock:
                for subj, rel, obj in mutations:
                    # Add nodes if they don't exist
                    if not self.graph.has_node(subj):
                        self.graph.add_node(subj, type="Entity", degree=0)
                    if not self.graph.has_node(obj):
                        self.graph.add_node(obj, type="Entity", degree=0)
                    
                    # Add or update edge
                    if self.graph.has_edge(subj, obj):
                        edge_data = self.graph.edges[subj, obj]
                        if source not in edge_data.get("sources", []):
                            edge_data["sources"].append(source)
                        edge_data["count"] = edge_data.get("count", 1) + 1
                        page_str = f"{source}:p{page}"
                        if page_str not in edge_data.get("pages", []):
                            edge_data["pages"].append(page_str)
                    else:
                        self.graph.add_edge(
                            subj,
                            obj,
                            relation=rel,
                            sources=[source],
                            pages=[f"{source}:p{page}"],
                            count=1
                        )
                
                # Update degree attribute for visualization sizing
                for node in self.graph.nodes:
                    self.graph.nodes[node]["degree"] = self.graph.degree(node)
                
        except Exception as e:
            print(f"Error extracting relations from chunk: {e}")

    def add_relations_from_chunks_parallel(self, chunks: list, max_workers: int = 2):
        """
        Processes a list of chunks in parallel using ThreadPoolExecutor.
        max_workers=2 is optimal for an RTX 3050 4GB:
          - Two gemma3:1b (0.8 GB each) fit in VRAM simultaneously.
          - LLM inference runs concurrently; graph writes are serialised via _write_lock.
        """
        if not chunks:
            return
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.add_relations_from_chunk, chunk): i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Parallel extraction error on chunk {futures[future]}: {e}")

    def traverse_subgraph(self, seed_entities: list[str], max_depth: int = 1, source_filter: list[str] = None) -> list[dict]:
        """
        Traverses the graph starting from seed entities up to max_depth,
        returning a list of relations (edges) found, optionally filtered by sources.
        """
        visited_nodes = set()
        retrieved_relations = []
        
        # Normalize seed entities
        normalized_seeds = [seed.strip().title() for seed in seed_entities]
        
        # Queue format: (node, depth)
        queue = [(node, 0) for node in normalized_seeds if self.graph.has_node(node)]
        
        for node, _ in queue:
            visited_nodes.add(node)
            
        # Perform BFS up to max_depth
        current_queue = list(queue)
        next_queue = []
        
        for depth in range(max_depth):
            for node, d in current_queue:
                # Find all neighbors (incoming and outgoing)
                # Outgoing edges
                for neighbor in self.graph.successors(node):
                    edge_data = self.graph.edges[node, neighbor]
                    sources = edge_data.get("sources", [])
                    pages = edge_data.get("pages", [])
                    
                    if source_filter is not None:
                        # Filter sources and pages based on source_filter
                        filtered_sources = [s for s in sources if s in source_filter]
                        if not filtered_sources:
                            continue
                        filtered_pages = [p for p in pages if any(p.startswith(s + ":p") for s in source_filter)]
                        sources = filtered_sources
                        pages = filtered_pages
                        
                    rel_dict = {
                        "subject": node,
                        "relation": edge_data.get("relation", "connected_to"),
                        "object": neighbor,
                        "sources": sources,
                        "pages": pages,
                        "count": len(sources)
                    }
                    if rel_dict not in retrieved_relations:
                        retrieved_relations.append(rel_dict)
                    if neighbor not in visited_nodes:
                        visited_nodes.add(neighbor)
                        next_queue.append((neighbor, d + 1))
                # Incoming edges
                for predecessor in self.graph.predecessors(node):
                    edge_data = self.graph.edges[predecessor, node]
                    sources = edge_data.get("sources", [])
                    pages = edge_data.get("pages", [])
                    
                    if source_filter is not None:
                        # Filter sources and pages based on source_filter
                        filtered_sources = [s for s in sources if s in source_filter]
                        if not filtered_sources:
                            continue
                        filtered_pages = [p for p in pages if any(p.startswith(s + ":p") for s in source_filter)]
                        sources = filtered_sources
                        pages = filtered_pages
                        
                    rel_dict = {
                        "subject": predecessor,
                        "relation": edge_data.get("relation", "connected_to"),
                        "object": node,
                        "sources": sources,
                        "pages": pages,
                        "count": len(sources)
                    }
                    if rel_dict not in retrieved_relations:
                        retrieved_relations.append(rel_dict)
                    if predecessor not in visited_nodes:
                        visited_nodes.add(predecessor)
                        next_queue.append((predecessor, d + 1))
            current_queue = next_queue
            next_queue = []
            
        return retrieved_relations

    def find_seeds_in_query(self, query: str) -> list[str]:
        """Matches nodes in the graph against words/phrases in the user's query."""
        seeds = []
        query_lower = query.lower()
        
        # Simple string-matching: see if any node name is contained in the query
        for node in self.graph.nodes:
            if node.lower() in query_lower:
                seeds.append(node)
        return seeds

    def generate_visualization_html(self, output_filename: str = "graph.html") -> str:
        """
        Generates an interactive Pyvis HTML visualization of the knowledge graph.
        Returns the absolute path to the HTML file.
        """
        net = Network(
            height="500px", 
            width="100%", 
            bgcolor="#09090b",  # Dark mode canvas background
            font_color="#a1a1aa",  # Muted gray label color
            directed=True,
            notebook=False
        )
        
        # Apply physics options for premium layout feel
        net.set_options("""
        var options = {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -80,
              "centralGravity": 0.01,
              "springLength": 200,
              "springConstant": 0.06
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": { "iterations": 150 }
          },
          "edges": {
            "smooth": {
              "type": "continuous",
              "forceDirection": "none"
            },
            "hoverWidth": 2
          },
          "interaction": {
            "hover": true
          }
        }
        """)
        
        # Get degrees of all nodes in full graph
        degrees = dict(self.graph.degree())
        sorted_nodes = sorted(self.graph.nodes, key=lambda x: degrees.get(x, 0), reverse=True)
        viz_nodes = set(sorted_nodes[:80])
        
        # Filter edges for visualization
        edges_to_add = []
        nodes_with_edges = set()
        for u, v in self.graph.edges:
            if u not in viz_nodes or v not in viz_nodes:
                continue
            edge_data = self.graph.edges[u, v]
            count = edge_data.get("count", 1)
            
            # Noise pruning: skip if count is 1 AND both endpoints have degree <= 1 in full graph
            if count < 2 and degrees.get(u, 0) <= 1 and degrees.get(v, 0) <= 1:
                continue
                
            edges_to_add.append((u, v, edge_data))
            nodes_with_edges.add(u)
            nodes_with_edges.add(v)
            
        # Nodes to add: if we have edges, show nodes that have edges. Otherwise just show all viz_nodes
        nodes_to_add = nodes_with_edges if nodes_with_edges else viz_nodes

        # Add nodes with deg-based sizing and custom colors
        for node in nodes_to_add:
            deg = degrees.get(node, 1)
            # Uniform node sizing constrained between 6 and 24 based on connection density
            size = min(max(6, deg * 2.5), 24)
            
            # Three-tier node coloring & font settings
            if deg >= 5:
                color = "#4f46e5"  # Hub: Indigo
                font_size = 13
            elif deg >= 2:
                color = "#06b6d4"  # Mid-tier: Cyan
                font_size = 0  # Hidden by default
            else:
                color = "#52525b"  # Leaf: Zinc
                font_size = 0  # Hidden by default
            
            net.add_node(
                node, 
                label=node, 
                title=f"Entity: {node}\nDegree: {deg}", 
                size=size,
                color=color,
                font={"size": font_size, "color": "#f4f4f5"}
            )
            
        # Add edges
        for u, v, edge_data in edges_to_add:
            relation = edge_data.get("relation", "")
            short_rel = relation[:20] + "..." if len(relation) > 20 else relation
            sources = ", ".join(edge_data.get("sources", []))
            
            # Faint gray edge highlighting to indigo on hover, with relation label displayed on edge
            net.add_edge(
                u, 
                v, 
                label=short_rel,
                title=f"Relation: {relation}\nSources: {sources}",
                color={
                    "color": "rgba(161, 161, 170, 0.25)",
                    "highlight": "#4f46e5",
                    "hover": "#4f46e5"
                },
                width=1.5,
                font={"size": 9, "color": "#71717a", "strokeWidth": 2, "strokeColor": "#09090b"}
            )
            
        # Save HTML file
        output_path = os.path.join(config.GRAPH_DIR, output_filename)
        net.save_graph(output_path)
        
        # Inject custom hover event listeners for node labels in Vis.js
        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                
                # Check for standard pyvis initialization patterns
                old_init = "network = new vis.Network(container, data, options);"
                new_init = (
                    "network = new vis.Network(container, data, options);\n"
                    "    network.on(\"hoverNode\", function(e) {\n"
                    "        network.body.data.nodes.update({id: e.node, font: {size: 12, color: '#f4f4f5'}});\n"
                    "    });\n"
                    "    network.on(\"blurNode\", function(e) {\n"
                    "        var nodeData = network.body.data.nodes.get(e.node);\n"
                    "        var originalSize = 0;\n"
                    "        if (nodeData && nodeData.title && nodeData.title.indexOf(\"Degree: \") !== -1) {\n"
                    "            var parts = nodeData.title.split(\"Degree: \");\n"
                    "            var deg = parseInt(parts[1]);\n"
                    "            if (deg >= 5) {\n"
                    "                originalSize = 13;\n"
                    "            }\n"
                    "        }\n"
                    "        network.body.data.nodes.update({id: e.node, font: {size: originalSize, color: '#f4f4f5'}});\n"
                    "    });"
                )
                if old_init in html_content:
                    html_content = html_content.replace(old_init, new_init)
                else:
                    old_init_var = "var network = new vis.Network(container, data, options);"
                    new_init_var = (
                        "var network = new vis.Network(container, data, options);\n"
                        "    network.on(\"hoverNode\", function(e) {\n"
                        "        network.body.data.nodes.update({id: e.node, font: {size: 12, color: '#f4f4f5'}});\n"
                        "    });\n"
                        "    network.on(\"blurNode\", function(e) {\n"
                        "        var nodeData = network.body.data.nodes.get(e.node);\n"
                        "        var originalSize = 0;\n"
                        "        if (nodeData && nodeData.title && nodeData.title.indexOf(\"Degree: \") !== -1) {\n"
                        "            var parts = nodeData.title.split(\"Degree: \");\n"
                        "            var deg = parseInt(parts[1]);\n"
                        "            if (deg >= 5) {\n"
                        "                originalSize = 13;\n"
                        "            }\n"
                        "        }\n"
                        "        network.body.data.nodes.update({id: e.node, font: {size: originalSize, color: '#f4f4f5'}});\n"
                        "    });"
                    )
                    html_content = html_content.replace(old_init_var, new_init_var)
                
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
            except Exception as e:
                print(f"Error injecting hover event listeners: {e}")
                
        return output_path

    def delete_by_source(self, source_name: str):
        """Removes all edges and orphan nodes associated with the deleted source."""
        edges_to_remove = []
        edges_to_modify = []
        
        for u, v in self.graph.edges:
            edge_data = self.graph.edges[u, v]
            sources = edge_data.get("sources", [])
            pages = edge_data.get("pages", [])
            
            if source_name in sources:
                new_sources = [s for s in sources if s != source_name]
                if not new_sources:
                    edges_to_remove.append((u, v))
                else:
                    new_pages = [p for p in pages if not p.startswith(source_name + ":p")]
                    edges_to_modify.append((u, v, new_sources, new_pages))
                    
        # Remove edges
        for u, v in edges_to_remove:
            self.graph.remove_edge(u, v)
            
        # Modify remaining edges
        for u, v, new_sources, new_pages in edges_to_modify:
            self.graph.edges[u, v]["sources"] = new_sources
            self.graph.edges[u, v]["pages"] = new_pages
            self.graph.edges[u, v]["count"] = len(new_sources)
            
        # Remove orphan nodes (nodes with 0 degree)
        orphans = list(nx.isolates(self.graph))
        self.graph.remove_nodes_from(orphans)
        
        # Recompute degrees for visualization
        for node in self.graph.nodes:
            self.graph.nodes[node]["degree"] = self.graph.degree(node)
            
        self.save()

    def reset(self):
        """Clears the graph."""
        self.graph = nx.DiGraph()
        if os.path.exists(self.graph_path):
            try:
                os.remove(self.graph_path)
            except Exception:
                pass
        self.save()
