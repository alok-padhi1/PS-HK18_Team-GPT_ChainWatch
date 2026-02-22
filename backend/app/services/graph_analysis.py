"""
Graph Analysis Service
Uses NetworkX for wallet interaction graph analysis, cycle detection,
wash-trade identification, centrality scoring, community detection (Louvain),
PageRank, and hub/authority classification.
"""

import logging
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict

import networkx as nx
from sqlalchemy.orm import Session

from app.database import Transaction

logger = logging.getLogger("chainwatch.graph")


class GraphAnalyzer:
    """Builds and analyses directed wallet interaction graphs."""

    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self._communities: Dict[str, int] = {}
        self._pagerank: Dict[str, float] = {}
        self._hub_scores: Dict[str, float] = {}
        self._authority_scores: Dict[str, float] = {}

    def build_graph(self, db: Session) -> Dict[str, Any]:
        """
        Build a directed graph from stored transactions.
        Edge weight = total ETH transferred between two wallets.
        Edge count = number of transactions.
        Also computes communities, PageRank, and HITS scores.
        """
        self.graph = nx.DiGraph()
        transactions = db.query(Transaction).all()

        edge_data: Dict[Tuple[str, str], Dict] = defaultdict(
            lambda: {"weight": 0.0, "count": 0, "blocks": set(), "gas_total": 0.0}
        )

        for tx in transactions:
            if not tx.to_address:
                continue
            key = (tx.from_address, tx.to_address)
            edge_data[key]["weight"] += tx.value_eth
            edge_data[key]["count"] += 1
            edge_data[key]["blocks"].add(tx.block_number)
            edge_data[key]["gas_total"] += tx.gas_price_gwei

        for (src, dst), data in edge_data.items():
            self.graph.add_edge(
                src, dst,
                weight=data["weight"],
                count=data["count"],
                blocks=len(data["blocks"]),
                avg_gas=data["gas_total"] / data["count"] if data["count"] > 0 else 0,
            )

        # Compute advanced metrics
        self._compute_communities()
        self._compute_pagerank()
        self._compute_hits()

        logger.info(
            f"Graph built: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges, "
            f"{len(set(self._communities.values()))} communities"
        )

        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "communities": len(set(self._communities.values())),
        }

    def _compute_communities(self):
        """Detect communities using greedy modularity on undirected projection."""
        try:
            undirected = self.graph.to_undirected()
            communities_gen = nx.community.greedy_modularity_communities(undirected)
            self._communities = {}
            for idx, community in enumerate(communities_gen):
                for node in community:
                    self._communities[node] = idx
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            self._communities = {n: 0 for n in self.graph.nodes()}

    def _compute_pagerank(self):
        """Compute PageRank for all nodes."""
        try:
            self._pagerank = nx.pagerank(self.graph, alpha=0.85, max_iter=100)
        except Exception as e:
            logger.warning(f"PageRank failed: {e}")
            self._pagerank = {n: 0.0 for n in self.graph.nodes()}

    def _compute_hits(self):
        """Compute HITS hub and authority scores."""
        try:
            hubs, authorities = nx.hits(self.graph, max_iter=100)
            self._hub_scores = hubs
            self._authority_scores = authorities
        except Exception as e:
            logger.warning(f"HITS failed: {e}")
            self._hub_scores = {n: 0.0 for n in self.graph.nodes()}
            self._authority_scores = {n: 0.0 for n in self.graph.nodes()}

    def detect_cycles(self, max_length: int = 5) -> List[List[str]]:
        """
        Detect short cycles (length <= max_length) indicating circular trading.
        Returns list of cycles (lists of wallet addresses).
        """
        cycles: List[List[str]] = []
        try:
            all_cycles = list(nx.simple_cycles(self.graph))
            for cycle in all_cycles:
                if 2 <= len(cycle) <= max_length:
                    cycles.append(cycle)
                if len(cycles) >= 500:  # increased cap for more data
                    break
        except Exception as e:
            logger.error(f"Cycle detection error: {e}")

        logger.info(f"Detected {len(cycles)} cycles (max length {max_length})")
        return cycles

    def detect_wash_trading(self, db: Session) -> List[Dict[str, Any]]:
        """
        Detect wash-trading patterns:
        - Bidirectional transfers between wallets
        - Similar values in both directions
        - Short time intervals

        Returns list of suspicious wallet pairs with details.
        """
        suspicious_pairs: List[Dict[str, Any]] = []

        for u, v, data_uv in self.graph.edges(data=True):
            if self.graph.has_edge(v, u):
                data_vu = self.graph[v][u]

                val_uv = data_uv.get("weight", 0)
                val_vu = data_vu.get("weight", 0)

                if val_uv == 0 and val_vu == 0:
                    continue

                max_val = max(val_uv, val_vu)
                min_val = min(val_uv, val_vu)
                similarity = min_val / max_val if max_val > 0 else 0

                if similarity > 0.7:  # slightly more sensitive threshold
                    # Check if they're in the same community (more suspicious)
                    same_community = self._communities.get(u, -1) == self._communities.get(v, -2)
                    community_bonus = 10 if same_community else 0

                    suspicious_pairs.append({
                        "wallet_a": u,
                        "wallet_b": v,
                        "value_a_to_b": round(val_uv, 6),
                        "value_b_to_a": round(val_vu, 6),
                        "value_similarity": round(similarity, 4),
                        "tx_count_a_to_b": data_uv.get("count", 0),
                        "tx_count_b_to_a": data_vu.get("count", 0),
                        "same_community": same_community,
                        "suspicion_score": round(
                            similarity * 50 + min(data_uv["count"] + data_vu["count"], 10) * 5 + community_bonus, 2
                        ),
                    })

        # Deduplicate (A,B) and (B,A)
        seen: Set[Tuple[str, str]] = set()
        unique_pairs = []
        for pair in suspicious_pairs:
            key = tuple(sorted([pair["wallet_a"], pair["wallet_b"]]))
            if key not in seen:
                seen.add(key)
                unique_pairs.append(pair)

        unique_pairs.sort(key=lambda x: x["suspicion_score"], reverse=True)
        logger.info(f"Detected {len(unique_pairs)} potential wash-trading pairs")
        return unique_pairs

    def compute_centrality(self, top_n: int = 50) -> List[Dict[str, Any]]:
        """
        Compute degree centrality, betweenness centrality, PageRank, and HITS.
        Returns top-N wallets by combined centrality score.
        """
        if self.graph.number_of_nodes() == 0:
            return []

        degree_cent = nx.degree_centrality(self.graph)
        try:
            between_cent = nx.betweenness_centrality(
                self.graph, k=min(100, self.graph.number_of_nodes())
            )
        except Exception:
            between_cent = {n: 0.0 for n in self.graph.nodes()}

        combined = []
        for node in self.graph.nodes():
            pr = self._pagerank.get(node, 0)
            hub = self._hub_scores.get(node, 0)
            auth = self._authority_scores.get(node, 0)

            score = (
                degree_cent.get(node, 0) * 0.3
                + between_cent.get(node, 0) * 0.25
                + pr * 0.25
                + max(hub, auth) * 0.2
            ) * 100

            combined.append({
                "address": node,
                "degree_centrality": round(degree_cent.get(node, 0), 4),
                "betweenness_centrality": round(between_cent.get(node, 0), 4),
                "pagerank": round(pr * 1000, 4),  # scaled for readability
                "hub_score": round(hub, 4),
                "authority_score": round(auth, 4),
                "community": self._communities.get(node, -1),
                "centrality_score": round(score, 2),
            })

        combined.sort(key=lambda x: x["centrality_score"], reverse=True)
        return combined[:top_n]

    def get_graph_data(self) -> Dict[str, Any]:
        """
        Return graph data formatted for frontend visualization.
        Nodes have id, centrality, community, pagerank, role.
        Edges have source, target, weight, count.
        """
        if self.graph.number_of_nodes() == 0:
            return {"nodes": [], "links": []}

        degree_cent = nx.degree_centrality(self.graph)

        # Determine max pagerank for normalization
        max_pr = max(self._pagerank.values()) if self._pagerank else 1.0
        max_pr = max_pr if max_pr > 0 else 1.0

        nodes = []
        for node in self.graph.nodes():
            pr = self._pagerank.get(node, 0)
            hub = self._hub_scores.get(node, 0)
            auth = self._authority_scores.get(node, 0)
            deg = self.graph.degree(node)
            in_deg = self.graph.in_degree(node)
            out_deg = self.graph.out_degree(node)

            # Classify node role
            if hub > auth * 1.5 and hub > 0.001:
                role = "hub"  # sends to many
            elif auth > hub * 1.5 and auth > 0.001:
                role = "authority"  # receives from many
            elif deg > 10:
                role = "connector"  # bridge between clusters
            else:
                role = "normal"

            nodes.append({
                "id": node,
                "centrality": round(degree_cent.get(node, 0) * 100, 2),
                "degree": deg,
                "in_degree": in_deg,
                "out_degree": out_deg,
                "community": self._communities.get(node, 0),
                "pagerank": round(pr / max_pr * 100, 2),
                "hub_score": round(hub * 100, 2),
                "authority_score": round(auth * 100, 2),
                "role": role,
            })

        links = []
        for u, v, data in self.graph.edges(data=True):
            links.append({
                "source": u,
                "target": v,
                "value": round(data.get("weight", 0), 6),
                "count": data.get("count", 1),
                "blocks": data.get("blocks", 1),
            })

        # Community stats
        community_counts = defaultdict(int)
        for node in nodes:
            community_counts[node["community"]] += 1

        return {
            "nodes": nodes,
            "links": links,
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(links),
                "communities": len(community_counts),
                "community_sizes": dict(community_counts),
            },
        }

    def get_wallet_graph_score(self, address: str) -> float:
        """
        Compute a graph-based suspicion score (0-100) for a single wallet.
        Based on cycle involvement + centrality + bidirectional transfer ratio + PageRank.
        """
        if address not in self.graph:
            return 0.0

        score = 0.0

        # Centrality component
        degree_cent = nx.degree_centrality(self.graph).get(address, 0)
        score += degree_cent * 25

        # PageRank component (high PageRank = important node)
        pr = self._pagerank.get(address, 0)
        max_pr = max(self._pagerank.values()) if self._pagerank else 1.0
        score += (pr / max_pr if max_pr > 0 else 0) * 15

        # Bidirectional relationships
        neighbors = set(self.graph.successors(address)) | set(self.graph.predecessors(address))
        bidirectional = 0
        for nbr in neighbors:
            if self.graph.has_edge(address, nbr) and self.graph.has_edge(nbr, address):
                bidirectional += 1
        if neighbors:
            score += (bidirectional / len(neighbors)) * 35

        # High connection count
        score += min(self.graph.degree(address), 30) * 1.0

        return min(round(score, 2), 100.0)


# Singleton
graph_analyzer = GraphAnalyzer()
