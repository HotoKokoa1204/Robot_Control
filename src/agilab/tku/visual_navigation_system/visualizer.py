from typing import Any, Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import torch
from matplotlib.patches import Patch
from torch_geometric.utils import to_networkx


class GraphVisualizer:
    """
    負責將圖結構轉換為 NetworkX 格式，計算最短路徑並將結果繪製出來。
    """

    def __init__(self, graph_dict: Dict[str, Any], config: Dict[str, Any]):
        """
        Args:
            graph_dict (Dict[str, Any]): 包含建立好的圖結構、節點映射、邊緣分類的字典。
            config (Dict[str, Any]): 設定檔，用於載入繪圖相關參數。
        """
        self.graph_data = graph_dict["graph_data"]
        self.node_to_video_map = graph_dict["node_to_video_map"]
        self.temporal_edges = graph_dict["temporal_edges"]
        self.cross_video_edges = graph_dict["cross_video_edges"]
        self.cross_video_nodes = graph_dict["cross_video_nodes"]

        self.video_names = config["graph"]["video_names"]
        self.layout_prog = config["graph"].get("layout_prog", "sfdp")

        self.my_colors = [
            "#e6194B",
            "#3cb44b",
            "#ffe119",
            "#4363d8",
            "#f58231",
            "#911eb4",
            "#46f0f0",
            "#f032e6",
            "#bcf60c",
            "#fabebe",
            "#008080",
            "#e6beff",
            "#9a6324",
            "#fffac8",
            "#800000",
            "#aaffc3",
            "#808000",
            "#ffd8b1",
            "#000075",
            "#808080",
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
        ]

        if len(self.my_colors) < len(self.video_names):
            print("顏色數量少於影片數量，將自動循環使用顏色。")

        self.G = to_networkx(self.graph_data, to_undirected=True)
        self._calculate_layout()

    def _calculate_layout(self):
        """計算並儲存節點的視覺化座標。"""
        try:
            from networkx.drawing.nx_agraph import graphviz_layout

            self.pos = graphviz_layout(self.G, prog=self.layout_prog)
        except ImportError:
            try:
                from networkx.drawing.nx_pydot import graphviz_layout

                self.pos = graphviz_layout(self.G, prog=self.layout_prog)
            except ImportError:
                print(
                    "找不到 pygraphviz 套件或 pydot 套件，\
                        改為使用 spring_layout，可能導致繪圖較擁擠。"
                )
                self.pos = nx.spring_layout(self.G)

    def find_shortest_path(
        self, input_feat1: torch.Tensor, input_feat2: torch.Tensor
    ) -> List[int]:
        """
        找出兩個特徵向量在圖上的最短路徑。
        """

        def find_nearest_node(x: torch.Tensor, target_feat: torch.Tensor):
            distances = torch.norm(x - target_feat.unsqueeze(0), dim=1)
            return torch.argmin(distances).item()

        start_node = find_nearest_node(self.graph_data.x, input_feat1)
        end_node = find_nearest_node(self.graph_data.x, input_feat2)

        print(f"起點節點 ID: {start_node}, 終點節點 ID: {end_node}")
        try:
            path_nodes = nx.shortest_path(self.G, source=start_node, target=end_node)
            print(f"找到最短路徑長度: {len(path_nodes)}")
            return path_nodes
        except nx.NetworkXNoPath:
            print("起點與終點之間不存在連通路徑！")
            return []

    def draw_graph(self, path_nodes: List[int], output_path: str = None):
        """
        將背景節點、跨影片交接點、最短路徑分層繪製出來。
        """
        plt.figure(figsize=(14, 10))

        # 第一層：繪製弱化的背景
        background_nodes = [
            n
            for n in self.G.nodes()
            if n not in path_nodes and n not in self.cross_video_nodes
        ]
        background_colors = [
            self.my_colors[self.node_to_video_map[n] % len(self.my_colors)]
            for n in background_nodes
        ]

        nx.draw_networkx_nodes(
            self.G,
            self.pos,
            nodelist=background_nodes,
            node_size=15,
            node_color=background_colors,
            alpha=0.4,
            edgecolors="none",
        )
        nx.draw_networkx_edges(
            self.G, self.pos, edgelist=self.temporal_edges, edge_color="gray", width=0.5
        )

        # 第二層：高亮顯示最短路徑
        if path_nodes:
            nx.draw_networkx_nodes(
                self.G,
                self.pos,
                nodelist=path_nodes,
                node_size=75,
                node_color="orange",
                edgecolors="none",
            )
            path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
            nx.draw_networkx_edges(
                self.G,
                self.pos,
                edgelist=path_edges,
                edge_color="black",
                style="dashed",
                width=1.5,
            )

        # 第三層：疊加跨影片特殊交接節點
        nx.draw_networkx_nodes(
            self.G,
            self.pos,
            nodelist=list(self.cross_video_nodes),
            node_size=75,
            alpha=0.7,
            node_color="blueviolet",
            edgecolors="black",
            linewidths=1.5,
        )
        nx.draw_networkx_edges(
            self.G,
            self.pos,
            edgelist=self.cross_video_edges,
            edge_color="red",
            width=1.5,
        )

        # 第四層：畫起點與終點 (確保在最上層)
        if path_nodes:
            nx.draw_networkx_nodes(
                self.G,
                self.pos,
                nodelist=[path_nodes[0]],
                node_size=250,
                node_color="red",
                edgecolors="black",
                linewidths=2,
            )
            nx.draw_networkx_nodes(
                self.G,
                self.pos,
                nodelist=[path_nodes[-1]],
                node_size=250,
                node_color="green",
                edgecolors="black",
                linewidths=2,
            )

        # 設定圖例
        legend_elements = [
            Patch(
                facecolor=self.my_colors[i % len(self.my_colors)],
                label=self.video_names[i],
            )
            for i in range(len(self.video_names))
        ]
        legend_elements += [
            Patch(facecolor="blueviolet", edgecolor="black", label="Cross-Video Nodes"),
            Patch(facecolor="orange", label="Path Nodes"),
            Patch(facecolor="red", edgecolor="black", label="Start"),
            Patch(facecolor="green", edgecolor="black", label="End"),
        ]

        plt.legend(handles=legend_elements, loc="upper right", ncol=2, fontsize="small")
        plt.title("Indoor Positioning Graph with Shortest Path Prediction")
        plt.axis("off")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300)
            print(f"圖表已儲存至: {output_path}")
        else:
            plt.show()
