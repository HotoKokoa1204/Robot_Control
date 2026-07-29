import os
from typing import Any, Dict, List

import numpy as np
import torch
from torch_geometric.data import Data


class GraphBuilder:
    """
    負責載入特徵向量檔案並建立 PyTorch Geometric 的圖資料結構 (節點與邊緣)。
    """

    def __init__(self, config: Dict[str, Any], feature_paths: List[str] = None):
        """
        Args:
            config (Dict[str, Any]): 設定檔字典。
            feature_paths (List[str], optional): 若不提供，預設從 config 讀取。
        """
        self.config = config

        if feature_paths is None:
            base_dir = config["graph"]["features_base_dir"]
            self.feature_paths = [
                os.path.join(base_dir, p) for p in config["graph"]["feature_subpaths"]
            ]
        else:
            self.feature_paths = feature_paths

        self.features_all = []
        self.start_indices = []
        self.video_ranges = []
        self.node_to_video_map = {}

        self.edge_list = []
        self.temporal_edges = []
        self.cross_video_edges = []
        self.cross_video_nodes = set()

        self.x = None
        self.edge_index = None
        self.graph_data = None

    def _load_features_and_nodes(self):
        """讀取所有特徵向量並記錄起始索引。"""
        curr_index = 0
        for i, path in enumerate(self.feature_paths):
            if not os.path.exists(path):
                print(f"找不到特徵檔 {path}，將使用隨機矩陣替代")
                # Fallback for testing if paths are missing
                feat = np.random.rand(100, 512).astype(np.float32)
            else:
                feat = np.load(path)

            self.features_all.append(feat)
            self.start_indices.append(curr_index)

            end_index = curr_index + feat.shape[0]
            self.video_ranges.append((curr_index, end_index))

            for node_idx in range(curr_index, end_index):
                self.node_to_video_map[node_idx] = i

            curr_index = end_index

        self.x = torch.tensor(
            np.concatenate(self.features_all, axis=0), dtype=torch.float
        )

    def _build_temporal_edges(self):
        """建立時序邊緣 (每部影片內部相鄰幀)。"""
        # 從 config 讀取需要建立雙向邊的影片索引 (例如旋轉影片)
        bidirectional_indices = self.config["graph"].get(
            "bidirectional_video_indices", [16, 22]
        )

        for i, feat in enumerate(self.features_all):
            start = self.start_indices[i]

            # 針對特定旋轉影片建立雙向邊
            if i in bidirectional_indices:
                for j in range(len(feat) - 1):
                    self.edge_list.append([start + j, start + j + 1])
                    self.edge_list.append([start + j + 1, start + j])
            else:
                for j in range(len(feat) - 1):
                    self.edge_list.append([start + j, start + j + 1])

    def _build_cross_video_edges(self):
        """建立跨影片的交接邊緣。"""
        s = self.start_indices

        # 從 config.yaml 讀取跨影片連線
        cross_edges = self.config["graph"].get("cross_video_edges", [])

        try:
            edges = []
            for edge_cfg in cross_edges:
                from_vid, from_frame, to_vid, to_frame = edge_cfg
                from_node = s[from_vid] + from_frame
                to_node = s[to_vid] + to_frame
                edges.append([from_node, to_node])

            self.edge_list.extend(edges)
        except IndexError:
            print("特徵檔數量與設定的跨影片邊緣索引不吻合，無法建立完整的跨影片邊緣。")

    def _categorize_edges(self):
        """將所有建立的邊緣分為時序與跨影片兩類。"""
        for edge in self.edge_list:
            if self.node_to_video_map.get(edge[0]) == self.node_to_video_map.get(
                edge[1]
            ):
                self.temporal_edges.append(edge)
            else:
                self.cross_video_edges.append(edge)
                self.cross_video_nodes.update(edge)

    def build(self) -> Dict[str, Any]:
        """
        執行圖建立流程，並回傳封裝好的結果字典。
        """
        self._load_features_and_nodes()
        self._build_temporal_edges()
        self._build_cross_video_edges()
        self._categorize_edges()

        self.edge_index = (
            torch.tensor(self.edge_list, dtype=torch.long).t().contiguous()
        )
        self.graph_data = Data(x=self.x, edge_index=self.edge_index)

        return {
            "graph_data": self.graph_data,
            "features_all": self.features_all,
            "start_indices": self.start_indices,
            "video_ranges": self.video_ranges,
            "node_to_video_map": self.node_to_video_map,
            "temporal_edges": self.temporal_edges,
            "cross_video_edges": self.cross_video_edges,
            "cross_video_nodes": self.cross_video_nodes,
        }
