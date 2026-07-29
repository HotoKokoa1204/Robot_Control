import argparse
import os
import sys

import numpy as np
import torch
import yaml

from agilab.tku.visual_navigation_system.graph_builder import GraphBuilder
from agilab.tku.visual_navigation_system.visualizer import GraphVisualizer

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
)


def load_config():
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "configs", "config.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="計算最短路徑並進行視覺化")
    parser.add_argument(
        "--start_feat", type=str, default=None, help="起點的特徵向量路徑 (.npy)"
    )
    parser.add_argument(
        "--end_feat", type=str, default=None, help="終點的特徵向量路徑 (.npy)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="輸出的圖片路徑，若未指定則使用 yaml 中的設定",
    )

    args = parser.parse_args()
    config = load_config()

    output_path = (
        args.output
        if args.output
        else config["graph"].get("output_shortest_path_image", "Map/Shortest_Path.png")
    )

    builder = GraphBuilder(config=config)
    graph_dict = builder.build()
    data = graph_dict["graph_data"]

    if data.num_nodes == 0:
        print("地圖中沒有節點，無法計算最短路徑！")
        return

    # 如果使用者沒有指定起終點，從 Graph 中首尾抓取特徵作為展示
    if args.start_feat and os.path.exists(args.start_feat):
        start_feature = torch.tensor(np.load(args.start_feat)[0], dtype=torch.float)
    else:
        print("未指定起點或起點檔案不存在，將使用第一部影片的第一幀作為起點。")
        start_feature = data.x[0]

    if args.end_feat and os.path.exists(args.end_feat):
        end_feature = torch.tensor(np.load(args.end_feat)[0], dtype=torch.float)
    else:
        print("未指定終點或終點檔案不存在，將使用最後一部影片的最後一幀作為終點。")
        end_feature = data.x[-1]

    visualizer = GraphVisualizer(graph_dict, config)

    path_nodes = visualizer.find_shortest_path(start_feature, end_feature)

    if path_nodes:
        visualizer.draw_graph(path_nodes=path_nodes, output_path=output_path)
    else:
        print("無法找到路徑，僅能輸出背景圖。")
        visualizer.draw_graph(path_nodes=[], output_path=output_path)


if __name__ == "__main__":
    main()
