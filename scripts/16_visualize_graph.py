import argparse
import os
import sys

import yaml

from visual_navigation_system.graph_builder import GraphBuilder
from visual_navigation_system.visualizer import GraphVisualizer

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def load_config():
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "config.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="產生完整地圖 Graph 視覺化")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="輸出的圖片路徑，若未指定則使用 yaml 中的設定檔路徑",
    )

    args = parser.parse_args()
    config = load_config()

    output_path = (
        args.output
        if args.output
        else config["graph"].get("output_graph_image", "Map/Graph_Visualization.png")
    )

    builder = GraphBuilder(config=config)
    graph_dict = builder.build()

    data = graph_dict["graph_data"]
    print(f"圖建立完成！總共節點數量: {data.num_nodes}, 邊緣數量: {data.num_edges}")

    visualizer = GraphVisualizer(graph_dict, config)

    # 傳入空路徑，僅繪製背景節點與交接點
    visualizer.draw_graph(path_nodes=[], output_path=output_path)


if __name__ == "__main__":
    main()
