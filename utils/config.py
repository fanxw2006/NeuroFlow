import os
import yaml

CONFIG_PATH = "config.yaml"

def create_default():
    cfg = {
        "image_dir": os.getcwd(),
        "ontology_json": "./atlas/allen_mouse_10um_java-Ontology.json",
        "output_dir": "./output",

        "cellpose": {
            "model_type": "cyto3",
            "device": "cpu"
        },

        "atlas": {
            "resolution": 25
        }
    }

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f)

    print("已生成 config.yaml，请修改路径后重新运行")


def load_config():
    if not os.path.exists(CONFIG_PATH):
        create_default()
        exit()

    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f)