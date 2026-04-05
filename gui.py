import sys
from PyQt5.QtWidgets import *
import json
from utils.config import load_config, save_config
import subprocess
import os

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LANG"] = "zh_CN.UTF-8"
os.environ["LC_ALL"] = "zh_CN.UTF-8"

class CellposePanel(QWidget):
    def __init__(self):
        super().__init__()

        layout = QFormLayout()

        self.model = QComboBox()
        self.model.addItems(["cyto3", "cyto2", "nuclei"])

        self.device = QComboBox()
        self.device.addItems(["cpu", "cuda", "mps"])

        self.auto_diam = QCheckBox()

        self.cellprob = QDoubleSpinBox()
        self.cellprob.setRange(-6, 6)
        self.cellprob.setValue(0.0)

        self.flow = QDoubleSpinBox()
        self.flow.setRange(0, 1)
        self.flow.setValue(0.4)

        layout.addRow("模型:", self.model)
        layout.addRow("设备:", self.device)
        layout.addRow("自动直径:", self.auto_diam)
        layout.addRow("cellprob:", self.cellprob)
        layout.addRow("flow:", self.flow)

        self.setLayout(layout)

    def get_params(self):
        return {
            "model_type": self.model.currentText(),
            "device": self.device.currentText(),
            "auto_diameter": self.auto_diam.isChecked(),
            "cellprob_threshold": self.cellprob.value(),
            "flow_threshold": self.flow.value()
        }

class RegionSelector(QWidget):
    def __init__(self, ontology_path):
        super().__init__()

        self.ontology_path = ontology_path
        self.data = self.load_json()

        layout = QVBoxLayout()

        self.input = QLineEdit()
        self.input.setPlaceholderText("输入脑区（如 CA1, VISp）")
        layout.addWidget(self.input)

        self.search_btn = QPushButton("搜索")
        self.search_btn.clicked.connect(self.search_region)
        layout.addWidget(self.search_btn)

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        self.manual_input = QLineEdit()
        self.manual_input.setPlaceholderText("或手动输入脑区代号，多个脑区请用空格分隔")
        layout.addWidget(self.manual_input)

        self.select_btn = QPushButton("选择脑区")
        self.select_btn.clicked.connect(self.select_region)
        self.list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        layout.addWidget(self.select_btn)

        self.setLayout(layout)

        self.selected_ids = []

    def load_json(self):
        with open(self.ontology_path) as f:
            return json.load(f)

    def search(self, node, query, results):
        name = node.get("data", {}).get("name", "").lower()
        acronym = node.get("data", {}).get("acronym", "").lower()

        if query in name or query in acronym:
            results.append(node)

        for c in node.get("children", []):
            self.search(c, query, results)

    def search_region(self):
        query = self.input.text().lower()
        results = []
        self.search(self.data["root"], query, results)

        self.list_widget.clear()

        for r in results:
            d = r["data"]
            text = f"{d['name']} ({d['acronym']}) | id={r['id']}"
            self.list_widget.addItem(text)
            self.list_widget.item(self.list_widget.count()-1).setData(1, r["id"])

    def select_region(self):
        items = self.list_widget.selectedItems()

        selected_ids = [item.data(1) for item in items]

        manual_text = self.manual_input.text().strip()
        if manual_text:
            try:
                manual_ids = list(map(int, manual_text.split()))
                selected_ids.extend(manual_ids)
            except ValueError:
                QMessageBox.warning(self, "错误", "手动输入必须是数字，用空格分隔")
                return

        if not selected_ids:
            QMessageBox.warning(self, "提示", "请至少选择或输入一个脑区")
            return

        self.selected_ids = list(set(selected_ids))
        QMessageBox.information(self,"选择成功", f"已选择 {len(self.selected_ids)} 个脑区")


class GUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Brain Analysis Pro")

        layout = QVBoxLayout()

        # 路径
        self.btn_path = QPushButton("选择数据路径")
        self.btn_path.clicked.connect(self.select_path)
        layout.addWidget(self.btn_path)

        self.cfg = load_config()

        # 脑区
        self.region_selector = RegionSelector(self.cfg["ontology_json"])
        layout.addWidget(self.region_selector)

        # 参数面板
        self.cellpose_panel = CellposePanel()
        layout.addWidget(self.cellpose_panel)

        # 运行按钮
        self.btn_run = QPushButton("运行分析")
        self.btn_run.clicked.connect(self.run_pipeline)
        layout.addWidget(self.btn_run)

        # 进度条 ⭐
        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        # 日志 ⭐
        self.log = QTextEdit()
        layout.addWidget(self.log)

        self.first_run_checkbox = QCheckBox("初次启动（安装环境）")
        layout.addWidget(self.first_run_checkbox)

        self.setLayout(layout)

    def select_path(self):
        path = QFileDialog.getExistingDirectory()

        self.cfg = load_config()
        self.cfg["image_dir"] = path
        save_config(self.cfg)

        self.log.append(f"路径已设置: {path}")

    def run_pipeline(self):
        if self.first_run_checkbox.isChecked():
            self.log.append("正在检查运行环境...")
            try:
                from main import check_env
                check_env()
                self.log.append("环境检查完成")
                self.first_run_checkbox.setChecked(False)
            except Exception as e:
                self.log.append(f"环境检查失败: {str(e)}")
                return
        
        self.log.append("开始运行 pipeline...")
        rids = self.region_selector.selected_ids
        if not rids:
            self.log.append("请先选择脑区")
            return
        
        params = self.cellpose_panel.get_params()

        cmd = [sys.executable,
            os.path.join(os.path.dirname(__file__), "main.py"),
            "--region-ids", ' '.join(map(str, rids)),
            "--model-type", params["model_type"],
            "--device", params["device"],
            "--cellprob-threshold", str(params["cellprob_threshold"]),
            "--flow-threshold", str(params["flow_threshold"]),
            "--auto-diameter", str(params["auto_diameter"])]
            
        try:
            subprocess.Popen(cmd)
            self.log.append("分析进程已启动")
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
            self.log.append(f"启动失败: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = GUI()
    win.show()
    sys.exit(app.exec_())