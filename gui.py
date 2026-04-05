import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtCore import QMetaObject, Q_ARG
import json
from utils.config import load_config, save_config
from utils.tiff_converter import batch_convert_tiff_to_png
import subprocess
import os

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LANG"] = "zh_CN.UTF-8"
os.environ["LC_ALL"] = "zh_CN.UTF-8"

# 版本信息
APP_NAME = "NeuroFlow"
APP_VERSION = "v1.3.1"

# 颜色方案
PRIMARY_COLOR = "#4CAF50"
SECONDARY_COLOR = "#2196F3"
BACKGROUND_COLOR = "#F5F5F5"
CARD_COLOR = "#FFFFFF"
TEXT_COLOR = "#333333"
BORDER_COLOR = "#E0E0E0"

class CellposePanel(QWidget):
    def __init__(self):
        super().__init__()

        # 设置样式
        self.setStyleSheet(
            "QWidget {" 
            "    background-color: " + CARD_COLOR + ";" 
            "    border-radius: 8px;" 
            "    padding: 15px;" 
            "}" 
            "QLabel {" 
            "    font-size: 14px;" 
            "    color: " + TEXT_COLOR + ";" 
            "    font-weight: 500;" 
            "}" 
            "QComboBox {" 
            "    border: 1px solid " + BORDER_COLOR + ";" 
            "    border-radius: 4px;" 
            "    padding: 8px;" 
            "    font-size: 14px;" 
            "    background-color: white;" 
            "}" 
            "QComboBox:hover {" 
            "    border-color: " + PRIMARY_COLOR + ";" 
            "}" 
            "QDoubleSpinBox {" 
            "    border: 1px solid " + BORDER_COLOR + ";" 
            "    border-radius: 4px;" 
            "    padding: 8px;" 
            "    font-size: 14px;" 
            "    background-color: white;" 
            "}" 
            "QDoubleSpinBox:hover {" 
            "    border-color: " + PRIMARY_COLOR + ";" 
            "}" 
            "QCheckBox {" 
            "    font-size: 14px;" 
            "    color: " + TEXT_COLOR + ";" 
            "}" 
        )

        layout = QFormLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        self.model = QComboBox()
        self.model.addItems(["cyto3", "cyto2", "nuclei"])

        self.device = QComboBox()
        self.device.addItems(["cpu", "cuda", "mps"])

        self.auto_diam = QCheckBox()

        self.cellprob = QDoubleSpinBox()
        self.cellprob.setRange(-6, 6)
        self.cellprob.setValue(0.0)
        # 安装事件过滤器禁用滚轮
        self.cellprob.installEventFilter(self)

        self.flow = QDoubleSpinBox()
        self.flow.setRange(0, 1)
        self.flow.setValue(0.4)
        # 安装事件过滤器禁用滚轮
        self.flow.installEventFilter(self)

        layout.addRow("模型类型:", self.model)
        layout.addRow("计算设备:", self.device)
        layout.addRow("自动直径:", self.auto_diam)
        layout.addRow("CellProb阈值:", self.cellprob)
        layout.addRow("Flow阈值:", self.flow)

        self.setLayout(layout)
    
    def eventFilter(self, obj, event):
        """事件过滤器，禁用滚轮事件"""
        if event.type() == QEvent.Wheel:
            # 过滤掉滚轮事件
            if obj == self.cellprob or obj == self.flow:
                return True
        return super().eventFilter(obj, event)

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

        # 设置样式
        self.setStyleSheet(
            "QWidget {" 
            "    background-color: " + CARD_COLOR + ";" 
            "    border-radius: 8px;" 
            "    padding: 15px;" 
            "}" 
            "QLabel {" 
            "    font-size: 14px;" 
            "    color: " + TEXT_COLOR + ";" 
            "    font-weight: 500;" 
            "}" 
            "QPushButton {" 
            "    background-color: " + PRIMARY_COLOR + ";" 
            "    color: white;" 
            "    border: none;" 
            "    border-radius: 4px;" 
            "    padding: 8px 16px;" 
            "    font-size: 14px;" 
            "    font-weight: 500;" 
            "}" 
            "QPushButton:hover {" 
            "    background-color: #45a049;" 
            "}" 
            "QLineEdit {" 
            "    border: 1px solid " + BORDER_COLOR + ";" 
            "    border-radius: 4px;" 
            "    padding: 8px;" 
            "    font-size: 14px;" 
            "}" 
            "QLineEdit:hover {" 
            "    border-color: " + PRIMARY_COLOR + ";" 
            "}" 
            "QListWidget {" 
            "    border: 1px solid " + BORDER_COLOR + ";" 
            "    border-radius: 4px;" 
            "    font-size: 14px;" 
            "    min-height: 150px;" 
            "}" 
            "QListWidget:hover {" 
            "    border-color: " + PRIMARY_COLOR + ";" 
            "}" 
            "QListWidget::item:selected {" 
            "    background-color: " + PRIMARY_COLOR + ";" 
            "    color: white;" 
            "}" 
        )

        self.ontology_path = ontology_path
        self.data = self.load_json()

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # 添加标题
        title_label = QLabel("脑区选择")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: " + TEXT_COLOR + ";")
        layout.addWidget(title_label)

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

        # 设置窗口标题和大小
        self.setWindowTitle(f"{APP_NAME} - {APP_VERSION}")
        self.setMinimumSize(800, 700)
        
        # 设置主样式
        self.setStyleSheet(
            "QWidget {" 
            "    background-color: " + BACKGROUND_COLOR + ";" 
            "    color: " + TEXT_COLOR + ";" 
            "}" 
            "QPushButton {" 
            "    background-color: " + PRIMARY_COLOR + ";" 
            "    color: white;" 
            "    border: none;" 
            "    border-radius: 6px;" 
            "    padding: 10px 20px;" 
            "    font-size: 14px;" 
            "    font-weight: 500;" 
            "}" 
            "QPushButton:hover {" 
            "    background-color: #45a049;" 
            "}" 
            "QPushButton:disabled {" 
            "    background-color: #9E9E9E;" 
            "}" 
            "QProgressBar {" 
            "    border: 1px solid " + BORDER_COLOR + ";" 
            "    border-radius: 4px;" 
            "    text-align: center;" 
            "    height: 20px;" 
            "}" 
            "QProgressBar::chunk {" 
            "    background-color: " + PRIMARY_COLOR + ";" 
            "    border-radius: 4px;" 
            "}" 
            "QTextEdit {" 
            "    border: 1px solid " + BORDER_COLOR + ";" 
            "    border-radius: 4px;" 
            "    font-size: 13px;" 
            "    font-family: Consolas, monospace;" 
            "}" 
            "QCheckBox {" 
            "    font-size: 14px;" 
            "    color: " + TEXT_COLOR + ";" 
            "}" 
        )

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # 滚动内容 widget
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(20, 20, 20, 20)
        scroll_layout.setSpacing(20)

        # 头部标题
        header_layout = QHBoxLayout()
        header_layout.setSpacing(10)
        
        # 标题标签
        title_label = QLabel(f"{APP_NAME}")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: " + PRIMARY_COLOR + ";")
        
        # 版本标签
        version_label = QLabel(APP_VERSION)
        version_label.setStyleSheet("font-size: 14px; color: " + TEXT_COLOR + ";")
        
        # 环境检查选项
        self.first_run_checkbox = QCheckBox("初次启动（安装环境）")
        
        # 占位符
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(version_label)
        header_layout.addItem(spacer)
        header_layout.addWidget(self.first_run_checkbox)
        scroll_layout.addLayout(header_layout)

        # 数据路径选择
        path_layout = QVBoxLayout()
        path_layout.setSpacing(10)
        
        path_label = QLabel("数据路径")
        path_label.setStyleSheet("font-size: 16px; font-weight: 500; color: " + TEXT_COLOR + ";")
        path_layout.addWidget(path_label)
        
        self.btn_path = QPushButton("选择数据路径")
        self.btn_path.setMinimumHeight(40)
        self.btn_path.clicked.connect(self.select_path)
        path_layout.addWidget(self.btn_path)
        
        scroll_layout.addLayout(path_layout)

        self.cfg = load_config()

        # 脑区选择
        scroll_layout.addWidget(self._create_section_label("脑区选择"))
        self.region_selector = RegionSelector(self.cfg["ontology_json"])
        scroll_layout.addWidget(self.region_selector)

        # 参数面板
        scroll_layout.addWidget(self._create_section_label("Cellpose 参数"))
        self.cellpose_panel = CellposePanel()
        scroll_layout.addWidget(self.cellpose_panel)

        # 运行控制
        control_layout = QVBoxLayout()
        control_layout.setSpacing(10)
        
        # 运行按钮
        self.btn_run = QPushButton("运行分析")
        self.btn_run.setMinimumHeight(45)
        self.btn_run.setStyleSheet(
            "QPushButton {" 
            "    background-color: " + SECONDARY_COLOR + ";" 
            "    font-size: 16px;" 
            "    padding: 12px;" 
            "}" 
            "QPushButton:hover {" 
            "    background-color: #1976D2;" 
            "}"
        )
        self.btn_run.clicked.connect(self.run_pipeline)
        control_layout.addWidget(self.btn_run)
        
        scroll_layout.addLayout(control_layout)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setValue(0)
        scroll_layout.addWidget(self.progress)

        # 日志
        log_label = QLabel("运行日志")
        log_label.setStyleSheet("font-size: 14px; font-weight: 500; color: " + TEXT_COLOR + ";")
        scroll_layout.addWidget(log_label)
        
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(200)
        scroll_layout.addWidget(self.log)

        # 添加底部间距
        scroll_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # 设置滚动内容
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        self.setLayout(main_layout)
    
    def _create_section_label(self, text):
        """创建章节标题"""
        label = QLabel(text)
        label.setStyleSheet("font-size: 16px; font-weight: 500; color: " + TEXT_COLOR + "; margin-top: 10px;")
        return label

    def select_path(self):
        path = QFileDialog.getExistingDirectory()

        self.cfg = load_config()
        self.cfg["image_dir"] = path
        save_config(self.cfg)

        self.log.append(f"路径已设置: {path}")

    def run_pipeline(self):
        skip_env_check = not self.first_run_checkbox.isChecked()
        
        if self.first_run_checkbox.isChecked():
            self.log.append("正在检查运行环境...")
            try:
                from main import check_env
                check_env()
                self.log.append("环境检查完成")
            except Exception as e:
                self.log.append(f"环境检查失败: {str(e)}")
                return
        
        self.log.append("开始运行 pipeline...")
        rids = self.region_selector.selected_ids
        if not rids:
            self.log.append("请先选择脑区")
            return
        
        # 检查并转换TIFF文件
        self.log.append("检查并转换TIFF文件为PNG格式...")
        try:
            converted_files = batch_convert_tiff_to_png(self.cfg["image_dir"])
            if converted_files:
                self.log.append(f"已转换 {len(converted_files)} 个TIFF文件")
            else:
                self.log.append("无需转换TIFF文件")
        except Exception as e:
            self.log.append(f"TIFF转换失败: {str(e)}")
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
        
        if skip_env_check:
            cmd.append("--skip-env-check")
            
        try:
            # 启动子进程并捕获输出
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8'
            )
            
            # 实时读取输出并显示到日志
            self.log.append("分析进程已启动，正在运行...")
            
            # 创建一个线程来读取输出
            def read_output():
                while True:
                    line = process.stdout.readline()
                    if not line:
                        break
                    # 在主线程中更新日志
                    QMetaObject.invokeMethod(
                        self.log,
                        "append",
                        Qt.QueuedConnection,
                        Q_ARG(str, line.strip())
                    )
                
                # 进程结束后检查退出状态
                exit_code = process.wait()
                QMetaObject.invokeMethod(
                    self.log,
                    "append",
                    Qt.QueuedConnection,
                    Q_ARG(str, "分析完成！" if exit_code == 0 else f"分析失败，退出码: {exit_code}")
                )
            
            # 启动线程
            import threading
            output_thread = threading.Thread(target=read_output)
            output_thread.daemon = True
            output_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))
            self.log.append(f"启动失败: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = GUI()
    win.show()
    sys.exit(app.exec_())