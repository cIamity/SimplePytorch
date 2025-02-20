from .config import *
from .canvas_window import GraphicsScene, GraphicsView
from .canvas_item import LineItem
from .export import ExportDialog

#todo：print替换;

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.file_path = None   # 工程文件路径
        self.setupUi(self)

    def setupUi(self, MainWindow):
        # 主窗口
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1536, 864)
        MainWindow.setStyleSheet("background-color: #d0e7ff;")  # 浅蓝色背景
        MainWindow.setWindowTitle("SimplePytorch")

        # 中心窗口
        self.central_widget = QtWidgets.QWidget(MainWindow)
        self.central_widget.setObjectName("central_widget")
        self.central_layout = QtWidgets.QGridLayout(self.central_widget)
        self.central_layout.setObjectName("central_layout")
        self.central_layout.setContentsMargins(10, 10, 10, 10)  # 设置中心窗口的整体外边距
        
        # 水平分割器
        self.central_splitter = QSplitter(QtCore.Qt.Horizontal, self.central_widget)
        self.central_splitter.setObjectName("central_splitter")
        self.central_splitter.setHandleWidth(8)  # 设置拖拽区域宽度

        #region 菜单栏
        menu_bar = QMenuBar(MainWindow)
        self.setMenuBar(menu_bar)

        # 创建“文件”菜单
        file_menu = QMenu("文件", self)
        menu_bar.addMenu(file_menu)

        # 添加“打开”选项
        open_action = QAction("打开", self)
        file_menu.addAction(open_action)
        open_action.triggered.connect(self.open_project)

        # 添加“保存”选项
        save_action = QAction("保存", self)
        save_action.setShortcut("Ctrl+S")  # 绑定快捷键 Ctrl+S
        file_menu.addAction(save_action)
        save_action.triggered.connect(self.save_project)

        # 添加“另存为”选项
        save_as_action = QAction("另存为", self)
        file_menu.addAction(save_as_action)
        save_as_action.triggered.connect(self.save_project_as)

        # 添加“导出”选项
        export_action = QAction("导出", self)
        file_menu.addAction(export_action)
        export_action.triggered.connect(self.export_model)

        #endregion


        #region 模块区
        self.module_widget = QWidget(self.central_splitter)
        self.module_widget.setObjectName("module_widget")
        self.module_widget.setStyleSheet("border: 0px solid blue; border-radius: 5px; background-color: white;")  # 模块区最外面有蓝色描边，白色背景
        self.module_layout = QVBoxLayout(self.module_widget)
        self.module_layout.setObjectName("module_layout")
        self.module_layout.setContentsMargins(0, 0, 0, 0)  # 去掉内边距
        self.module_layout.setSpacing(0)  # 去掉控件间距

        # 顶部类别选择下拉框
        self.category_selector = CenteredComboBox(self.module_widget)
        self.category_selector.addItem("All Layers")  # 默认显示所有模块
        self.category_selector.addItems(sorted(set(layer["type"] for layer in layer_config.values())))

        # 列表组件
        self.module_list = QListWidget(self.module_widget)
        self.module_list.setObjectName("module_list")
        self.module_list.setVerticalScrollBar(CustomScrollBar(self.module_list))  # 使用自定义滚动条
        self.module_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 关闭水平滚动条
        self.module_list.setStyleSheet(
            "QListWidget { border: none; padding: 5px; background-color: white; outline: none;}"    # 无边框，内边距，白色背景, 无虚线框
            "QListWidget::item { padding: 3px; }"                                                   # 列表项内边距
            "QListWidget::item:selected { background-color: #0078d7; color: white; }"               # 选中项背景色，字体颜色
        )

        self.update_module_list()
        self.category_selector.currentTextChanged.connect(self.update_module_list)
        self.module_list.itemSelectionChanged.connect(self.on_module_selected)
        #endregion


        #region 画布区
        self.canvas_scene = GraphicsScene(self)
        self.canvas_view = GraphicsView(self.central_splitter)
        self.canvas_view.setScene(self.canvas_scene)
        self.canvas_view.setStyleSheet("background-color: #f0f0f0;")  # 设定背景色
        #endregion


        #region 主布局设置
        MainWindow.setCentralWidget(self.central_widget)
        self.central_layout.addWidget(self.central_splitter)
        self.module_layout.addWidget(self.category_selector)
        self.module_layout.addWidget(self.module_list)

        # 后续初始化设置
        self.central_splitter.setSizes([100, 1000])  # 设置水平分割器比例
        
        # 修改菜单栏样式
        menu_bar.setStyleSheet("""
            QMenuBar {
                background-color: white; /* 设置菜单栏背景为白色 /
            }
            QMenuBar::item {
                background: transparent;
                padding: 0px 0px; /* 上下内边距 左右内边距 */
                margin: 0px; /* 移除额外间距 */
            }
            QMenuBar::item:selected {
                background: rgba(30, 144, 255, 0.3); /* 选中时背景变为透明的浅蓝色 */
                color: black; /* 选中时文本仍然为黑色 */
                padding: 0px 0px; /* 上下内边距 左右内边距 */
            }
        """)

        # 修改文件下拉菜单栏样式
        file_menu.setStyleSheet("""
            QMenu {
                background-color: #f0f0f0; /* 设置菜单栏背景为白色 /

            }
            QMenu::item {
                background: transparent;
                padding: 0px 0px; /* 上下内边距 左右内边距 */
                margin: 0px; /* 移除额外间距 */
            }
            QMenu::item:selected {
                background: rgba(30, 144, 255, 0.3); /*选中时背景变为透明的浅蓝色 */
                color: black; /* 选中时文本仍然为黑色 */
            }
        """)
        #endregion


    #region 其他函数
    
    # 保存行为函数
    def save_project(self):
        if self.file_path:
            file_path = self.file_path
        else:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, "保存项目", "", "JSON 文件 (*.json);;所有文件 (*)", options=options)

            if not file_path:
                return  # 用户未选择文件，直接返回
            
        # 生成保存数据
        project_data = {
            "counter": self.canvas_scene.counter,
            "modules": [],
            "lines": []
        }
    
        # 遍历所有模块
        for node in self.canvas_scene.G.nodes:
            item = self.canvas_scene.G.nodes[node]["item"]
            module_data = {
                "name": item.name,
                "layer_name": item.layer_name,
                "position": (item.x(), item.y()),
                "parameters": {}
            }
            
            # 处理神经网络层参数
            if item.nn:
                params = inspect.signature(item.nn.__class__).parameters

                for param_name in params:
                    if param_name == "self":
                        continue

                    if hasattr(item.nn, param_name):
                        param_value = getattr(item.nn, param_name)

                        # 转换逻辑
                        if isinstance(param_value, (int, float, str, bool)):  
                            module_data["parameters"][param_name] = {"__type__": type(param_value).__name__, "value": param_value}

                        elif isinstance(param_value, (list, tuple)):  
                            module_data["parameters"][param_name] = {"__type__": type(param_value).__name__, "value": list(param_value)}

                        elif isinstance(param_value, np.ndarray):  
                            module_data["parameters"][param_name] = {"__type__": "ndarray", "value": param_value.tolist()}

                        elif isinstance(param_value, torch.nn.Parameter):  
                            continue  #  不保存权重

                        else:
                            print(f"无法序列化参数: {param_name} ({type(param_value)})，跳过")

            project_data["modules"].append(module_data)

        # 遍历所有连线
        for start, end, data in self.canvas_scene.G.edges(data=True):
            project_data["lines"].append({
                "start": start,
                "end": end,
                "anchors": data["line"].anchors  # 存储连线的锚点
            })

        # 将数据写入 JSON 文件
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(project_data, f, indent=4)    # 缩进为4空格
            QMessageBox.information(self, "保存成功", "项目已成功保存！")
            self.file_path = file_path
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"无法保存文件：{str(e)}")

    # 另存为行为函数
    def save_project_as(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "另存为", "", "JSON 文件 (*.json);;所有文件 (*)", options=options
        )

        if not file_path:
            return  # 用户取消操作

        self.file_path = file_path  # 更新当前文件路径
        self.save_project()  # 调用 save_project 进行保存

    # 打开行为函数
    def open_project(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "打开项目", "", "JSON 文件 (*.json);;所有文件 (*)", options=options)

        if not file_path:
            return  # 用户未选择文件，直接返回

        # 读取 JSON 文件
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                project_data = json.load(f)
            
            # 清空当前画布
            self.canvas_scene.clear()
            self.canvas_scene.G.clear()

            # 重新加载模块
            for module_data in project_data["modules"]:
                module_name = module_data["name"]
                layer_name = module_data["layer_name"]
                x, y = module_data["position"]

                # 创建模块
                module_item = self.canvas_scene.create_item(layer_name)
                module_item.name = module_name
                module_item.setPos(x, y)
                self.canvas_scene.addItem(module_item)
                self.canvas_scene.G.add_node(module_name, item=module_item)

                # 还原参数
                if "parameters" in module_data and module_item.nn:
                    layer_class = type(module_item.nn)  # 获取当前层的类
                    module_params = {}  # 保存初始化参数
                    for param_name, param_data in module_data["parameters"].items():
                        if not isinstance(param_data, dict) or "__type__" not in param_data:
                            restored_value = param_data  # 直接返回普通值
                        else:
                            param_type = param_data["__type__"]
                            value = param_data["value"]

                            # 直接写恢复逻辑
                            if param_type == "int":
                                restored_value = int(value)
                            elif param_type == "float":
                                restored_value = float(value)
                            elif param_type == "str":
                                restored_value = str(value)
                            elif param_type == "bool":
                                restored_value = bool(value)
                            elif param_type == "list":
                                restored_value = list(value)
                            elif param_type == "tuple":
                                restored_value = tuple(value)
                            elif param_type == "ndarray":
                                restored_value = np.array(value)
                            else:
                                print(f"未知类型 {param_type}，默认返回原值")
                                restored_value = value

                        module_params[param_name] = restored_value

                    module_item.nn = layer_class(**module_params)

            # 重新加载连线
            for line in project_data["lines"]:
                start_name = line["start"]
                end_name = line["end"]
                anchors = line["anchors"]

                # 恢复参数
                line_item = LineItem(anchors[0])  # 以起点创建连线
                line_item.anchors = anchors  # 直接恢复锚点
                line_item.start_name = start_name
                line_item.end_name = end_name
                line_item.done = True
                
                self.canvas_scene.addItem(line_item)
                line_item.update_path(update_arrow=True)
                self.canvas_scene.G.add_edge(start_name, end_name, line=line_item)

            # 重新加载计数器
            self.canvas_scene.counter = project_data.get("counter", 0)

            QMessageBox.information(self, "打开成功", "项目已成功加载！")

            self.file_path = file_path

        except Exception as e:
            QMessageBox.critical(self, "格式错误", f"无法解析文件：{str(e)}")
    
    # 导出行为函数
    def export_model(self):
        self.export_dialog = ExportDialog(self.canvas_scene)
        self.export_dialog.exec_()
    
    # 更新模块区列表项
    def update_module_list(self):
        """更新模块列表，考虑类别选择"""
        selected_category = self.category_selector.currentText()
        self.module_list.clear()

        for module_name, module_info in layer_config.items():
            if selected_category != "All Layers" and module_info["type"] != selected_category:
                continue  # 过滤掉不属于当前类别的模块

            item = QListWidgetItem(module_name)
            self.module_list.addItem(item)
    
    # 槽函数，列表变化时触发
    def on_module_selected(self):
        selected_item = self.module_list.currentItem()
        if selected_item:
            self.canvas_scene.set_selected_module(selected_item.text())

    #endregion

#region 自定义组件
# 自定义模块区顶部类别选择下拉框
class CenteredComboBox(QComboBox):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)  # 设置高度
        self.setMaxVisibleItems(11)  # 下拉列表显示 11 个项
        self.setItemDelegate(CustomDelegate())  # 设定自定义字体
        self.setStyleSheet("""
            /* 主体样式 */
            QComboBox {
                font-style: italic;     /* 斜体 */
                font-weight: bold;      /* 粗体 */
                font-size: 18px;        /* 字号 */
                font-family: 'Arial';   /* 字体 */
                padding: 0px;           /* 内边距 */
                background-color: transparent; /* 让背景透明，由 paintEvent 处理 */
            }

            /* 下拉按钮 */
            QComboBox::drop-down { 
                border: none; 
                background: transparent; 
            }
            QComboBox::down-arrow { 
                image: none;  /* 隐藏下拉箭头 */
            }

            /* 下拉框样式 */
            QComboBox QAbstractItemView {
                border: 1px solid #1e90ff;  /* 边框颜色 */
                background-color: white;  /* 背景色 */
                selection-background-color: #8fcafe;  /* 选中项背景色 */
                selection-color: black;  /* 选中项文本颜色 */
                outline: 0;  /* 取消选中项的虚线框 */
            }
        """)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制背景（蓝色渐变）
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(143, 202, 254))  # 浅蓝色
        gradient.setColorAt(1, QColor(30, 144, 255))  # 深蓝色
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        rect = self.rect()
        painter.drawRoundedRect(rect, 3, 3)  # 圆角半径 

        # 绘制文本
        text = self.currentText()
        painter.setPen(QColor(0, 0, 0))  # 文字颜色：黑色
        painter.drawText(rect, Qt.AlignCenter, text)  # 文字居中绘制

        painter.end()

# 自定义下拉框字体
class CustomDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        # 设置字体样式
        font = QFont("Arial", 10)  # 字体 & 字号
        font.setBold(False)         # 设置粗体
        font.setItalic(True)       # 设置斜体
        font.setUnderline(False)   # 是否添加下划线
        font.setStrikeOut(False)   # 是否添加删除线
        
        option.font = font  # 应用字体

# 自定义滚动条
class CustomScrollBar(QScrollBar):
    def __init__(self, parent=None):
        super().__init__(Qt.Vertical, parent)
        self.setStyleSheet("""
            QScrollBar:vertical {
                border: none;
                background: transparent;  /* 透明背景 */
                width: 10px;  /* 设置滚动条宽度 */
                margin: 0px 0px 0px 0px;  /* 确保紧贴右侧 */
            }
            QScrollBar::handle:vertical {
                background: rgba(30, 144, 255, 150); /* 滑块颜色 */
                border-radius: 4px; /* 圆角 */
                min-height: 20px; /* 确保滑块不会消失 */
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                background: none; /* 移除上下箭头 */
            }
        """)
#endregion

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())

def start():
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())
    