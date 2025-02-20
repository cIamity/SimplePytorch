from .config import *

class ExportDialog(QDialog):
    def __init__(self, scene, parent=None):
        super().__init__(parent)
        self.setWindowTitle("导出神经网络")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)   
        self.setGeometry(QApplication.desktop().screen().rect().center().x() - 300,  # 设置窗口位置大小
                         QApplication.desktop().screen().rect().center().y() - 200, 
                         600, 400)      
        
        self.scene = scene

        # 获取 PyTorch 模型
        self.model = self.scene.get_model()

        # 创建文本框
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)

        # 处理无效模型情况
        if self.model is None:
            self.text_edit.setText("未定义模型：无联通路径")
        else:
            self.text_edit.setText(str(self.model))

        # 创建导出按钮
        self.export_button = QPushButton("导出", self)
        self.export_button.clicked.connect(self.save_model)
        self.export_button.setEnabled(self.model is not None)  # 无效模型禁用导出按钮

        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        layout.addWidget(self.export_button)
        self.setLayout(layout)

    # 导出 PyTorch 模型为 .pt 文件
    def save_model(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "保存 PyTorch 模型", "", "PyTorch 模型 (*.pt);;所有文件 (*)", options=options)
        # 没有选择路径则回到导出窗口
        if not file_path:           
            return
        torch.save(self.model, file_path)
        self.accept()  

class TorchModel(nn.Module):
    def __init__(self, graph):
        super(TorchModel, self).__init__()
        # 网络层存储
        self.layers = nn.ModuleDict()
        # 输入输出节点列表
        self.input_nodes = []
        self.output_nodes = []
        # 节点拓扑排序以及前置节点列表
        self.node_topo = []
        self.node_predecessors = {}

        self.build_model(graph)

    # 构建各层
    def build_model(self, graph):
        # 先筛选出有效节点存入valid_nodes
        valid_nodes = set()

        # 获取所有输入输出节点
        input_nodes = [n for n in graph.nodes if n.startswith("Input")]
        output_nodes = [n for n in graph.nodes if n.startswith("Output")]
        
        # 获取所有从输入节点出发可达的节点
        valid_from_inputs = set()
        for input_node in input_nodes:
            valid_from_inputs.update(nx.descendants(graph, input_node))
        valid_from_inputs.update(input_nodes)  

        # 获取所有能到达输出节点的所有节点
        valid_from_outputs = set()
        for output_node in output_nodes:
            valid_from_outputs.update(nx.ancestors(graph, output_node))
        valid_from_outputs.update(output_nodes)  

        # 计算交集：同时在输入可达路径 和 输出可达路径上的节点
        valid_nodes = valid_from_inputs.intersection(valid_from_outputs)

    
        # 获取并排序有效的输入输出节点
        self.input_nodes = sorted(
            [node for node in graph.nodes if node.startswith("Input") and node in valid_nodes],  
            key=lambda x: int(x.split('_')[-1])
        )
        self.output_nodes = sorted(
            [node for node in graph.nodes if node.startswith("Output") and node in valid_nodes],  
            key=lambda x: int(x.split('_')[-1])
        )

        # 确定节点连接关系以及前向传播方法
        for node in nx.topological_sort(graph):
            # 跳过无效节点
            if node not in valid_nodes:
                continue

            # 存储拓扑排序的节点以及其前置节点列表
            self.node_topo.append(node)
            self.node_predecessors[node] = [p for p in graph.predecessors(node) if p in valid_nodes]

            # 存储前向传播方法
            module_item = graph.nodes[node]["item"]
            layer_name = module_item.layer_name

            # 输入输出层不需要前向传播方法
            if layer_name == "Input" or layer_name == "Output":
                continue    

            #  网络层确定的前向传播方法
            elif "nn" in layer_config[layer_name]:
               self.layers[node] = module_item.nn  

            else:
                raise ValueError(f"模块 {node} 没有前向传播方法")

    # 前向传播
    def forward(self, *inputs):
        outputs = {}    # 记录所有节点的输出

        # 确定输入层的输出
        for i, node in enumerate(self.input_nodes):
            outputs[node] = inputs[i]

        # 按拓扑结构确定其余节点输出
        for node in self.node_topo:
            # 节点是输入模块
            if node in self.input_nodes:
                continue

            input_tensors = [outputs[p] for p in self.node_predecessors[node]]

            # 节点是输出模块
            if node in self.output_nodes:
                outputs[node] = input_tensors[0]

            # 节点是网络层
            else:
                outputs[node] = self.layers[node](*input_tensors)

        return tuple(outputs[node] for node in self.output_nodes)