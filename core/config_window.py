from .config import *

class ModuleConfigDialog(QDialog):
    def __init__(self, module_item, parent=None):
        super().__init__(parent)
        self.module_item = module_item
        self.setWindowTitle(f"{module_item.layer_name} 配置")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint) 
        self.setGeometry(QApplication.desktop().screen().rect().center().x() - 300,  # 设置窗口位置大小
                         QApplication.desktop().screen().rect().center().y() - 200, 
                         600, 400)      

        layout = QVBoxLayout()

        # 模块名称
        self.name_label = QLabel(f"模块名称: {module_item.name}")
        self.name_label.setWordWrap(True)  # 允许自动换行
        self.name_label.setFixedHeight(25)  # 限制高度为 1 行
        layout.addWidget(self.name_label)

        # 模块介绍
        introduction = layer_config.get(module_item.layer_name, {}).get("introduction", "暂无介绍")
        self.intro_label = QLabel(f"模块介绍: {introduction}")
        self.intro_label.setWordWrap(True)      # 自动换行
        self.intro_label.setFixedHeight(40)  # 限制高度为 1 行（最多两行）
        layout.addWidget(self.intro_label)

        # 参数编辑区
        self.param_inputs = {}
        # 若有可配置参数
        if module_item.nn:
            form_layout = QFormLayout()

            # 获取与构造函数相关参数
            params = inspect.signature(module_item.nn.__class__).parameters

            for name in params:
                if name == "self":  # 跳过'self'
                    continue
                
                if hasattr(module_item.nn, name):
                    value = getattr(module_item.nn, name)

                    if isinstance(value, bool):  
                        # 布尔类型 → 下拉菜单
                        param_input = QComboBox()
                        param_input.addItems(["True", "False"])
                        param_input.setCurrentText(str(value))
                        self.param_inputs[name] = param_input
                        form_layout.addRow(QLabel(name), param_input)

                    elif isinstance(value, (int, float, str, tuple)):  
                        # 数值类型 & 字符串 → 文本输入框
                        param_input = QLineEdit(str(value))
                        self.param_inputs[name] = param_input
                        form_layout.addRow(QLabel(name), param_input)

            layout.addLayout(form_layout)

            # 保存按钮
            self.confirm_button = QPushButton("保存")
            self.confirm_button.clicked.connect(self.save_changes)
            layout.addWidget(self.confirm_button)
            
        #  若无可配置参数，则显示提示信息
        else:
            self.no_param_label = QLabel("该层无可配置参数")
            layout.addWidget(self.no_param_label)

        self.setLayout(layout)

    # 更新 ModuleItem 的 nn 参数
    def save_changes(self):
        if self.module_item.nn:
            new_params = {}  # 新参数字典
            # 读取参数存入字典
            for key, input_field in self.param_inputs.items():
                try:
                    # 类型转换
                    if isinstance(input_field, QComboBox):  
                        # 布尔值转换
                        new_value = input_field.currentText() == "True"
                    else:
                        original_value = getattr(self.module_item.nn, key)  #  获取原数据
                        input_text = input_field.text()                     # 获取输入字符串

                        # 处理元组类型
                        if isinstance(original_value, tuple):
                            new_value = eval(input_text, {"__builtins__": None}, {})  # 将字符串转换为元组
                            if not isinstance(new_value, tuple):  # 确保转换结果仍是元组
                                raise ValueError
                        else:
                            # 其余按原类型转换
                            new_value = type(original_value)(input_text)

                    new_params[key] = new_value  # 存储新参数

                except ValueError:
                    QMessageBox.warning(self, "参数错误", f"参数 '{key}' 的输入无效，请检查格式！")
                    return
                
            # 根据新参数重新创建网络层
            layer_type = type(self.module_item.nn)  # 获取原始层类型
            self.module_item.nn = layer_type(**new_params)  # 重新初始化
            
        self.accept()