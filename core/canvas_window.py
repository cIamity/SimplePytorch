from .config import *
from .canvas_item import ModuleItem, LineItem
from .config_window import ModuleConfigDialog
from .export import TorchModel

#region 画布视角设置
class GraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super(GraphicsView, self).__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)  # 抗锯齿
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)  # 缩放围绕鼠标指针进行
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)  # 视图完全更新
        self.is_panning = False  # 画布拖拽状态

        # 添加坐标标签
        self.coord_label = QLabel(self)
        self.coord_label.setStyleSheet(
                    "background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #8fcafe, stop:1 #1e90ff);" # 蓝色渐变背景
                    "color: black; padding: 2px;"      # 字体颜色
                    "font-family: 'Arial'; font-size: 13px;"        # 字体，字号
                    "border-radius: 3px;"  # 圆角半径
                )
        self.coord_label.setFixedSize(coord_label_width, coord_label_height)    # 标签大小

        # 为视图的视口安装事件过滤器
        self.viewport().installEventFilter(self)

    def eventFilter(self, source, event):
        if event.type() == QMouseEvent.MouseMove and source is self.viewport():  # 检查事件类型是否为鼠标移动事件，确保事件来源是当前视图的视口
            self.updateCoordLabel(event.pos())                                   # 更新坐标标签
        return super(GraphicsView, self).eventFilter(source, event)              # 调用父类的事件过滤器，确保其他事件处理逻辑不受影响

    def resizeEvent(self, event):
        super(GraphicsView, self).resizeEvent(event)     # 调用父类的重置事件
        self.coord_label.move(self.viewport().width() -  coord_label_width, self.viewport().height() - coord_label_height) # 标签移动到右下角
    
    def updateCoordLabel(self, pos):
        """更新坐标标签"""
        scene_pos = self.mapToScene(pos)  # 将鼠标的视口坐标转换为场景坐标
        self.coord_label.setText(f"x: {scene_pos.x():.1f} , y: {-scene_pos.y():.1f}")  # 更新坐标标签文本内容

    def wheelEvent(self, event: QWheelEvent):
        """重写滚轮事件"""
        if event.modifiers() == Qt.ControlModifier:
            # Ctrl + 滚轮 → 缩放画布
            current_scale = self.transform().m11()  # 获取当前缩放比例

            if event.angleDelta().y() > 0:       
            # 放大
                if current_scale < max_zoom:
                    self.scale(zoom_factor, zoom_factor)  
            
            else:
            # 缩小                
                if current_scale > min_zoom:
                    self.scale(1 / zoom_factor, 1 / zoom_factor)
        
        else:
            delta = event.angleDelta().y()  # 获取滚轮的 Y 轴移动量
            delta = int(delta / 230 * 100)  # 转换为标准滚动量

            if event.modifiers() == Qt.ShiftModifier:
                # Shift + 滚轮 → 水平滚动
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta)

            else:
                # 普通滚轮 → 垂直滚动
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta)

            self.updateCoordLabel(event.pos())  # 更新坐标标签

    def mousePressEvent(self, event):
        """按住鼠标中键拖动画布"""
        if event.button() == Qt.MiddleButton:
            self.is_panning = True
            self.setCursor(Qt.ClosedHandCursor)  # 改变鼠标指针形状
            self.start_pos = event.pos()  # 记录初始位置
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """鼠标中键拖拽画布"""
        if self.is_panning:
            delta = event.pos() - self.start_pos  # 计算移动距离
            self.start_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self.updateCoordLabel(event.pos())  # 更新坐标标签
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """释放鼠标中键时停止拖拽"""
        if event.button() == Qt.MiddleButton:
            self.is_panning = False
            self.setCursor(Qt.ArrowCursor)  # 还原鼠标指针
        else:
            super().mouseReleaseEvent(event)
#endregion


#region 画布场景设置
class GraphicsScene(QGraphicsScene):
    def __init__(self, parent=None):
        super(GraphicsScene, self).__init__(parent)
        self.setSceneRect(-canvas_size, -canvas_size, canvas_size*2, canvas_size*2)  # 画布区域
        self.G = nx.DiGraph()  # 初始化有向图

        # 鼠标操作逻辑变量
        self.selected_in_list = None  # 存储当前列表中选中的模块名称
        self.temp_item = None  # 临时预览的矩形
        self.select_item = None    # 选中的模块

        self.press_pos = None # 被选中后按下模块的位置
        self.pressed_item = None  # 被选中后按下的模块
        self.dragged_item = None    # 被拖动的模块

        self.close_dot = None  # 靠近的模块边缘点
        self.close_module = None # 靠近的模块
        self.start_module = None # 连线开始模块
        self.drawing_line = None # 正在进行的连线

        self.center_offset_x = module_width / 2 # 矩形中心水平偏移量
        self.center_offset_y = module_height / 2 # 矩形中心垂直偏移量

        self.long_press_timer = QTimer()  # 计时器
        self.long_press_timer.setSingleShot(True)  # 只触发一次
        self.long_press_timer.timeout.connect(self.onLongPress)  # 绑定超时事件

        # 初始化计数字典，根据计数为同名模块赋不同名字
        self.counter = {key: 0 for key in layer_config}


    #region 鼠标单击
    def mousePressEvent(self, event):
        #region 鼠标左键
        if event.button() == Qt.LeftButton:
            # 若已有预览模块，确定放置
            if self.temp_item:
                self.temp_item.setOpacity(1.0)  # 取消半透明
                self.G.add_node(self.temp_item.name, item=self.temp_item)       # 添加节点
                self.temp_item = None  # 取消预览状态

            # 若已在画线中
            elif self.drawing_line:
                # 计算坐标
                pos = event.scenePos()
                x = round(pos.x() / grid_size) * grid_size
                y = round(pos.y() / grid_size) * grid_size

                # 有邻近点，判断连线是否合法
                if self.close_dot:
                    # 获取头尾模块名
                    start_name = self.start_module.name
                    end_name= self.close_module.name

                    # 头尾模块相同
                    if start_name == end_name:
                        # 弹出提示窗口
                        msg_box = QMessageBox(self.views()[0])  # 设置父对象，确保窗口显示
                        msg_box.setIcon(QMessageBox.Warning)
                        msg_box.setWindowTitle("连线非法")
                        msg_box.setText("起点和终点为同一模块！")
                        msg_box.setStandardButtons(QMessageBox.Ok)  # 添加按钮
                        msg_box.exec_()  # 运行消息框

                    # 模块已有连接
                    elif self.G.has_edge(start_name, end_name):
                        # 弹出提示窗口
                        msg_box = QMessageBox(self.views()[0])  # 设置父对象，确保窗口显示
                        msg_box.setIcon(QMessageBox.Warning)
                        msg_box.setWindowTitle("连线非法")
                        msg_box.setText("起点和终点已存在连接！")
                        msg_box.setStandardButtons(QMessageBox.Ok)  # 添加按钮
                        msg_box.exec_()  # 运行消息框

                    # 以输入模块为终点 
                    elif self.close_module.layer_name == "Input":
                        # 弹出提示窗口
                        msg_box = QMessageBox(self.views()[0])  # 设置父对象，确保窗口显示
                        msg_box.setIcon(QMessageBox.Warning)
                        msg_box.setWindowTitle("连线非法")
                        msg_box.setText("不能以输入模块为终点！")
                        msg_box.setStandardButtons(QMessageBox.Ok)  # 添加按钮
                        msg_box.exec_()  # 运行消息框

                    elif self.close_module.layer_name == "Output" and self.G.in_degree(end_name) > 0:
                        # 弹出提示窗口
                        msg_box = QMessageBox(self.views()[0])  # 设置父对象，确保窗口显示
                        msg_box.setIcon(QMessageBox.Warning)
                        msg_box.setWindowTitle("连线非法")
                        msg_box.setText("输出模块只能接入一个输入！")
                        msg_box.setStandardButtons(QMessageBox.Ok)  # 添加按钮
                        msg_box.exec_()  # 运行消息框
                    

                    # 存在环路
                    elif nx.has_path(self.G, end_name, start_name):
                        # 弹出提示窗口
                        msg_box = QMessageBox(self.views()[0])  # 设置父对象，确保窗口显示
                        msg_box.setIcon(QMessageBox.Warning)
                        msg_box.setWindowTitle("环路警告")
                        msg_box.setText("存在环路，请注意连线！")
                        msg_box.setStandardButtons(QMessageBox.Ok)  # 添加按钮
                        msg_box.exec_()  # 运行消息框


                    # 合法
                    else:
                        # 将连线添加进图
                        self.G.add_edge(start_name, end_name, line = self.drawing_line)

                        self.drawing_line.start_name = start_name
                        self.drawing_line.end_name = end_name
                        self.drawing_line.done = True
                        self.drawing_line.update_path(update_arrow=True)
                        self.drawing_line = None
                        self.start_module = None
                        self.views()[0].setCursor(Qt.ArrowCursor) # 还原鼠标指针

                # 若点击空白处，建立锚点
                else:
                    self.drawing_line.add_anchor((x, y))

            # 若已选中物体
            elif self.select_item:  
                # 获取点击的项
                clicked_item = self.itemAt(event.scenePos(), QTransform()) 
                if isinstance(clicked_item, QGraphicsTextItem):
                    clicked_item = clicked_item.parentItem()
                
                # 若所选项是模块
                if isinstance(self.select_item, ModuleItem):
                    # 点击的是模块
                    if isinstance(clicked_item, ModuleItem):
                        # 点击的是已选中的模块
                        if clicked_item == self.select_item:
                            # 判断是否是长按
                            self.pressed_item = clicked_item
                            self.press_pos = event.scenePos() #记录按下位置
                            self.dragged_item = None  # 重置长按状态
                        
                            # 启动长按计时器（500ms）
                            self.long_press_timer.start(dragging_time)
                            # 若不是长按，在鼠标释放事件进行短按逻辑

                        # 点击其他模块，选中新模块
                        else:
                            self.select_item.setSelected(False)
                            self.select_item = clicked_item
                            self.select_item.setSelected(True)
                    
                    # 点击的是连线
                    elif isinstance(clicked_item, LineItem):
                        self.select_item.setSelected(False)
                        self.select_item = clicked_item
                        self.select_item.setSelected(True)
                
                    # 点击空白处，取消选中
                    else:
                        self.select_item.setSelected(False)
                        self.select_item = None
                
                # 若所选项是连线
                elif isinstance(self.select_item, LineItem):
                    # 点击的是连线
                    if isinstance(clicked_item, LineItem):
                        # 点击的是已选中的连线
                        if clicked_item == self.select_item:
                            pass

                        # 点击其他连线
                        else:
                            self.select_item.setSelected(False)
                            self.select_item = clicked_item
                            self.select_item.setSelected(True)
                    
                    # 点击的是模块
                    elif isinstance(clicked_item, ModuleItem):
                        self.select_item.setSelected(False)
                        self.select_item = clicked_item
                        self.select_item.setSelected(True)
                
                    # 点击空白处，取消选中
                    else:
                        self.select_item.setSelected(False)
                        self.select_item = None

            # 若有邻近点，开始连线
            elif self.close_dot:
                # 判断是否合法连线
                if self.close_module.layer_name == "Output":
                    # 弹出提示窗口
                    msg_box = QMessageBox(self.views()[0])  # 设置父对象，确保窗口显示
                    msg_box.setIcon(QMessageBox.Warning)
                    msg_box.setWindowTitle("连线非法")
                    msg_box.setText("不能以输出模块为起点")
                    msg_box.setStandardButtons(QMessageBox.Ok)  # 添加按钮
                    msg_box.exec_()  # 运行消息框
                
                else:
                    x = self.close_dot.sceneBoundingRect().center().x()
                    y = self.close_dot.sceneBoundingRect().center().y()
                    start_dot = (x, y)
                    self.views()[0].setCursor(Qt.BlankCursor)    #  鼠标指针消失
                    self.start_module = self.close_module
                    self.drawing_line = LineItem(start_dot)
                    self.addItem(self.drawing_line)
                    self.drawing_line.update_path()
        
            # 普通状态
            else:
                # 获取点击的项
                clicked_item = self.itemAt(event.scenePos(), QTransform()) 
                if isinstance(clicked_item, QGraphicsTextItem):
                    clicked_item = clicked_item.parentItem()

                # 若点击已存在的模块或连线，执行选中逻辑
                if isinstance(clicked_item, ModuleItem) or isinstance(clicked_item, LineItem):
                    self.select_item = clicked_item
                    self.select_item.setSelected(True)

                # 列表选中时点击空白处进入预览态
                elif self.selected_in_list:
                    pos = event.scenePos()
                    grid_x = round(pos.x() / grid_size) * grid_size
                    grid_y = round(pos.y() / grid_size) * grid_size

                    # 生成半透明预览矩形
                    self.temp_item = self.create_item(self.selected_in_list)
                    
                    # 矩形中心与鼠标位置对齐
                    self.temp_item.setPos(grid_x - self.center_offset_x, grid_y - self.center_offset_y)

                    self.temp_item.setOpacity(0.5)  # 设置半透明
                    self.addItem(self.temp_item)
        #endregion

        #region 鼠标右键
        elif event.button() == Qt.RightButton:
            # 若处于预览态
            if self.temp_item:
                self.removeItem(self.temp_item)
                self.temp_item = None

            # 如果处于画线态：
            elif self.drawing_line:
                # 若只有两个或一个锚点，则取消画线
                if len(self.drawing_line.anchors) < 3:
                    self.drawing_line.anchors.clear()
                    self.drawing_line.update_path()
                    self.removeItem(self.drawing_line)
                    self.drawing_line = None
                    self.start_module = None
                    self.views()[0].setCursor(Qt.ArrowCursor)  # 还原鼠标指针
                    
                # 删掉倒数第二个锚点
                else:
                    self.drawing_line.remove_second_last_anchor()

            # 普通状态
            else:
                # 获取点击的项
                clicked_item = self.itemAt(event.scenePos(), QTransform()) 
                if isinstance(clicked_item, QGraphicsTextItem):
                    clicked_item = clicked_item.parentItem()

                # 点击的是已存在的模块
                if isinstance(clicked_item, ModuleItem):
                    # 若存在已选择模块
                    if self.select_item:
                        # 若选中模块与点击模块不一致
                        if clicked_item != self.select_item:
                            # 变换选中模块
                            self.select_item.setSelected(False)
                            self.select_item = clicked_item
                            self.select_item.setSelected(True)

                    # 若不存在已选模块
                    else:
                        # 将点击模块加入选中
                        self.select_item = clicked_item
                        self.select_item.setSelected(True)

                    # 变换完选择模块后，进行打开列表操作（待完成）

                # 点击空白处，取消选中
                elif self.select_item:
                    self.select_item.setSelected(False)
                    self.select_item = None
        # endregion
    # endregion


    #region 鼠标双击
    def mouseDoubleClickEvent(self, event):
        # 获取点击的项
        clicked_item = self.itemAt(event.scenePos(), QTransform()) 
        if isinstance(clicked_item, QGraphicsTextItem):
            clicked_item = clicked_item.parentItem()
            
        # 若已有预览模块(即第一次点击空白处进入预览态)
        if self.temp_item:
            # 直接放置模块
            self.temp_item.setOpacity(1.0)  # 取消半透明
            self.G.add_node(self.temp_item.name, item=self.temp_item)       # 添加节点
            self.temp_item = None  # 取消预览状态

        # 双击不干扰画线操作
        if self.drawing_line or self.close_dot:
            pass

        # 若已选中模块(即第一次点击模块进入选中)
        elif self.select_item:
            # 点击的是模块
            if isinstance(clicked_item, ModuleItem):
                # 点击的是已选中的模块
                if clicked_item == self.select_item:
                    # 判断是否是长按
                    self.pressed_item = clicked_item
                    self.press_pos = event.scenePos() #记录按下位置
                    self.dragged_item = None  # 重置长按状态
                
                    # 启动长按计时器（500ms）
                    self.long_press_timer.start(dragging_time)
                    # 若不是长按，在鼠标释放事件进行短按逻辑

        # 普通态
        # 第一次单击放置模块 or 第一次单击取消选中
        elif self.selected_in_list:
            # 情况一：选中逻辑
            if isinstance(clicked_item, ModuleItem):
                self.select_item = clicked_item
                self.select_item.setSelected(True)

            # 情况二：放置逻辑
            else:
                pos = event.scenePos()
                grid_x = round(pos.x() / grid_size) * grid_size
                grid_y = round(pos.y() / grid_size) * grid_size

                # 生成矩形
                self.temp_item = self.create_item(self.selected_in_list)

                # 矩形中心与鼠标位置对齐
                self.temp_item.setPos(grid_x - self.center_offset_x, grid_y - self.center_offset_y)

                self.addItem(self.temp_item)
                self.G.add_node(self.temp_item.name, item=self.temp_item)       # 添加节点
                self.temp_item = None  # 取消预览状态
    # endregion


    #region 鼠标释放事件
    def mouseReleaseEvent(self, event):
        # 若按住了选中模块
        if self.pressed_item:
            #若检测为短按
            if self.long_press_timer.isActive(): 
                self.long_press_timer.stop()  # 停止长按检测计时器
                # 弹出模块设置窗口
                dialog = ModuleConfigDialog(self.pressed_item)
                dialog.exec_()  # 显示配置窗口

            # 若为长按松手，确定放置
            elif self.dragged_item:
                # 获取鼠标位置
                pos = event.scenePos()
                grid_x = round(pos.x() / grid_size) * grid_size
                grid_y = round(pos.y() / grid_size) * grid_size

                # 计算位移
                dx = grid_x - self.pressed_item.sceneBoundingRect().center().x()
                dy = grid_y - self.pressed_item.sceneBoundingRect().center().y()

                # 移动原模块到新位置
                self.pressed_item.setPos(grid_x - self.center_offset_x, grid_y - self.center_offset_y)
                #  更新模块相关的连线
                self.update_line(self.pressed_item, dx, dy)

                self.removeItem(self.dragged_item)  # 从画布中删除被拖动的模块
                self.select_item = self.pressed_item

                self.views()[0].setCursor(Qt.ArrowCursor)  # 还原鼠标指针

        self.pressed_item = None  # 取消长按目标
        self.press_pos = None
        self.dragged_item = None  # 重置拖动模块

        super(GraphicsScene, self).mouseReleaseEvent(event)
    #endregion


    #region 鼠标移动事件
    def mouseMoveEvent(self, event):
        # 计算鼠标位置
        pos = event.scenePos()
        grid_x = round(pos.x() / grid_size) * grid_size
        grid_y = round(pos.y() / grid_size) * grid_size

        # 预览态
        if self.temp_item:
            self.temp_item.setPos(grid_x - self.center_offset_x, grid_y - self.center_offset_y)

        # 拖拽态
        elif self.dragged_item:
            self.dragged_item.setPos(grid_x - self.center_offset_x, grid_y - self.center_offset_y)

        # 红点检测与连线
        else:
            # 若连线中,多出连线步骤
            if self.drawing_line:
                # 若鼠标与最终锚点同位置，不做操作
                if self.drawing_line.anchors[-1] == (grid_x, grid_y):
                    pass
        
                # 不同位置，则更新最后一个锚点，重写绘制路线
                elif self.drawing_line:
                    self.drawing_line.remove_last_anchor()
                    self.drawing_line.add_anchor((grid_x, grid_y))
                    self.drawing_line.update_path()

            object = self.get_close_module(grid_x, grid_y)
            # 若鼠标靠近某模块边缘
            if object is not None:

                # 若目前已有红点显示，判断是否同一红点
                if self.close_dot:
                    x = self.close_dot.sceneBoundingRect().center().x()
                    y = self.close_dot.sceneBoundingRect().center().y()

                    # 若是同一红点
                    if grid_x  == x and grid_y  == y:
                        pass
                    
                    # 若不是
                    else:
                        # 创建矩形红点
                        self.red_dot = QGraphicsRectItem(grid_x - red_dot_edge/2 , grid_y - red_dot_edge/2, red_dot_edge, red_dot_edge)
                        self.red_dot.setBrush(QBrush(Qt.red))  # 设置矩形为红色
                        self.red_dot.setOpacity(0.7)  # 设置半透明
                        self.addItem(self.red_dot)  # 将红点添加到场景
                        self.removeItem(self.close_dot)  # 从画布中删除已有红点指示
                        self.close_dot = self.red_dot # 添加新红点
                        self.close_module = object      
                
                # 若目前无红点
                else:
                    # 创建矩形红点
                    self.red_dot = QGraphicsRectItem(grid_x - red_dot_edge/2 , grid_y - red_dot_edge/2, red_dot_edge, red_dot_edge)
                    self.red_dot.setBrush(QBrush(Qt.red))  # 设置矩形为红色
                    self.red_dot.setOpacity(0.7)  # 设置半透明
                    self.addItem(self.red_dot)  # 将红点添加到场景
                    self.close_dot = self.red_dot # 添加新红点
                    self.close_module = object

            # 鼠标远离模块
            elif self.close_dot:
                self.removeItem(self.close_dot)  # 从画布中删除红点指示         
                self.close_dot = None
                self.close_module = None
    #endregion


    #region 键盘事件
    def keyPressEvent(self, event):
        # 如果当前有选中的物体
        if self.select_item:  
            # delete 删除选中物体
            if event.key() == Qt.Key_Delete:

                # 删除模块
                if isinstance(self.select_item, ModuleItem):
                    module_name = self.select_item.name  # 获取模块名称

                    # 获取所有相关的连线
                    edges = list(self.G.in_edges(module_name, data=True)) + list(self.G.out_edges(module_name, data=True))

                    # 删除所有相关的连线
                    for start, end, data in edges:
                        line_item = data['line']
                        self.removeItem(line_item.arrow_head)  # 移除箭头
                        self.removeItem(line_item)  # 移除连线
                        self.G.remove_edge(start, end)  # 从图中删除边

                    # 删除模块
                    self.G.remove_node(module_name)  # 从拓扑图中删除节点
                    self.removeItem(self.select_item)  # 从画布中删除模块
                    self.select_item = None  # 清空选中状态

                # 删除连线
                elif isinstance(self.select_item, LineItem):
                    self.removeItem(self.select_item.arrow_head)
                    self.removeItem(self.select_item)
                    self.G.remove_edge(self.select_item.start_name, self.select_item.end_name)  # 从图中删除边
                    self.select_item = None  # 清空选中状态
        else:
            super(GraphicsScene, self).keyPressEvent(event)
    #endregion


    #region 导出函数
    def get_model(self):
        model = TorchModel(self.G)
        if not model.node_topo or not model.input_nodes or not model.output_nodes:
            return None  # 无效模型
        else:
            return model
    #endregion


    #region 其他函数

    # 模块移动后的连线更新
    def update_line(self, item, dx, dy):
        item_name = item.name
        edges = list(self.G.in_edges(item_name, data=True)) + list(self.G.out_edges(item_name, data=True))

        for start, end, data in edges:
            line = data['line']  # 获取存储的 LineItem 对象

            if start == item_name:
                # 模块是起点，更新起点锚点
                x, y = line.anchors[0]
                line.anchors[0] = (x + dx, y + dy)

            if end == item_name:
                # 模块是终点，更新终点锚点
                x, y = line.anchors[-1]
                line.anchors[-1] = (x + dx, y + dy)

            # 更新连线与箭头
            line.update_path(update_arrow = True)


    # 获取某网格点在哪个模块边缘
    def get_close_module(self, x, y):
        for item in self.items():
            if isinstance(item, ModuleItem):
                pos = item.pos()
                left = pos.x()
                top = pos.y()
                right = left + module_width
                bottom = top + module_height

                if x == left or x == right:
                    if top < y < bottom:
                        return item

                if y == top or y == bottom:
                    if left < x < right:
                        return item
        return None
    

    # 槽函数，长按定时器用尽后触发长按逻辑
    def onLongPress(self):
        if self.pressed_item:
            pos = self.press_pos
            grid_x = round(pos.x() / grid_size) * grid_size
            grid_y = round(pos.y() / grid_size) * grid_size

            # 生成半透明预览矩形
            self.dragged_item = self.pressed_item.clone()

            # 矩形中心与鼠标位置对齐
            self.dragged_item.setPos(grid_x - self.center_offset_x, grid_y - self.center_offset_y)

            self.dragged_item.setOpacity(0.5)  # 设置半透明
            self.addItem(self.dragged_item)

            self.views()[0].setCursor(Qt.ClosedHandCursor)  # 改变鼠标指针形状        
            self.long_press_timer.stop()  # 防止短按逻辑触发


    # 创建模块函数
    def create_item(self, layer_name):
        count = self.counter[layer_name]
        self.counter[layer_name] += 1
        return ModuleItem(layer_name, count)


    # 设置当前选中的模块，由主窗口调用
    def set_selected_module(self, module_name):
        self.selected_in_list = module_name
        if self.temp_item:
            self.removeItem(self.temp_item)
            self.temp_item = None  # 取消已有的预览
    

    # 重写方法画布背景网格
    def drawBackground(self, painter: QPainter, rect: QRectF):
        scene_rect = self.sceneRect()
        left = int(scene_rect.left())
        right = int(scene_rect.right())
        top = int(scene_rect.top())
        bottom = int(scene_rect.bottom())

        n_size = 10 * grid_size
        # 绘制细网格
        painter.setPen(QPen(Qt.lightGray, 1))
        for x in range(left, right+1):
            if x % grid_size == 0:
                painter.drawLine(x, top, x, bottom)
        for y in range(top, bottom+1):
            if y % grid_size == 0:
                painter.drawLine(left, y, right, y)

        # 绘制粗网格
        painter.setPen(QPen(Qt.darkGray, 2))
        for x in range(left, right+1):
            if x % n_size == 0:
                painter.drawLine(x, top, x, bottom)
        for y in range(top, bottom+1):
            if y % n_size == 0:
                painter.drawLine(left, y, right, y)
    #endregion
#endregion