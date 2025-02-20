from .config import *

#region 模块矩形
class ModuleItem(QGraphicsRectItem):
    def __init__(self, layer_name, count, name=None):
        # 克隆时只需要执行的操作
        super().__init__(0, 0, module_width, module_height)

        # 模块名
        self.layer_name = layer_name

        # 若有指定实例名
        if name:
            self.name = name
        else:
            # 计数命名
            self.name = f"{layer_name}_{count}"

        # 文本
        # 输入输出模块显示实例名
        if self.layer_name == "Input" or self.layer_name == "Output":
            self.text = QGraphicsTextItem(self.name, self)
        # 其他模块显示模块名
        else:
            self.text = QGraphicsTextItem(self.layer_name, self)

        self.text.setDefaultTextColor(Qt.black)         # 字体颜色
        font = QFont("Arial", 16, QFont.Bold)           # 字体，字号，粗体
        self.text.setFont(font)

        # 文本居中
        text_rect = self.text.boundingRect()
        self.text.setPos((module_width - text_rect.width()) / 2, (module_height - text_rect.height()) / 2)

        # 设置描边
        self.default_pen = QPen(Qt.black, 2) # 默认黑色描边
        self.selected_pen = QPen(Qt.red, 3)  # 选中时红色描边
        self.setPen(self.default_pen)

        # 非克隆时要多执行的操作
        if count != -1:
            # 颜色
            gradient = QLinearGradient(0, 0, 0, module_height)  # 渐变
            gradient.setColorAt(0, QColor( *(layer_config[layer_name]["color_top"])    ))  # 顶部颜色
            gradient.setColorAt(1, QColor( *(layer_config[layer_name]["color_bottom"]) ))  # 底部颜色
            self.setBrush(QBrush(gradient))
            
            # 添加阴影效果
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(5)  # 阴影模糊半径
            shadow.setColor(QColor(0, 0, 0, 50))  # 阴影颜色
            shadow.setOffset(3, 3)  # 阴影偏移
            self.setGraphicsEffect(shadow)
            
            # 模块神经网络设置
            self.nn = None
            if "nn" in layer_config[layer_name]:
                # 获取对应的层类型
                layer_type = layer_config[layer_name]["nn"]
                # 获取默认参数
                default_params = layer_config[layer_name]["default"]
                # 使用参数初始化层
                self.nn = layer_type(*default_params)
    
    # 创建当前对象的拖拽副本
    def clone(self):
        new_item = ModuleItem(self.layer_name, -1, self.name)
        new_item.setBrush(self.brush())  # 复制外观
        new_item.setPen(self.pen())      # 复制边框
        return new_item

    # 重写 setSelected 方法，改变选中状态时的描边颜色
    def setSelected(self, selected):
        super().setSelected(selected)
        if selected:
            self.setPen(self.selected_pen)  # 选中时红色描边
        else:
            self.setPen(self.default_pen)  # 未选中时黑色描边

    # 重写 paint 方法，绘制圆角矩形
    def paint(self, painter, option, widget):
        painter.setBrush(self.brush())
        painter.setPen(self.pen())
        rect = self.rect()
        painter.drawRoundedRect(rect, corner_radius, corner_radius)  # 绘制圆角矩形
#endregion

#region 模块连线
class LineItem(QGraphicsPathItem):
    def __init__(self, start_dot):
        super().__init__()
        self.anchors = [start_dot]  # 存储所有锚点，起点作为第一个锚点
        self.anchors_mark = []  # 存储锚点标记实例
        self.arrow_head = None  # 存储终点箭头

        self.start_name = None # 起点模块名
        self.end_name = None # 终点模块名
        self.done = False       # 标识是否画完

        self.default_line_color = QPen(QColor(150, 0, 150), 4)  # 紫色，加粗
        self.select_line_color = QPen(Qt.red, 4)             # 红色
        self.default_arrow_color = QColor(150, 0, 150)        # 紫色
        self.select_arrow_color = QColor(Qt.red)              # 红色
        self.setPen(self.default_line_color)  

    # 添加锚点
    def add_anchor(self, pos):
        if pos not in self.anchors[:-1]:
            self.anchors.append(pos)
            self.update_path()

    # 根据锚点更新连线
    def update_path(self, update_arrow=False):
        if len(self.anchors) < 2:
            self.update_anchor_mark()  # 锚点刷新 "×"标识
            return  # 只有一个点，不用画线

        path = QPainterPath()
        path.moveTo(*self.anchors[0])  # 移动到起点

        for i in range(1, len(self.anchors)):
            prev_x, prev_y = self.anchors[i - 1]
            curr_x, curr_y = self.anchors[i]

            # **确保直角连线**
            if prev_x != curr_x and prev_y != curr_y:
                # 先水平，再垂直
                mid_x, mid_y = prev_x, curr_y
                path.lineTo(mid_x, mid_y)
                path.lineTo(curr_x, curr_y)
            else:
                # 直接连线
                path.lineTo(curr_x, curr_y)
        self.setPath(path)
        
        # 锚点刷新 "×"标识
        self.update_anchor_mark() 

        if update_arrow:
            self.draw_arrow()


    # 锚点更新"×"标识
    def update_anchor_mark(self):
        # 清除旧标记    
        for mark in self.anchors_mark:          # 删除斜线
            self.scene().removeItem(mark)  
        self.anchors_mark.clear()  # 清空列表

        if self.done is not True:
            # 绘制新标记
            size = anchor_label_size  # "×" 大小
            for pos in self.anchors:
                line1 = QGraphicsLineItem(pos[0] - size, pos[1] - size, pos[0] + size, pos[1] + size)
                line2 = QGraphicsLineItem(pos[0] - size, pos[1] + size, pos[0] + size, pos[1] - size)

                for line in (line1, line2):
                    line.setPen(QPen(QColor(0, 100, 0), 2))  # 墨绿色
                    self.scene().addItem(line)
                    self.anchors_mark.append(line)

    # 箭头绘制
    def draw_arrow(self):
        # 清除旧箭头
        if self.arrow_head:  
            self.scene().removeItem(self.arrow_head)  # 从场景中移除旧箭头
            self.arrow_head = None  # 清空引用，防止重复删除
            
        # 计算箭头方向
        p1 = QPointF(*self.anchors[-2])  # 倒数第二个点
        p2 = QPointF(*self.anchors[-1])  # 终点
        offset = grid_size * 0.35

        if p1.x() == p2.x():  
            # 垂直方向
            if p2.y() > p1.y():  # 向下
                top = QPointF(p2.x() , p2.y() + offset)
                left = QPointF(p2.x() - arrow_size/2, p2.y() - arrow_size + offset)
                right = QPointF(p2.x() + arrow_size/2, p2.y() - arrow_size + offset)
            else:  # 向上
                top = QPointF(p2.x() , p2.y() - offset)
                left = QPointF(p2.x() - arrow_size/2, p2.y() + arrow_size - offset)
                right = QPointF(p2.x() + arrow_size/2, p2.y() + arrow_size - offset)
        else:
            # 水平方向
            if p2.x() > p1.x():  # 向右
                top = QPointF(p2.x() + offset, p2.y())
                left = QPointF(p2.x() - arrow_size + offset, p2.y() - arrow_size/2)
                right = QPointF(p2.x() - arrow_size + offset, p2.y() + arrow_size/2)
            else:  # 向左
                top = QPointF(p2.x() - offset, p2.y())
                left = QPointF(p2.x() + arrow_size - offset, p2.y() - arrow_size/2)
                right = QPointF(p2.x() + arrow_size - offset, p2.y() + arrow_size/2)


        # **创建箭头三角形**
        arrow_head = QPolygonF([top, left, right])

        # 添加新箭头
        self.arrow_head = QGraphicsPathItem()
        arrow_path = QPainterPath()
        arrow_path.addPolygon(arrow_head)
        self.arrow_head.setPath(arrow_path)

        # 外观
        self.arrow_head.setBrush(self.default_arrow_color)  # 墨绿色填充
        self.scene().addItem(self.arrow_head)
        self.arrow_head.setPen(QPen(Qt.NoPen))  # 移除描边

    # 删除最后一个锚点与对应的 '×' 标识
    def remove_last_anchor(self):
        if len(self.anchors) > 1:
            # 移除最后一个锚点
            self.anchors.pop()  

            # 删除最后一个 "×" 标识
            if self.anchors_mark:
                for _ in range(2):
                    mark = self.anchors_mark.pop()
                    self.scene().removeItem(mark)  # 删除斜线

            self.update_path()  # 更新连线

    # 删除倒数第二个锚点及其 '×' 标识
    def remove_second_last_anchor(self):
        if len(self.anchors) > 2:  # 至少要有 3 个锚点，才能删除倒数第二个
            index = len(self.anchors) - 2  # 倒数第二个索引
            
            # 删除锚点
            del self.anchors[index]

            # 删除对应的 "×" 标识
            if self.anchors_mark:
                for _ in range(2):  # 删除两条斜线
                    mark = self.anchors_mark.pop(index * 2)
                    self.scene().removeItem(mark)  

            # 更新连线
            self.update_path()
    
    # 重写 setSelected 方法，改变选中状态时的描边颜色
    def setSelected(self, selected):
        super().setSelected(selected)
        if selected:
            # 选中时描边
            self.setPen(self.select_line_color)  
            self.arrow_head.setBrush(self.select_arrow_color)
        else:
            # 未选中时恢复默认描边
            self.setPen(self.default_line_color)  
            self.arrow_head.setBrush(self.default_arrow_color)

    # 重写 shape 方法，检测对象时返回路径而不是矩形包围盒
    def shape(self):
        path = self.path()  # 获取当前路径
        
        # 使用 QPainterPathStroker 来严格限制检测范围
        stroker = QPainterPathStroker()
        stroker.setWidth(self.pen().widthF() + 10)  # 设定检测区域 = 线宽 + 10 像素
        return stroker.createStroke(path)  # 返回严格限制的路径
#endregion