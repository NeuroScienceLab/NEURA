"""
Workflow Graph Component - Visual workflow diagram using QGraphicsView

Features:
- Vertical layout with connected nodes
- Data flow labels on edges
- Node state highlighting (pending, running, completed, error)
- Mouse wheel zoom + drag pan
- Click nodes to view details
- Dark/Light theme support
"""

from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsRectItem,
    QGraphicsPathItem, QGraphicsTextItem, QGraphicsEllipseItem,
    QGraphicsSimpleTextItem
)
from PySide6.QtCore import Qt, QPointF, Signal, QObject
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QPainterPath, QFont,
    QFontMetrics, QPolygonF
)


class WorkflowNode(QGraphicsRectItem):
    """
    Clickable workflow node with status indicator
    """

    def __init__(self, index, step_id, step_name, x, y, width=160, height=36, theme='dark'):
        super().__init__(x, y, width, height)
        self.index = index
        self.step_id = step_id
        self.step_name = step_name
        self.state = 'pending'
        self.details = {}
        self.theme = theme
        self.width = width
        self.height = height
        self._x = x
        self._y = y

        # Enable interaction
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFlag(QGraphicsRectItem.ItemIsSelectable)

        # Setup visual elements
        self._setup_style()
        self._add_content()

    def _setup_style(self):
        """Setup node visual style based on state"""
        self._update_colors()

    def _get_state_colors(self):
        """Get colors based on state and theme"""
        if self.theme == 'light':
            colors = {
                'pending': ('#FFFFFF', '#CCCCCC', '#333333'),
                'running': ('#E3F2FD', '#2196F3', '#1565C0'),
                'completed': ('#E8F5E9', '#4CAF50', '#2E7D32'),
                'error': ('#FFEBEE', '#F44336', '#C62828')
            }
        else:
            colors = {
                'pending': ('#3D3D3D', '#555555', '#CCCCCC'),
                'running': ('#1E3A5F', '#0A84FF', '#FFFFFF'),
                'completed': ('#1E3D2F', '#34C759', '#FFFFFF'),
                'error': ('#3D1E1E', '#FF3B30', '#FFFFFF')
            }
        return colors.get(self.state, colors['pending'])

    def _update_colors(self):
        """Update node colors based on current state"""
        bg, border, text = self._get_state_colors()
        self.setBrush(QBrush(QColor(bg)))
        self.setPen(QPen(QColor(border), 2))
        self._text_color = QColor(text)
        self._border_color = QColor(border)

    def _add_content(self):
        """Add node number and step name"""
        bg, border, text = self._get_state_colors()

        # Number circle on the left
        circle_size = 22
        circle_x = self._x + 8
        circle_y = self._y + (self.height - circle_size) / 2

        self.circle = QGraphicsEllipseItem(circle_x, circle_y, circle_size, circle_size, self)
        self.circle.setBrush(QBrush(QColor(border)))
        self.circle.setPen(QPen(Qt.NoPen))

        # Number text
        self.num_text = QGraphicsSimpleTextItem(str(self.index + 1), self.circle)
        self.num_text.setBrush(QBrush(QColor('#FFFFFF' if self.theme == 'dark' else '#FFFFFF')))
        font = QFont('Arial', 10, QFont.Bold)
        self.num_text.setFont(font)

        # Center number in circle
        text_rect = self.num_text.boundingRect()
        self.num_text.setPos(
            circle_x + (circle_size - text_rect.width()) / 2,
            circle_y + (circle_size - text_rect.height()) / 2
        )

        # Step name
        self.name_text = QGraphicsSimpleTextItem(self.step_name, self)
        self.name_text.setBrush(QBrush(QColor(text)))
        self.name_text.setFont(QFont('Microsoft YaHei', 10))

        # Position name text
        name_rect = self.name_text.boundingRect()
        self.name_text.setPos(
            self._x + 38,
            self._y + (self.height - name_rect.height()) / 2
        )

        # Status indicator on the right
        indicator_size = 8
        indicator_x = self._x + self.width - 16
        indicator_y = self._y + (self.height - indicator_size) / 2

        self.status_indicator = QGraphicsEllipseItem(
            indicator_x, indicator_y, indicator_size, indicator_size, self
        )
        self._update_status_indicator()

    def _update_status_indicator(self):
        """Update status indicator color"""
        status_colors = {
            'pending': '#666666',
            'running': '#0A84FF',
            'completed': '#34C759',
            'error': '#FF3B30'
        }
        color = status_colors.get(self.state, '#666666')
        self.status_indicator.setBrush(QBrush(QColor(color)))
        self.status_indicator.setPen(QPen(Qt.NoPen))

    def set_state(self, state):
        """Update node state and refresh visuals"""
        self.state = state
        self._update_colors()
        self._update_status_indicator()

        # Update text colors
        bg, border, text = self._get_state_colors()
        self.name_text.setBrush(QBrush(QColor(text)))
        self.circle.setBrush(QBrush(QColor(border)))

    def set_theme(self, theme):
        """Update theme and refresh visuals"""
        self.theme = theme
        self._update_colors()
        self._update_status_indicator()

        bg, border, text = self._get_state_colors()
        self.name_text.setBrush(QBrush(QColor(text)))
        self.circle.setBrush(QBrush(QColor(border)))

    def paint(self, painter, option, widget):
        """Custom paint for rounded rectangle"""
        painter.setRenderHint(QPainter.Antialiasing)

        bg, border, _ = self._get_state_colors()
        painter.setBrush(QBrush(QColor(bg)))
        painter.setPen(QPen(QColor(border), 2))

        # Draw rounded rectangle
        rect = self.rect()
        painter.drawRoundedRect(rect, 6, 6)

    def hoverEnterEvent(self, event):
        """Highlight on hover"""
        self.setOpacity(0.9)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Remove highlight"""
        self.setOpacity(1.0)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        """Handle click"""
        if event.button() == Qt.LeftButton:
            # Signal will be emitted via scene
            pass
        super().mousePressEvent(event)


class WorkflowEdge(QGraphicsPathItem):
    """
    Edge connecting two nodes with arrow and data flow label
    """

    def __init__(self, start_node, end_node, data_label="", theme='dark'):
        super().__init__()
        self.start_node = start_node
        self.end_node = end_node
        self.data_label = data_label
        self.theme = theme
        self.label_item = None
        self.arrow_item = None

        self._draw_edge()
        if data_label:
            self._add_label()

    def _get_edge_color(self):
        """Get edge color based on theme"""
        return '#888888' if self.theme == 'light' else '#666666'

    def _draw_edge(self):
        """Draw connecting line with arrow"""
        color = self._get_edge_color()

        # Get connection points
        start_rect = self.start_node.rect()
        end_rect = self.end_node.rect()

        start_x = start_rect.center().x()
        start_y = start_rect.bottom()
        end_x = end_rect.center().x()
        end_y = end_rect.top()

        start_point = QPointF(start_x, start_y)
        end_point = QPointF(end_x, end_y)

        # Draw bezier curve
        path = QPainterPath()
        path.moveTo(start_point)

        # Control points for smooth curve
        ctrl_offset = (end_y - start_y) / 3
        ctrl1 = QPointF(start_x, start_y + ctrl_offset)
        ctrl2 = QPointF(end_x, end_y - ctrl_offset)

        path.cubicTo(ctrl1, ctrl2, end_point)

        self.setPath(path)
        self.setPen(QPen(QColor(color), 2))

        # Add arrow
        self._add_arrow(end_point, color)

    def _add_arrow(self, point, color):
        """Add arrow head at end point"""
        arrow_size = 8

        # Triangle pointing down
        arrow_path = QPainterPath()
        arrow_path.moveTo(point.x(), point.y())
        arrow_path.lineTo(point.x() - arrow_size/2, point.y() - arrow_size)
        arrow_path.lineTo(point.x() + arrow_size/2, point.y() - arrow_size)
        arrow_path.closeSubpath()

        self.arrow_item = QGraphicsPathItem(arrow_path, self)
        self.arrow_item.setBrush(QBrush(QColor(color)))
        self.arrow_item.setPen(QPen(Qt.NoPen))

    def _add_label(self):
        """Add data flow label"""
        if not self.data_label:
            return

        color = '#666666' if self.theme == 'light' else '#888888'

        # Position label to the right of the line
        start_rect = self.start_node.rect()
        end_rect = self.end_node.rect()

        mid_y = (start_rect.bottom() + end_rect.top()) / 2
        label_x = start_rect.center().x() + 10

        self.label_item = QGraphicsSimpleTextItem(self.data_label, self)
        self.label_item.setBrush(QBrush(QColor(color)))
        self.label_item.setFont(QFont('Microsoft YaHei', 8))
        self.label_item.setPos(label_x, mid_y - 8)

    def set_theme(self, theme):
        """Update theme"""
        self.theme = theme
        color = self._get_edge_color()
        self.setPen(QPen(QColor(color), 2))

        if self.arrow_item:
            self.arrow_item.setBrush(QBrush(QColor(color)))

        if self.label_item:
            label_color = '#666666' if theme == 'light' else '#888888'
            self.label_item.setBrush(QBrush(QColor(label_color)))


class WorkflowGraphScene(QGraphicsScene):
    """
    Scene containing all workflow nodes and edges
    """

    def __init__(self, steps, data_flows, theme='dark'):
        super().__init__()
        self.steps = steps
        self.data_flows = data_flows
        self.theme = theme
        self.nodes = []
        self.edges = []

        self._build_graph()

    def _build_graph(self):
        """Build the workflow graph"""
        node_width = 160
        node_height = 36
        vertical_spacing = 50
        x_center = 100

        # Create nodes
        for i, (step_id, step_name) in enumerate(self.steps):
            x = x_center - node_width / 2
            y = i * (node_height + vertical_spacing)

            node = WorkflowNode(
                i, step_id, step_name,
                x, y, node_width, node_height,
                self.theme
            )
            self.addItem(node)
            self.nodes.append(node)

        # Create edges
        for i in range(len(self.nodes) - 1):
            data_label = self.data_flows.get(i, "")
            edge = WorkflowEdge(
                self.nodes[i], self.nodes[i + 1],
                data_label, self.theme
            )
            self.addItem(edge)
            self.edges.append(edge)

    def set_theme(self, theme):
        """Update theme for all items"""
        self.theme = theme
        for node in self.nodes:
            node.set_theme(theme)
        for edge in self.edges:
            edge.set_theme(theme)

    def mousePressEvent(self, event):
        """Handle mouse press to detect node clicks"""
        item = self.itemAt(event.scenePos(), self.views()[0].transform() if self.views() else None)

        # Find the node that was clicked
        while item:
            if isinstance(item, WorkflowNode):
                # Emit signal through view
                for view in self.views():
                    if hasattr(view, 'node_clicked'):
                        view.node_clicked.emit(item.index, item.details)
                break
            item = item.parentItem()

        super().mousePressEvent(event)


class WorkflowGraphView(QGraphicsView):
    """
    View for workflow graph with zoom and pan support
    """

    node_clicked = Signal(int, dict)

    def __init__(self, steps, data_flows, theme='dark'):
        super().__init__()

        # Create scene
        self._scene = WorkflowGraphScene(steps, data_flows, theme)
        self.setScene(self._scene)
        self.theme = theme

        # View settings
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Apply theme style
        self._apply_theme_style()

        # Initial zoom to fit
        self.scale(0.85, 0.85)

    def _apply_theme_style(self):
        """Apply style based on theme"""
        if self.theme == 'light':
            self.setStyleSheet("""
                QGraphicsView {
                    background-color: #F5F5F5;
                    border: none;
                    border-radius: 8px;
                }
            """)
        else:
            self.setStyleSheet("""
                QGraphicsView {
                    background-color: #1E1E1E;
                    border: none;
                    border-radius: 8px;
                }
            """)

    def wheelEvent(self, event):
        """Mouse wheel zoom"""
        factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(factor, factor)
        else:
            self.scale(1 / factor, 1 / factor)

    def update_node_state(self, index, state, details=None):
        """Update a node's state"""
        if 0 <= index < len(self._scene.nodes):
            self._scene.nodes[index].set_state(state)
            if details:
                self._scene.nodes[index].details = details

    def reset_all_states(self):
        """Reset all nodes to pending state"""
        for node in self._scene.nodes:
            node.set_state('pending')
            node.details = {}

    def set_theme(self, theme):
        """Update theme"""
        self.theme = theme
        self._apply_theme_style()
        self._scene.set_theme(theme)

    def fit_in_view(self):
        """Fit the entire graph in the view"""
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
