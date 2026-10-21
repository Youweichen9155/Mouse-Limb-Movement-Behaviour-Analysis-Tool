"""Select inclusive swing regions on the measured speed curve."""
import numpy as np
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QPainterPath
from PySide6.QtWidgets import (QWidget, QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QSpinBox,
    QComboBox, QDialogButtonBox, QMessageBox, QAbstractItemView)
from .phases import normalize_swing_regions


class SpeedCanvas(QWidget):
    selected = Signal(int, int)

    def __init__(self, work, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(250)
        self.setMouseTracking(True)
        self.frames = work.source_frame.to_numpy(int)
        self.regions = []
        self.values = np.zeros(len(work))
        self.unit = 'px/frame'
        self.anchor = self.cursor = None
        self.reset_view()

    def reset_view(self):
        self.left, self.right = float(self.frames[0]), float(self.frames[-1])
        self.update()

    def area(self):
        return QRectF(65, 35, max(1, self.width()-85), max(1, self.height()-78))

    def pixel(self, frame):
        r = self.area()
        return r.left() + (frame-self.left) / max(1, self.right-self.left) * r.width()

    def frame_at(self, x):
        r = self.area()
        value = self.left + (x-r.left()) / r.width() * (self.right-self.left)
        return int(np.clip(round(value), self.frames[0], self.frames[-1]))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor('white'))
        r = self.area()
        visible = (self.frames >= self.left) & (self.frames <= self.right)
        finite = self.values[visible & np.isfinite(self.values)]
        lo, hi = (min(0., float(finite.min())), max(0., float(finite.max()))) if len(finite) else (0., 1.)
        pad = max(1e-6, (hi-lo)*.08)
        lo, hi = lo-pad, hi+pad
        p.save(); p.setClipRect(r)
        for a, b in self.regions:
            p.fillRect(QRectF(self.pixel(a), r.top(), max(1, self.pixel(b+1)-self.pixel(a)), r.height()), QColor('#ffe4e4'))
        if self.anchor is not None:
            a, b = sorted((self.anchor, self.cursor))
            p.fillRect(QRectF(self.pixel(a), r.top(), max(1, self.pixel(b+1)-self.pixel(a)), r.height()), QColor('#ffc2c2'))
        path = QPainterPath(); drawing = False
        for f, value in zip(self.frames, self.values):
            if f < self.left-1 or f > self.right+1 or not np.isfinite(value):
                drawing = False
                continue
            point = QPointF(self.pixel(f), r.bottom() - (value-lo)/(hi-lo)*r.height())
            if drawing: path.lineTo(point)
            else: path.moveTo(point)
            drawing = True
        p.setPen(QPen(QColor('#222222'), 1.2)); p.drawPath(path)
        p.restore()
        p.setPen(QPen(QColor('#64748b'), 1)); p.drawRect(r)
        for i in range(5):
            value = lo+(hi-lo)*i/4
            y = r.bottom()-r.height()*i/4
            p.drawText(QRectF(0, y-10, 58, 20), Qt.AlignRight | Qt.AlignVCenter, f'{value:.2g}')
            frame = self.left+(self.right-self.left)*i/4
            p.drawText(QRectF(self.pixel(frame)-30, r.bottom()+3, 60, 20), Qt.AlignCenter, str(round(frame)))
        p.drawText(QRectF(r.left(), 5, r.width(), 25), Qt.AlignLeft, self.unit+' · 足端相对髋部')
        p.drawText(QRectF(r.left(), r.bottom()+22, r.width(), 20), Qt.AlignCenter, '源帧号（从 0 开始） · 滚轮缩放，拖动框选')

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.area().contains(event.position()):
            self.anchor = self.cursor = self.frame_at(event.position().x())
            self.update()

    def mouseMoveEvent(self, event):
        if self.anchor is not None:
            self.cursor = self.frame_at(event.position().x()); self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.anchor is not None:
            a, b = sorted((self.anchor, self.frame_at(event.position().x())))
            self.anchor = self.cursor = None
            self.selected.emit(a, b); self.update()

    def wheelEvent(self, event):
        if not self.area().contains(event.position()):
            return
        total = self.frames[-1]-self.frames[0]
        span = np.clip((self.right-self.left) * (.75 if event.angleDelta().y() > 0 else 1/.75), min(10, total), total)
        fraction = (event.position().x()-self.area().left())/self.area().width()
        centre = self.left+fraction*(self.right-self.left)
        self.left = float(np.clip(centre-fraction*span, self.frames[0], self.frames[-1]-span))
        self.right = self.left+span
        event.accept(); self.update()


class SwingRegionDialog(QDialog):
    def __init__(self, work, regions=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('按速率曲线划定摆动相')
        self.resize(1100, 700)
        self.work = work
        self.regions = normalize_swing_regions(regions or [], work.source_frame)
        self.refreshing = False
        layout = QVBoxLayout(self)
        note = QLabel('在曲线上按住左键拖出摆动区间，可重复添加。红色区域为摆动相，其余有效帧为支撑相。起止帧均包含在选区内。')
        note.setWordWrap(True); layout.addWidget(note)
        bar = QHBoxLayout(); layout.addLayout(bar)
        self.metric_combo = QComboBox(); self.metric_combo.addItem('速率曲线', 'foot_speed'); self.metric_combo.addItem('前向速度曲线', 'foot_forward_velocity')
        bar.addWidget(self.metric_combo)
        reset = QPushButton('查看全段'); bar.addWidget(reset); bar.addStretch()
        self.canvas = SpeedCanvas(work); layout.addWidget(self.canvas, 1)
        self.metric_combo.currentIndexChanged.connect(self.change_metric); self.change_metric()
        reset.clicked.connect(self.canvas.reset_view)
        self.canvas.selected.connect(self.add_region)
        bounds = QHBoxLayout(); layout.addLayout(bounds)
        self.begin, self.end = QSpinBox(), QSpinBox()
        for label, box in [('起始帧', self.begin), ('结束帧', self.end)]:
            box.setRange(int(work.source_frame.iloc[0]), int(work.source_frame.iloc[-1]))
            bounds.addWidget(QLabel(label)); bounds.addWidget(box)
        add = QPushButton('添加区间'); add.clicked.connect(lambda:self.add_region(self.begin.value(), self.end.value())); bounds.addWidget(add)
        delete = QPushButton('删除所选区间'); delete.clicked.connect(self.delete_selected); bounds.addWidget(delete)
        clear = QPushButton('清空选区'); clear.clicked.connect(self.clear_regions); bounds.addWidget(clear)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(['摆动起始帧（含）', '摆动结束帧（含）'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMaximumHeight(200)
        self.table.itemChanged.connect(self.table_changed)
        layout.addWidget(self.table)
        self.status = QLabel(); layout.addWidget(self.status)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText('保存并生成整体图')
        buttons.button(QDialogButtonBox.Cancel).setText('取消')
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        layout.addWidget(buttons); self.refresh()

    def change_metric(self):
        for suffix, unit in [('cm_s','cm/s'), ('px_s','px/s'), ('px_per_frame','px/frame')]:
            col = self.metric_combo.currentData()+'_'+suffix
            if col in self.work and np.isfinite(self.work[col]).any():
                self.canvas.values = self.work[col].to_numpy(float); self.canvas.unit = unit; break
        self.canvas.update()

    def refresh(self):
        self.refreshing = True
        self.table.setRowCount(len(self.regions))
        for row, pair in enumerate(self.regions):
            for col, value in enumerate(pair): self.table.setItem(row, col, QTableWidgetItem(str(value)))
        self.refreshing = False
        self.canvas.regions = self.regions; self.canvas.update()
        self.status.setText(f'已选 {len(self.regions)} 个摆动区间；保存后输出一幅整段姿态图。')

    def add_region(self, a, b):
        try:
            self.regions = normalize_swing_regions([*self.regions, [a,b]], self.work.source_frame)
            self.begin.setValue(a); self.end.setValue(b); self.refresh()
        except ValueError as e: QMessageBox.warning(self, '区间帧号', str(e))

    def table_changed(self, item):
        if self.refreshing: return
        try:
            pairs = [[self.table.item(r,c).text() for c in range(2)] for r in range(self.table.rowCount())]
            self.regions = normalize_swing_regions(pairs, self.work.source_frame)
        except ValueError as e: QMessageBox.warning(self, '区间帧号', str(e))
        self.refresh()

    def delete_selected(self):
        rows = {item.row() for item in self.table.selectedItems()}
        self.regions = [v for i,v in enumerate(self.regions) if i not in rows]; self.refresh()

    def clear_regions(self):
        self.regions = []; self.refresh()
