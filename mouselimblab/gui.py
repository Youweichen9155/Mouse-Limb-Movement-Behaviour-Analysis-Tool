from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import json
import sys
import threading
import traceback
import cv2
import numpy as np
import pandas as pd
from .qt_runtime import prepare
prepare()
from PySide6.QtCore import Qt, QThread, Signal, QUrl, QTimer
from PySide6.QtGui import QImage,QPixmap,QPainter,QColor,QPen,QDesktopServices,QFont,QFontDatabase
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,
    QPushButton,QLabel,QLineEdit,QComboBox,QDoubleSpinBox,QSpinBox,QCheckBox,QFileDialog,QMessageBox,
    QTabWidget,QSplitter,QScrollArea,QSlider,QTableWidget,QTableWidgetItem,QTextEdit,QHeaderView)
from .analysis import Settings
from . import __version__
from .workflow import run_clip,video_info,write_json,overlay_frame,summarize_reviewed
from .tracking import JOINTS
from .video import open_video

class Canvas(QWidget):
    clicked=Signal(float,float)
    def __init__(self):
        super().__init__();self.setMinimumSize(500,330);self.image=None;self.points={};self.ground=None;self.rect=None
    def set_frame(self,frame):
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.image=QImage(rgb.data,rgb.shape[1],rgb.shape[0],rgb.strides[0],QImage.Format_RGB888).copy();self.update()
    def paintEvent(self,event):
        p=QPainter(self);p.fillRect(self.rectangular(),QColor('#101827'))
        if self.image is None:
            p.setPen(QColor('#cbd5e1'));p.drawText(self.rectangular(),Qt.AlignCenter,'打开视频以选择标记点\n或直接导入已有轨迹 CSV');return
        size=self.image.size();size.scale(self.size(),Qt.KeepAspectRatio)
        from PySide6.QtCore import QRectF
        r=QRectF((self.width()-size.width())/2,(self.height()-size.height())/2,size.width(),size.height());self.rect=r
        p.drawImage(r,self.image)
        sx=r.width()/self.image.width();sy=r.height()/self.image.height()
        if self.ground is not None:
            p.setPen(QPen(QColor('#f8fafc'),1,Qt.DashLine));p.drawLine(r.left(),r.top()+self.ground*sy,r.right(),r.top()+self.ground*sy)
        for i,(j,(x,y)) in enumerate(self.points.items()):
            p.setPen(QPen(QColor(['#e879f9','#4ade80','#60a5fa','#facc15'][i%4]),2))
            xx,yy=r.left()+x*sx,r.top()+y*sy
            p.drawEllipse(xx-5,yy-5,10,10);p.drawText(xx+8,yy-8,j)
    def rectangular(self):return super().rect()
    def mousePressEvent(self,event):
        if self.image is not None and self.rect and self.rect.contains(event.position()):
            x=(event.position().x()-self.rect.left())*self.image.width()/self.rect.width()
            y=(event.position().y()-self.rect.top())*self.image.height()/self.rect.height()
            self.clicked.emit(x,y)

class Worker(QThread):
    progress=Signal(str);done=Signal(str);failed=Signal(str)
    def __init__(self,spec,root):
        super().__init__();self.spec=spec;self.root=root;self.cancelled=threading.Event()
        # OpenBLAS matrix inversion used by Matplotlib can exceed macOS's
        # default worker-thread stack. Match the 8 MiB main-thread budget.
        self.setStackSize(8*1024*1024)
    def run(self):
        try:self.done.emit(str(run_clip(self.spec,self.root,self.progress.emit,self.cancelled.is_set)))
        except Exception:self.failed.emit(traceback.format_exc())

class Window(QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle(f'MouseLimbLab {__version__} · 小鼠后肢运动分析');self.resize(1250,880)
        self.video='';self.coordinates='';self.seeds={};self.target=None;self.frame_index=0
        self.result=None;self.tracks=None;self.review=None;self.edits=[];self.worker=None
        self.tabs=QTabWidget();self.setCentralWidget(self.tabs)
        page=QWidget();layout=QHBoxLayout(page);self.tabs.addTab(page,'1  数据与分析')
        formwidget=QWidget();form=QFormLayout(formwidget);scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(formwidget);scroll.setMinimumWidth(370)
        layout.addWidget(scroll,0);right=QVBoxLayout();layout.addLayout(right,1)
        self.engine=QComboBox();self.engine.addItem('我们的代码 · 四点运动学','local');self.engine.addItem('ALMA · 官方六点运动学','alma');form.addRow('分析引擎',self.engine)
        self.engine.currentIndexChanged.connect(self.engine_changed)
        self.mode_note=QLabel();self.mode_note.setWordWrap(True);form.addRow(self.mode_note)
        self.protocol=QComboBox();form.addRow('实验方式',self.protocol)
        self.video_label=QLabel('未选择视频');self.video_label.setWordWrap(True)
        form.addRow(self.button('打开视频…',self.open_video));form.addRow(self.video_label)
        self.csv_label=QLabel('未选择 CSV：使用四色标记追踪');self.csv_label.setWordWrap(True)
        form.addRow(self.button('导入轨迹 CSV…',self.open_csv));form.addRow(self.csv_label)
        form.addRow(self.button('清除 CSV，使用视频追踪',self.clear_csv))
        self.side=QComboBox();self.side.addItem('无侧别后缀','unsided');self.side.addItem('左侧 L','l');self.side.addItem('右侧 R','r');form.addRow('DLC 身体侧别',self.side)
        self.phase_method=QComboBox();self.phase_method.addItem('足端速度换向','velocity');self.phase_method.addItem('位移峰谷（原方法）','position');form.addRow('四点相位识别',self.phase_method)
        self.conf=self.number(.9,0,1,2);form.addRow('DLC 置信度阈值',self.conf)
        self.video_fps=self.number(30,.01,2000,3);form.addRow('视频播放帧率',self.video_fps)
        self.capture_fps=self.number(0,0,2000,3);self.capture_fps.setSpecialValueText('未知（仅播放时间）');form.addRow('实际采集帧率',self.capture_fps)
        self.scale=self.number(0,0,100000,3);self.scale.setSpecialValueText('未标定');form.addRow('空间标尺 px/cm',self.scale)
        self.belt=self.number(0,0,1000,3);self.belt.setSpecialValueText('未填写');form.addRow('跑带速度 cm/s',self.belt)
        self.ground=self.number(0,-100000,100000,2);self.ground_set=QCheckBox('已设置地面参考线');form.addRow(self.ground_set);form.addRow('地面 Y（像素）',self.ground)
        form.addRow(self.button('在当前画面点击地面参考线',lambda:self.set_target('ground')))
        seeds=QWidget();seedrow=QHBoxLayout(seeds);seedrow.setContentsMargins(0,0,0,0)
        self.seed_buttons=[]
        for j,label in zip(JOINTS,['髋','膝','踝','足端']):
            b=self.button(label,lambda checked=False,j=j:self.select_seed(j));seedrow.addWidget(b);self.seed_buttons.append(b)
        form.addRow('首帧标记点',seeds)
        self.direction=QComboBox();self.direction.addItem('自动','auto');self.direction.addItem('正向 / 向右','positive');self.direction.addItem('负向 / 向左','negative');form.addRow('方向',self.direction)
        self.gap=QSpinBox();self.gap.setRange(0,10);self.gap.setValue(2);form.addRow('最多插值缺失帧',self.gap)
        self.cutoff=self.number(6,.01,1000,2);form.addRow('ALMA 截止频率 Hz',self.cutoff)
        self.drag=self.number(.1,0,10,3);form.addRow('ALMA 拖曳净空 cm',self.drag)
        self.drag_frames=QSpinBox();self.drag_frames.setRange(1,100);self.drag_frames.setValue(4);form.addRow('ALMA 连续拖曳帧',self.drag_frames)
        self.max_height=self.number(1.5,.01,100,2);form.addRow('ALMA 最大步高 cm',self.max_height)
        self.max_stride=self.number(8,.01,100,2);form.addRow('ALMA 最大步长 cm',self.max_stride)
        self.animal=QLineEdit();self.group=QLineEdit();self.trial=QLineEdit()
        for label,w in [('动物编号',self.animal),('组别',self.group),('试次编号',self.trial)]:form.addRow(label,w)
        self.out=QLineEdit(str(Path.home()/'Documents'/'MouseLimbLab_Results'));form.addRow('结果文件夹',self.out)
        form.addRow(self.button('选择结果文件夹…',self.choose_output))
        self.overlay=QCheckBox('导出叠加轨迹视频');form.addRow(self.overlay)
        form.addRow(self.button('保存项目配置…',self.save_project));form.addRow(self.button('打开项目配置…',self.load_project))
        self.run_button=self.button('开始分析',self.start)
        self.cancel_button=self.button('取消运行',self.cancel);self.cancel_button.setEnabled(False)
        self.canvas=Canvas();self.canvas.clicked.connect(self.click_point);right.addWidget(self.canvas,1)
        self.slider=QSlider(Qt.Horizontal);self.slider.setRange(0,0);self.slider.valueChanged.connect(self.show_frame);right.addWidget(self.slider)
        nav=QHBoxLayout();nav.addWidget(self.button('上一帧',lambda:self.slider.setValue(self.slider.value()-1)));nav.addWidget(self.button('下一帧',lambda:self.slider.setValue(self.slider.value()+1)))
        self.frame_label=QLabel('帧 0');nav.addWidget(self.frame_label);right.addLayout(nav)
        actions=QHBoxLayout();actions.addWidget(self.run_button);actions.addWidget(self.cancel_button);right.addLayout(actions)
        self.hint=QLabel('选择视频或坐标文件，再选择分析引擎。');self.hint.setWordWrap(True);right.addWidget(self.hint)
        self.log=QTextEdit();self.log.setReadOnly(True);self.log.setMaximumHeight(150);right.addWidget(self.log)
        reviewpage=QWidget();rv=QVBoxLayout(reviewpage);self.tabs.addTab(reviewpage,'2  周期与轨迹复核')
        toolbar=QHBoxLayout();rv.addLayout(toolbar)
        for label,fn in [('打开已有结果…',self.open_result),('保存周期复核',self.save_review),('打开结果文件夹',self.open_result_folder),('打开分析报告',self.open_report),('按动物汇总…',self.aggregate)]:toolbar.addWidget(self.button(label,fn))
        tip=QLabel('纳入：1＝纳入，0＝排除，空白＝未复核。双击周期编号查看原视频。四点方法可修改起始、摆动开始、结束帧，再点击“应用相位修正并重分析”。');tip.setWordWrap(True);rv.addWidget(tip)
        phasebar=QHBoxLayout();rv.addLayout(phasebar)
        phasebar.addWidget(self.button('查看速度与相位图',lambda:self.open_plot('phase_velocity.png')))
        phasebar.addWidget(self.button('查看连续姿态图',lambda:self.open_plot('limb_sequence.png')))
        self.phase_apply=self.button('应用相位修正并重分析',self.save_phases);phasebar.addWidget(self.phase_apply)
        self.table=QTableWidget();self.table.cellDoubleClicked.connect(self.jump_cycle);rv.addWidget(self.table)
        self.correction_joint=QComboBox();self.correction_joint.addItems(['hip','knee','ankle','foot','mtp','crest'])
        correction=QHBoxLayout();correction.addWidget(QLabel('校正当前帧关节：'));correction.addWidget(self.correction_joint)
        correction.addWidget(self.button('去画面点击校正点',lambda:self.set_correction_target()));correction.addWidget(self.button('保存校正点并准备重分析',self.save_corrections));rv.addLayout(correction)
        helptext=QTextEdit();helptext.setReadOnly(True);helptext.setHtml('''<h2>开始使用</h2>
<p><b>颜料追踪</b>：打开视频，选择“我们的代码”，在首帧点击髋、膝、踝、足端；跑台视频再设置地面参考线。填写录制参数后点击“开始分析”。</p>
<p><b>ALMA 六点分析</b>：导入髂嵴、髋、膝、踝、跖趾关节（MTP）、趾尖六点 CSV，选择跑台或自发行走。填写实际采集帧率；跑台填写跑带速度，自发行走填写 px/cm。</p>
<p><b>DeepLabCut</b>：第 3 页支持抽帧、六点标注、训练、评估和预测。安装随包提供的训练环境，用自己的视频标注训练模型，再将预测送入所选分析方法。</p>
<p><b>查看结果</b>：检查叠加轨迹，按需校正坐标并重分析。在“周期与轨迹复核”中填 1 纳入、0 排除，保存后按动物汇总。每次分析生成新的结果文件夹。</p>
<p><b>姿态与相位</b>：自动导出连续姿态图和足端速度曲线。红色为摆动相，黑色为支撑相，灰色为未分类。四点方法默认按足端相对髋部的速度换向识别，可在复核表修改边界后重分析。悬空运动显示前伸/后撤。</p>
<p>分析完成后窗口保持打开。点击“打开视频…”，选择新视频并重新标记，即可继续分析。</p>
<h3>录制参数</h3>
<p>使用固定机位、正立侧视、固定帧率的视频。播放帧率由视频读取，实际采集帧率按录制设置填写。例如 240 fps 慢动作以 30 fps 导出：播放填 30，采集填 240。</p>
<h3>方法与输出</h3>
<p>四点方法支持跑台和悬空，输出角度、ROM 和标准化曲线；ALMA 支持跑台和自发行走，另附官方参数表。两种方法按各自定义计算，结果按引擎和实验类型汇总。</p>
<p>完整说明见随包 docs/USER_GUIDE_ZH.md；训练说明见 docs/DEEPLABCUT_ZH.md。<a href="https://doi.org/10.1038/s42003-022-03077-6">ALMA 原文</a>与第三方许可见 README。</p>''');self.tabs.addTab(helptext,'3  使用说明')
        self.engine_changed()
        from .dlc_gui import DlcPanel
        self.dlc_panel=DlcPanel(Canvas)
        self.dlc_panel.prediction_ready.connect(self.use_dlc_prediction)
        self.tabs.insertTab(2,self.dlc_panel,'3  DeepLabCut 训练')
        self.tabs.setTabText(3,'4  使用说明')
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    def use_dlc_prediction(self,video,coordinates,engine,likelihood):
        try:
            from .io import load_coordinates
            tracks,_=load_coordinates(coordinates,side='unsided',likelihood=likelihood)
            self.set_video(video,reset=True)
            self.coordinates=coordinates;self.csv_label.setText(Path(coordinates).name);self.csv_label.setToolTip(coordinates)
            self.engine.setCurrentIndex(self.engine.findData(engine));self.side.setCurrentIndex(0);self.conf.setValue(likelihood)
            self.capture_fps.setValue(0);self.belt.setValue(0);self.scale.setValue(0)
            self.tracks=tracks.set_index('source_frame');self.show_frame(0)
            self.hint.setText('六点预测已载入。填写本视频的采集帧率、跑带速度或标尺后，点击“开始分析”。')
            self.tabs.setCurrentIndex(0)
        except Exception as e:self.error(e)
    def button(self,label,fn):
        b=QPushButton(label);b.clicked.connect(fn);return b
    def number(self,value,low,high,decimals):
        w=QDoubleSpinBox();w.setRange(low,high);w.setDecimals(decimals);w.setValue(value);return w
    def error(self,text):QMessageBox.warning(self,'需要检查',str(text))
    def engine_changed(self):
        alma=self.engine.currentData()=='alma';self.protocol.clear();self.protocol.addItem('跑台行走','treadmill')
        self.protocol.addItem('自发行走' if alma else '悬空后肢运动','spontaneous' if alma else 'suspended')
        self.mode_note.setText('导入六点 CSV；跑台填写速度，自发行走填写标尺。' if alma else '首帧选择四个颜料标记，或导入四点 CSV。')
        for b in self.seed_buttons:b.setEnabled(not alma)
        for w in (self.cutoff,self.drag,self.drag_frames,self.max_height,self.max_stride):w.setEnabled(alma)
        self.phase_method.setEnabled(not alma);self.phase_apply.setEnabled(not alma)
        self.ground.setEnabled(not alma);self.ground_set.setEnabled(not alma)
    def set_target(self,target):self.target=target;self.hint.setText('请在画面中点击：'+target)
    def select_seed(self,j):
        if not self.video:return self.error('先打开视频。')
        self.slider.setValue(0);self.show_frame(0);self.set_target(j)
    def click_point(self,x,y):
        if not self.target:return
        if self.target=='ground':self.ground.setValue(y);self.ground_set.setChecked(True);self.canvas.ground=y
        elif self.target.startswith('edit:'):
            self.edits.append({'source_frame':self.frame_index,'joint':self.target[5:],'x_px':x,'y_px':y})
        else:self.seeds[self.target]=[x,y];self.canvas.points=dict(self.seeds)
        self.hint.setText(f'已记录 {self.target}：({x:.1f}, {y:.1f})');self.target=None;self.canvas.update()
    def open_video(self):
        p,_=QFileDialog.getOpenFileName(self,'选择原视频','','视频 (*.mp4 *.mov *.avi *.mkv *.m4v);;所有文件 (*)')
        if p:
            try:self.set_video(p,reset=True)
            except Exception as e:self.error(e)
    def set_video(self,p,reset=False):
        info=video_info(p);self.video=p;self.video_label.setText(Path(p).name);self.video_label.setToolTip(p)
        self.video_fps.setValue(info['video_fps']);self.slider.setRange(0,max(0,info['frame_count']-1))
        if reset:
            self.seeds={};self.tracks=None;self.canvas.points={};self.canvas.ground=None;self.ground_set.setChecked(False);self.edits=[]
            self.correction_path='';self.result=None;self.review=None
            self.phase_events_path='';self.target=None;self.clear_csv();self.table.setRowCount(0)
        self.slider.setValue(0);self.show_frame(0)
    def open_csv(self):
        p,_=QFileDialog.getOpenFileName(self,'选择逐帧或 DLC CSV','','CSV (*.csv)')
        if p:
            self.coordinates=p;self.csv_label.setText(Path(p).name);self.csv_label.setToolTip(p)
            self.correction_path='';self.result=None;self.review=None;self.tracks=None;self.edits=[]
            self.phase_events_path=''
    def clear_csv(self):self.coordinates='';self.csv_label.setText('未选择 CSV：使用四色标记追踪')
    def show_frame(self,n):
        self.frame_index=n;self.frame_label.setText(f'源帧 {n}')
        if not self.video:return
        cap=open_video(self.video);cap.set(cv2.CAP_PROP_POS_FRAMES,n);ok,frame=cap.read();cap.release()
        if ok:
            if self.tracks is not None and n in self.tracks.index:frame=overlay_frame(frame,self.tracks.loc[n])
            self.canvas.points=dict(self.seeds) if n==0 else {};self.canvas.set_frame(frame)
    def choose_output(self):
        p=QFileDialog.getExistingDirectory(self,'选择结果文件夹',self.out.text())
        if p:self.out.setText(p)
    def spec(self):
        s=Settings(protocol=self.protocol.currentData(),video_fps=self.video_fps.value(),capture_fps=self.capture_fps.value() or None,
                   px_per_cm=self.scale.value() or None,belt_cm_s=self.belt.value() or None,ground_y_px=self.ground.value() if self.ground_set.isChecked() else None,
                   max_gap_frames=self.gap.value(),direction=self.direction.currentData(),cutoff_hz=self.cutoff.value(),
                   alma_drag_clearance_cm=self.drag.value(),drag_min_frames=self.drag_frames.value(),step_height_max_cm=self.max_height.value(),stride_length_max_cm=self.max_stride.value())
        s.phase_method=self.phase_method.currentData()
        spec={'engine':self.engine.currentData(),'video':self.video,'coordinates':self.coordinates,'seeds':self.seeds,'settings':asdict(s),
              'side':self.side.currentData(),'likelihood':self.conf.value(),'animal_id':self.animal.text().strip(),'group':self.group.text().strip(),
              'trial_id':self.trial.text().strip(),'overlay':self.overlay.isChecked()}
        if getattr(self,'correction_path',''):spec['corrections']=self.correction_path
        if getattr(self,'phase_events_path','') and spec['engine']=='local':spec['phase_events']=self.phase_events_path
        return spec
    def save_project(self):
        p,_=QFileDialog.getSaveFileName(self,'保存配置','limb_project.json','JSON (*.json)')
        if p:write_json(p,self.spec())
    def load_project(self):
        p,_=QFileDialog.getOpenFileName(self,'打开配置','','JSON (*.json)')
        if p:
            try:
                from .cli import load_project
                self.apply_spec(load_project(p))
            except Exception as e:self.error(e)
    def apply_spec(self,spec):
        self.engine.setCurrentIndex(self.engine.findData(spec.get('engine','local')))
        if spec.get('video'):self.set_video(spec['video'],reset=True)
        else:self.video='';self.video_label.setText('无视频');self.canvas.image=None;self.canvas.update();self.tracks=None
        self.coordinates=spec.get('coordinates','');self.csv_label.setText(Path(self.coordinates).name if self.coordinates else '未选择 CSV')
        self.seeds=spec.get('seeds',{});self.canvas.points=dict(self.seeds)
        s=Settings(**spec.get('settings',{}));self.protocol.setCurrentIndex(self.protocol.findData(s.protocol))
        for w,v in [(self.video_fps,s.video_fps),(self.capture_fps,s.capture_fps),(self.scale,s.px_per_cm),(self.belt,s.belt_cm_s),(self.ground,s.ground_y_px),(self.gap,s.max_gap_frames),(self.cutoff,s.cutoff_hz),(self.drag,s.alma_drag_clearance_cm),(self.drag_frames,s.drag_min_frames),(self.max_height,s.step_height_max_cm),(self.max_stride,s.stride_length_max_cm)]:w.setValue(v or 0)
        self.ground_set.setChecked(s.ground_y_px is not None);self.canvas.ground=s.ground_y_px
        self.direction.setCurrentIndex(self.direction.findData(s.direction));self.side.setCurrentIndex(self.side.findData(spec.get('side','unsided')))
        self.conf.setValue(spec.get('likelihood',.9));self.overlay.setChecked(spec.get('overlay',False))
        for w,key in [(self.animal,'animal_id'),(self.group,'group'),(self.trial,'trial_id')]:w.setText(spec.get(key,''))
        self.correction_path=spec.get('corrections','');self.edits=[];self.canvas.update()
        self.phase_events_path=spec.get('phase_events','')
        self.phase_method.setCurrentIndex(self.phase_method.findData(s.phase_method))
    def start(self):
        try:
            spec=self.spec();Settings(**spec['settings']).validate(spec['engine'])
            if spec['engine']=='alma' and not spec['coordinates']:raise ValueError('ALMA 模式请导入六点轨迹 CSV。')
            if not spec['coordinates'] and set(spec['seeds'])!=set(JOINTS):raise ValueError('请在视频首帧选择四个关节点，或导入轨迹 CSV。')
            self.run_button.setEnabled(False);self.cancel_button.setEnabled(True)
            self.worker=Worker(spec,self.out.text());self.worker.progress.connect(self.log.append)
            self.worker.done.connect(self.finished);self.worker.failed.connect(self.failed);self.worker.start()
        except Exception as e:self.error(e)
    def cancel(self):
        if self.worker:self.worker.cancelled.set();self.log.append('已请求取消，当前计算结束后停止。')
    def finished(self,folder):
        self.run_button.setEnabled(True);self.cancel_button.setEnabled(False);self.load_result(folder);self.log.append(f'分析完成：{folder}')
        self.hint.setText('分析完成，结果已保存。可查看结果，或点击“打开视频…”继续分析下一段。')
    def failed(self,text):
        self.run_button.setEnabled(True);self.cancel_button.setEnabled(False);self.log.append(text);self.error(text.splitlines()[-1])
    def open_result(self):
        p=QFileDialog.getExistingDirectory(self,'选择单次分析结果目录',self.out.text())
        if p:
            try:self.load_result(p)
            except Exception as e:self.error(e)
    def load_result(self,p):
        folder=Path(p);spec=json.loads((folder/'project.json').read_text(encoding='utf-8'));self.apply_spec(spec)
        self.result=folder;self.tracks=pd.read_csv(folder/'frame_tracks.csv').set_index('source_frame')
        self.review=pd.read_csv(folder/'cycle_review.csv').fillna('');self.table.clear();self.table.setRowCount(len(self.review));self.table.setColumnCount(len(self.review.columns))
        labels={'cycle_id':'周期','source_frame_start':'起始帧','source_frame_transition':'摆动开始帧',
                'source_frame_end':'结束帧','automatic_include':'自动纳入','manual_include':'人工纳入','review_note':'备注'}
        self.table.setHorizontalHeaderLabels([labels.get(c,c) for c in self.review.columns])
        for r in range(len(self.review)):
            for c,col in enumerate(self.review.columns):
                item=QTableWidgetItem(str(self.review.iloc[r,c]))
                editable=('manual_include','review_note')
                if spec.get('engine','local')=='local':editable+=('source_frame_start','source_frame_transition','source_frame_end')
                if col not in editable:item.setFlags(item.flags()&~Qt.ItemIsEditable)
                self.table.setItem(r,c,item)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents);self.show_frame(self.slider.value())
    def save_review(self):
        if self.result is None:return self.error('先打开一份结果。')
        df=self.review.copy()
        for r in range(len(df)):
            for col in ('source_frame_start','source_frame_transition','source_frame_end'):
                if self.table.item(r,df.columns.get_loc(col)).text().strip()!=str(df.iloc[r][col]):
                    return self.error('相位帧号已修改，请点击“应用相位修正并重分析”。')
        for r in range(len(df)):
            for c,col in enumerate(df.columns):
                if col in ('manual_include','review_note'):df.loc[r,col]=self.table.item(r,c).text().strip()
        if not df.manual_include.astype(str).isin(['','0','1','0.0','1.0']).all():return self.error('纳入列只接受 1、0 或空白。')
        if ((df.manual_include.astype(str).isin(['1','1.0']))&~df.automatic_include.eq(True)).any():return self.error('该周期已自动排除。修正坐标并重新分析后，再选择纳入。')
        df.to_csv(self.result/'cycle_review.csv',index=False);self.review=df;self.log.append('已保存周期复核。')
    def jump_cycle(self,r,c):
        if self.review is not None and self.review.columns[c]=='cycle_id':
            self.tabs.setCurrentIndex(0);self.slider.setValue(int(self.review.iloc[r]['source_frame_start']))
    def save_phases(self):
        if self.result is None or self.review is None or self.review.empty:return self.error('先打开含周期的分析结果。')
        if self.engine.currentData()!='local':return self.error('ALMA 使用官方相位事件。')
        try:
            import uuid
            from .phases import manual_events
            columns=['source_frame_start','source_frame_transition','source_frame_end']
            rows=[{col:self.table.item(r,self.review.columns.get_loc(col)).text() for col in columns} for r in range(len(self.review))]
            events=pd.DataFrame(rows)
            valid=np.isfinite(self.tracks[[f'{j}_{a}_px_smooth' for j in JOINTS for a in ('x','y')]]).all(axis=1).to_numpy()
            manual_events(events,self.tracks.index.to_numpy(),valid)
            path=self.result/('manual_phase_events_'+uuid.uuid4().hex[:8]+'.csv')
            events.to_csv(path,index=False)
            self.phase_events_path=str(path);self.coordinates=str(self.result/'raw_tracks.csv')
            self.csv_label.setText('已加载轨迹与相位修正');self.tabs.setCurrentIndex(0);self.start()
        except Exception as e:self.error(e)
    def open_plot(self,name):
        if self.result and (self.result/name).exists():QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.result/name)))
        else:self.error('分析完成后可查看此图；旧版结果请重新分析。')
    def set_correction_target(self):
        if not self.video or self.result is None:return self.error('先打开含视频的分析结果。')
        self.tabs.setCurrentIndex(0);self.set_target('edit:'+self.correction_joint.currentText())
    def save_corrections(self):
        if not self.edits or self.result is None:return self.error('尚未记录校正点。')
        path=self.result/'manual_point_corrections.csv'
        edited=pd.DataFrame(self.edits)
        if path.exists():edited=pd.concat([pd.read_csv(path),edited]).drop_duplicates(['source_frame','joint'],keep='last')
        edited.to_csv(path,index=False);self.correction_path=str(path);self.coordinates=str(self.result/'frame_tracks.csv')
        self.csv_label.setText('已加载本次轨迹及校正点');self.edits=[];self.tabs.setCurrentIndex(0);self.log.append('校正已保存。点击开始分析，生成新的独立结果目录。')
    def open_report(self):
        if self.result:QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.result/'report.html')))
    def open_result_folder(self):
        if self.result:QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.result)))
    def aggregate(self):
        files,_=QFileDialog.getOpenFileNames(self,'选择各试次结果中的 project.json',self.out.text(),'JSON (*.json)')
        if not files:return
        out=QFileDialog.getSaveFileName(self,'新汇总目录名称',str(Path(self.out.text())/'animal_summary'),'')[0]
        if out:
            try:summarize_reviewed([Path(p).parent for p in files],out);QDesktopServices.openUrl(QUrl.fromLocalFile(out))
            except Exception as e:self.error(e)
    def closeEvent(self,event):
        if self.dlc_panel.busy():
            self.dlc_panel.cancel();event.ignore();self.dlc_panel.status.setText('正在停止训练任务，请稍后再次关闭。')
        elif self.worker and self.worker.isRunning():
            self.cancel();event.ignore();self.hint.setText('正在停止运行，请稍后再次关闭。')
        else:event.accept()

def create_application(argv=None):
    app=QApplication.instance() or QApplication(argv or []);app.setStyle('Fusion')
    font_path=Path(__file__).parent/'assets'/'NotoSansSC[wght].ttf'
    font_id=QFontDatabase.addApplicationFont(str(font_path))
    families=QFontDatabase.applicationFontFamilies(font_id)
    if not families:raise RuntimeError('Bundled Chinese font could not be loaded.')
    app.setFont(QFont(families[0],10))
    app.setStyleSheet('QMainWindow{background:#f3f6fa} QPushButton{padding:7px 10px} QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox{min-height:26px} QTabBar::tab{padding:10px 18px} QLabel{color:#243045}')
    return app

def main():
    app=create_application(sys.argv)
    window=Window();window.show()
    return app.exec()
