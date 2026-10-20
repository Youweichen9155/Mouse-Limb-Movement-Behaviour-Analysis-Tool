"""Six-point annotation and isolated DeepLabCut training workflow."""
from pathlib import Path
import json
import tempfile
import threading
import uuid
import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QProcess, QProcessEnvironment, QTimer, QUrl, QSettings
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QPushButton, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog,
    QMessageBox, QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit, QHeaderView)
from .dlc_project import (BODY_PARTS, BODY_LABELS, create_project, load_project,
    save_json, validate_annotations, default_python, worker_script, child_environment)
from .video import open_video

class Extractor(QThread):
    done = Signal(str)
    failed = Signal(str)
    progress = Signal(str)
    def __init__(self, options):
        super().__init__(); self.options = options; self.cancelled = threading.Event()
    def run(self):
        try:
            path = create_project(**self.options, progress=self.progress.emit, cancelled=self.cancelled.is_set)
            self.done.emit(str(path))
        except Exception as e:
            self.failed.emit(str(e))

class DlcPanel(QWidget):
    prediction_ready = Signal(str, str, str, float)
    def __init__(self, canvas_class):
        super().__init__()
        self.project = None; self.data = None; self.frame_index = 0; self.outputs = []
        self.process = None; self.extractor = None; self.stopping = False; self.buttons = []
        self.jobs_temp = tempfile.TemporaryDirectory(prefix='mouselimblab_dlc_')
        layout = QVBoxLayout(self)
        tip = QLabel('首次使用：安装训练环境 → 添加不同动物的视频 → 抽帧并标六点 → 训练和评估 → 预测 → 选择分析方法。')
        tip.setWordWrap(True); layout.addWidget(tip)
        envrow = QHBoxLayout(); layout.addLayout(envrow)
        envrow.addWidget(QLabel('DeepLabCut Python'))
        self.preferences=QSettings('MouseLimbLab','Training')
        self.python = QLineEdit(self.preferences.value('python',default_python())); self.python.setPlaceholderText('独立训练环境中的 python / python.exe')
        envrow.addWidget(self.python, 1)
        envrow.addWidget(self.button('选择…', self.pick_python)); envrow.addWidget(self.button('检查环境', lambda:self.start_action('probe')))
        envrow.addWidget(self.button('安装说明', self.install_help))
        projectrow = QHBoxLayout(); layout.addLayout(projectrow)
        self.project_label = QLabel('尚未打开训练项目'); projectrow.addWidget(self.project_label, 1)
        projectrow.addWidget(self.button('打开训练项目…', self.open_project))
        projectrow.addWidget(self.button('打开项目文件夹', self.open_folder))
        self.steps = QTabWidget(); layout.addWidget(self.steps, 1)
        videos = QWidget(); vl = QVBoxLayout(videos); self.steps.addTab(videos, '① 视频与抽帧')
        row = QHBoxLayout(); vl.addLayout(row)
        self.name = QLineEdit('MouseHindlimb'); self.scorer = QLineEdit('Researcher')
        row.addWidget(QLabel('项目名'));row.addWidget(self.name);row.addWidget(QLabel('标注者'));row.addWidget(self.scorer)
        row = QHBoxLayout(); vl.addLayout(row)
        row.addWidget(self.button('添加视频…', self.add_videos));row.addWidget(self.button('移除选中视频', self.remove_video))
        self.count = QSpinBox();self.count.setRange(2, 1000);self.count.setValue(20)
        row.addWidget(QLabel('每段均匀抽取帧数'));row.addWidget(self.count);row.addStretch()
        self.video_table = QTableWidget(0, 3);self.video_table.setHorizontalHeaderLabels(['视频路径', '动物编号', '用途'])
        self.video_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch);vl.addWidget(self.video_table)
        note = QLabel('同一只动物的所有视频必须放在同一组；测试组应包含未参加训练的动物。抽帧后可补选模糊、拖曳、遮挡等困难画面。原视频保持不变。')
        note.setWordWrap(True);vl.addWidget(note)
        row = QHBoxLayout();vl.addLayout(row)
        row.addWidget(self.button('创建新项目并抽帧…', self.extract))
        row.addWidget(self.button('保存动物编号与分组', self.save_video_roles))
        row.addWidget(self.button('追加当前视频的指定帧…', self.add_frame))
        labels = QWidget();ll = QVBoxLayout(labels);self.steps.addTab(labels, '② 六点标注')
        self.frame_select = QComboBox(); self.frame_select.currentIndexChanged.connect(self.show_label);ll.addWidget(self.frame_select)
        self.canvas = canvas_class();self.canvas.clicked.connect(self.place_point);ll.addWidget(self.canvas, 1)
        row = QHBoxLayout();ll.addLayout(row)
        self.part = QComboBox();self.part.addItems(BODY_LABELS);row.addWidget(QLabel('当前关节点'));row.addWidget(self.part)
        row.addWidget(self.button('该点不可见', self.mark_invisible));row.addWidget(self.button('清除该点', self.clear_point))
        self.point_state = QLabel();ll.addWidget(self.point_state)
        row = QHBoxLayout();ll.addLayout(row)
        row.addWidget(self.button('上一帧', lambda:self.frame_select.setCurrentIndex(max(0,self.frame_index-1))))
        row.addWidget(self.button('确认本帧完成并下一帧', self.complete_frame))
        row.addWidget(self.button('跳过，稍后标注', lambda:self.frame_select.setCurrentIndex(min(self.frame_select.count()-1,self.frame_index+1))))
        self.part.currentIndexChanged.connect(self.refresh_points)
        training = QWidget();tl=QVBoxLayout(training);self.steps.addTab(training,'③ 训练、评估与预测')
        form=QFormLayout();tl.addLayout(form)
        self.net=QComboBox();self.net.addItem('ResNet-50（先验证流程）','resnet_50');self.net.addItem('ResNet-101（较慢）','resnet_101')
        self.device=QComboBox()
        for label,value in [('自动检测','auto'),('CPU','cpu'),('Apple GPU','mps'),('NVIDIA GPU','cuda:0')]:self.device.addItem(label,value)
        self.epochs=QSpinBox();self.epochs.setRange(1,10000);self.epochs.setValue(200)
        self.batch=QSpinBox();self.batch.setRange(1,128);self.batch.setValue(4)
        self.threshold=QDoubleSpinBox();self.threshold.setRange(0,1);self.threshold.setDecimals(2);self.threshold.setValue(.9)
        for title,control in [('模型结构',self.net),('计算设备',self.device),('本次训练轮数（epochs）',self.epochs),('每批画面数',self.batch),('评估/分析置信度阈值',self.threshold)]:form.addRow(title,control)
        self.resume=QCheckBox('从最新快照继续训练（追加上述轮数）');form.addRow(self.resume)
        desc=QLabel('使用 DeepLabCut 3.0.2 / PyTorch。训练轮数与论文 TensorFlow 的迭代次数不等价。首次训练可能下载通用骨干网络权重；训练结束后必须检查独立动物的测试误差和叠加轨迹。')
        desc.setWordWrap(True);tl.addWidget(desc)
        row=QHBoxLayout();tl.addLayout(row)
        for title,action in [('准备训练数据','prepare'),('开始训练','train'),('评估模型','evaluate'),('预测视频…','predict')]:row.addWidget(self.button(title,lambda checked=False,a=action:self.start_action(a)))
        self.prediction_list=QComboBox();tl.addWidget(self.prediction_list)
        row=QHBoxLayout();tl.addLayout(row)
        row.addWidget(self.button('使用所选预测 → ALMA',lambda:self.use_prediction('alma')))
        row.addWidget(self.button('使用所选预测 → 我们的代码',lambda:self.use_prediction('local')))
        self.evaluation=QTextEdit();self.evaluation.setReadOnly(True);tl.addWidget(self.evaluation,1)
        self.status=QLabel('等待创建或打开项目。');self.status.setWordWrap(True);layout.addWidget(self.status)
        self.log=QTextEdit();self.log.setReadOnly(True);self.log.setMaximumHeight(140);layout.addWidget(self.log)
        self.stop=QPushButton('停止当前任务');self.stop.clicked.connect(self.cancel);self.stop.setEnabled(False);layout.addWidget(self.stop)

    def button(self, title, fn):
        button=QPushButton(title);button.clicked.connect(fn);self.buttons.append(button);return button
    def error(self, text):
        self.log.append(str(text));QMessageBox.warning(self,'DeepLabCut',str(text))
    def install_help(self):
        QMessageBox.information(self,'首次安装训练环境','双击随包提供的 Install_DeepLabCut_Mac.command 或 Install_DeepLabCut_Windows.bat。\n需先安装 Python 3.10–3.12，首次安装需要联网。\n安装后环境位于用户目录 MouseLimbLab_DLC，点击“选择”指定其中 bin/python（Mac）或 Scripts/python.exe（Windows）。\n已有可用环境可以直接选择；本机若检测到环境会自动填写。')
    def pick_python(self):
        path,_=QFileDialog.getOpenFileName(self,'选择 DeepLabCut 环境中的 Python')
        if path:self.python.setText(path);self.preferences.setValue('python',path)
    def open_folder(self):
        if self.project:QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.project.parent)))
    def add_videos(self):
        paths,_=QFileDialog.getOpenFileNames(self,'添加不同动物和场次的视频','','视频 (*.mp4 *.mov *.avi *.mkv *.m4v)')
        for path in paths:self.append_video({'path':path,'animal_id':'','split':'train'})
    def append_video(self, v):
        n=self.video_table.rowCount();self.video_table.insertRow(n)
        item=QTableWidgetItem(v['path']);item.setFlags(item.flags() & ~Qt.ItemIsEditable);self.video_table.setItem(n,0,item)
        self.video_table.setItem(n,1,QTableWidgetItem(v.get('animal_id','')))
        combo=QComboBox();combo.addItem('训练','train');combo.addItem('独立动物测试','test');combo.setCurrentIndex(combo.findData(v.get('split','train')));self.video_table.setCellWidget(n,2,combo)
    def video_rows(self):
        return [{'path':self.video_table.item(n,0).text(),'animal_id':self.video_table.item(n,1).text().strip(),
                 'split':self.video_table.cellWidget(n,2).currentData()} for n in range(self.video_table.rowCount())]
    def remove_video(self):
        if self.project:return self.error('已创建项目的视频保留用于追溯；需要移除时请创建新项目。')
        if self.video_table.currentRow()>=0:self.video_table.removeRow(self.video_table.currentRow())
    def extract(self):
        parent=QFileDialog.getExistingDirectory(self,'选择新训练项目保存位置')
        if not parent:return
        options={'parent':parent,'name':self.name.text().strip(),'scorer':self.scorer.text().strip(),
                 'videos':self.video_rows(),'frames_per_video':self.count.value()}
        self.set_busy(True);self.extractor=Extractor(options)
        self.extractor.progress.connect(self.log.append);self.extractor.done.connect(self.extracted);self.extractor.failed.connect(self.extraction_failed);self.extractor.start()
    def extracted(self, path):
        self.set_busy(False);self.load(path);self.steps.setCurrentIndex(1)
    def extraction_failed(self, message):
        self.set_busy(False);self.error(message)
    def save_video_roles(self):
        if not self.project:return self.error('请先创建或打开项目。')
        rows=self.video_rows()
        if set(v['path'] for v in rows)!=set(v['path'] for v in self.data['videos']):return self.error('已有项目仅能修改编号和用途；添加新视频请创建新项目。')
        bypath={v['path']:v for v in rows}
        for v in self.data['videos']:v.update(bypath[v['path']])
        self.save();self.status.setText('动物编号和分组已保存。')
    def open_project(self):
        path,_=QFileDialog.getOpenFileName(self,'打开训练项目','','训练项目 (dlc_project.json);;JSON (*.json)')
        if path:
            try:self.load(path)
            except Exception as e:self.error(e)
    def load(self, path):
        self.project=Path(path).resolve();self.data=load_project(self.project)
        self.project_label.setText(str(self.project));self.video_table.setRowCount(0)
        self.name.setText(self.data['name']);self.scorer.setText(self.data['scorer'])
        for v in self.data['videos']:self.append_video(v)
        self.reload_frames();self.outputs=[];self.prediction_list.clear()
        for manifest in sorted(self.project.parent.glob('predictions/*/prediction_manifest.json')):
            try:self.outputs.extend(json.loads(manifest.read_text(encoding='utf-8'))['outputs'])
            except (ValueError,KeyError):pass
        for out in self.outputs:self.prediction_list.addItem(Path(out['video']).name,out)
    def reload_frames(self):
        self.frame_select.blockSignals(True);self.frame_select.clear()
        for f in self.data['frames']:self.frame_select.addItem(f'{f["video_id"]} · 原始帧 {f["source_frame"]}')
        self.frame_select.blockSignals(False);self.show_label(0)
    def save(self):save_json(self.project,self.data)
    def show_label(self,n):
        if not self.data or not 0<=n<len(self.data['frames']):return
        self.frame_index=n;f=self.data['frames'][n]
        try:
            image=cv2.imdecode(np.frombuffer((self.project.parent/f['image']).read_bytes(),np.uint8),cv2.IMREAD_COLOR)
            if image is None:raise ValueError('图片无法解码。')
            self.canvas.set_frame(image)
            points=self.data['labels'].get(f['image'],{}).get('points',{})
            self.part.setCurrentIndex(next((i for i,p in enumerate(BODY_PARTS) if p not in points),0));self.refresh_points()
        except Exception as e:self.error(e)
    def current_label(self):
        if not self.data or not self.data['frames']:return None
        key=self.data['frames'][self.frame_index]['image']
        return self.data['labels'].setdefault(key,{'points':{},'complete':False})
    def place_point(self,x,y):
        label=self.current_label()
        if label is None:return
        label['points'][BODY_PARTS[self.part.currentIndex()]]=[x,y];label['complete']=False;self.save()
        self.part.setCurrentIndex(min(5,self.part.currentIndex()+1));self.refresh_points()
    def mark_invisible(self):
        label=self.current_label()
        if label is None:return
        label['points'][BODY_PARTS[self.part.currentIndex()]]=None;label['complete']=False;self.save()
        self.part.setCurrentIndex(min(5,self.part.currentIndex()+1));self.refresh_points()
    def clear_point(self):
        label=self.current_label()
        if label is None:return
        label['points'].pop(BODY_PARTS[self.part.currentIndex()],None);label['complete']=False;self.save();self.refresh_points()
    def refresh_points(self):
        label=self.current_label()
        if label is None:return
        points=label['points'];self.canvas.points={p:points[p] for p in BODY_PARTS if points.get(p) is not None};self.canvas.update()
        states=[f'{name}：'+('未标' if p not in points else ('不可见' if points[p] is None else '已标')) for p,name in zip(BODY_PARTS,BODY_LABELS)]
        self.point_state.setText('  /  '.join(states))
        done=sum(v.get('complete',False) for v in self.data['labels'].values())
        self.status.setText(f'已确认 {done} / {len(self.data["frames"])} 帧。点击画面标注 {self.part.currentText()}；每次点击自动保存。')
    def complete_frame(self):
        label=self.current_label()
        if label is None:return
        if set(label['points'])!=set(BODY_PARTS) or sum(p is not None for p in label['points'].values())<2:
            return self.error('请逐个标注六点，遮挡点选择“不可见”；至少需要两个可见点。')
        label['complete']=True;self.save();self.refresh_points();self.frame_select.setCurrentIndex(min(self.frame_select.count()-1,self.frame_index+1))
    def add_frame(self):
        if not self.project:return self.error('先打开训练项目，再选中视频表的一行。')
        row=self.video_table.currentRow()
        if row<0:return self.error('请先选中视频表的一行。')
        from PySide6.QtWidgets import QInputDialog
        video=self.data['videos'][row]
        n,ok=QInputDialog.getInt(self,'补充困难画面','原始零起始帧号',0,0,video['source_frames']-1)
        if not ok:return
        rel=f'labeled-data/{video["id"]}/img{n:08d}.png'
        if any(f['image']==rel for f in self.data['frames']):return self.error('该帧已经在标注列表中。')
        cap=open_video(video['path']);cap.set(cv2.CAP_PROP_POS_FRAMES,n);ok,frame=cap.read();cap.release()
        if not ok:return self.error('无法读取这帧。')
        ok,encoded=cv2.imencode('.png',frame)
        if not ok:return self.error('保存失败。')
        (self.project.parent/rel).write_bytes(encoded.tobytes())
        self.data['frames'].append({'image':rel,'video_id':video['id'],'source_frame':n,'width':frame.shape[1],'height':frame.shape[0]})
        self.save();self.reload_frames();self.frame_select.setCurrentIndex(len(self.data['frames'])-1);self.steps.setCurrentIndex(1)
    def set_busy(self,busy):
        for b in self.buttons:b.setEnabled(not busy)
        for w in (self.python,self.video_table,self.frame_select,self.canvas,self.part,self.net,self.device,self.epochs,self.batch,self.threshold,self.resume):w.setEnabled(not busy)
        self.stop.setEnabled(busy)
    def busy(self):
        return bool((self.process and self.process.state()!=QProcess.NotRunning) or (self.extractor and self.extractor.isRunning()))
    def start_action(self,action):
        if self.busy():return
        python=Path(self.python.text()).expanduser()
        if not python.is_file():return self.error('请先安装并选择 DeepLabCut Python。')
        if action!='probe' and not self.project:return self.error('请先创建或打开训练项目。')
        request={'action':action,'device':self.device.currentData(),'net_type':self.net.currentData(),
                 'epochs':self.epochs.value(),'batch_size':self.batch.value(),'pcutoff':self.threshold.value(),'resume':self.resume.isChecked()}
        if self.project:request['project']=str(self.project)
        if action=='prepare':
            try:validate_annotations(self.data)
            except Exception as e:return self.error(e)
        if action=='predict':
            videos,_=QFileDialog.getOpenFileNames(self,'选择需要预测的完整视频','','视频 (*.mp4 *.mov *.avi *.mkv *.m4v)')
            if not videos:return
            request['videos']=videos
        jobs=self.project.parent/'jobs' if self.project else Path(self.jobs_temp.name)
        folder=jobs/(action+'_'+uuid.uuid4().hex[:10]);folder.mkdir(parents=True)
        self.request_file=folder/'request.json';self.result_file=folder/'result.json';self.log_file=folder/'run.log'
        save_json(self.request_file,request);self.log_file.write_text('',encoding='utf-8')
        self.current_action=action;self.stopping=False
        self.process=QProcess(self);env=QProcessEnvironment()
        for k,v in child_environment().items():env.insert(k,v)
        env.insert('MPLCONFIGDIR',str(folder/'matplotlib'))
        self.process.setProcessEnvironment(env);self.process.setWorkingDirectory(str(folder))
        self.process.setProcessChannelMode(QProcess.MergedChannels);self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.process_finished);self.process.errorOccurred.connect(self.process_error)
        self.set_busy(True);self.status.setText(f'正在执行 {action}，日志会持续更新。')
        self.process.start(str(python),['-B','-u',str(worker_script()),str(self.request_file),str(self.result_file)])
    def read_output(self):
        text=bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace')
        with self.log_file.open('a',encoding='utf-8') as handle:handle.write(text)
        self.log.insertPlainText(text);self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
    def process_error(self,error):
        if error==QProcess.FailedToStart:self.set_busy(False);self.error('Python 无法启动，请检查训练环境路径。')
    def process_finished(self,code,status):
        self.read_output();self.set_busy(False)
        if self.stopping:self.status.setText('已停止。已有标注和训练快照保留；继续训练时勾选对应选项。');return
        if not self.result_file.exists():return self.error('任务未完成，请查看日志。')
        result=json.loads(self.result_file.read_text(encoding='utf-8'))
        if code or not result['ok']:return self.error(result.get('error','任务失败。'))
        info=result['result'];self.status.setText('完成：'+json.dumps(info,ensure_ascii=False))
        if self.project:self.data=load_project(self.project)
        if self.current_action=='evaluate':
            texts=[]
            for file in info.get('evaluation_files',[]):
                texts.append(str(file)+'\n'+Path(file).read_text(encoding='utf-8')[:15000])
            self.evaluation.setPlainText('\n\n'.join(texts) or '评估完成，详情见项目内 evaluation-results-pytorch 文件夹。')
        if self.current_action=='predict':
            for out in info['outputs']:self.outputs.append(out);self.prediction_list.addItem(Path(out['video']).name,out)
            self.prediction_list.setCurrentIndex(self.prediction_list.count()-len(info['outputs']))
            self.status.setText('预测完成。请使用所选预测进入运动学分析，并复核叠加轨迹。')
    def use_prediction(self,engine):
        out=self.prediction_list.currentData()
        if not out:return self.error('请先预测视频或打开含预测结果的项目。')
        self.prediction_ready.emit(out['video'],out['csv'],engine,self.threshold.value())
    def cancel(self):
        if self.extractor and self.extractor.isRunning():self.extractor.cancelled.set()
        if self.process and self.process.state()!=QProcess.NotRunning:
            self.stopping=True;self.process.terminate();process=self.process
            QTimer.singleShot(5000,lambda:process.kill() if process.state()!=QProcess.NotRunning else None)
