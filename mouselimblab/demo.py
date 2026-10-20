"""Synthetic data for software validation; not biological observations."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from .analysis import Settings
from .workflow import write_json

def demo_table(n=480):
    t=np.arange(n);phase=(t%60)/60
    x=np.where(phase<.65,35-70*phase/.65,-35+70*(phase-.65)/.35)
    lift=np.where(phase>=.65,25*np.sin(np.pi*(phase-.65)/.35),0)
    table=pd.DataFrame({'source_frame':t})
    points={'hip':(200+0*t,100+0*t),'knee':(220+8*np.sin(t/60*2*np.pi),150+0*t),
            'ankle':(190+.7*x,185-.4*lift),'foot':(200+x,220-lift),
            'mtp':(195+.9*x,207-.8*lift),'crest':(180+0*t,75+0*t)}
    for j,(px,py) in points.items():
        table[j+'_x_px']=px;table[j+'_y_px']=py;table[j+'_detected']=True
    return table

def make_demo(folder):
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=True)
    table=demo_table();table.to_csv(folder/'synthetic_tracks.csv',index=False)
    spec={'engine':'local','coordinates':str(folder/'synthetic_tracks.csv'),'settings':asdict(Settings(ground_y_px=220,capture_fps=30)),
          'animal_id':'SYNTHETIC_DEMO','group':'software_test','trial_id':'1'}
    write_json(folder/'demo_project.json',spec);return spec

def make_video_demo(folder):
    """Paint-marker video for exercising GUI tracking and overlay export."""
    import cv2
    from .tracking import JOINTS
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=True)
    table=demo_table(180)
    video=folder/'synthetic_markers.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),30,(400,280))
    assert writer.isOpened()
    colours=[(200,0,200),(0,220,0),(220,80,0),(0,160,255)]
    try:
        for _,row in table.iterrows():
            frame=np.full((280,400,3),30,np.uint8)
            for joint,colour in zip(JOINTS,colours):
                point=(round(row[joint+'_x_px']),round(row[joint+'_y_px']))
                cv2.circle(frame,point,6,colour,-1)
            writer.write(frame)
    finally:
        writer.release()
    seeds={j:[float(table.iloc[0][j+'_x_px']),float(table.iloc[0][j+'_y_px'])] for j in JOINTS}
    spec={'engine':'local','video':str(video),'seeds':seeds,'overlay':True,
          'settings':asdict(Settings(ground_y_px=220,capture_fps=30)),
          'animal_id':'SYNTHETIC_DEMO','group':'software_test','trial_id':'gui'}
    write_json(folder/'video_project.json',spec)
    return spec
