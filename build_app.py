"""Build a native app on the target OS. Windows EXE is built on Windows."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import argparse
import stat

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--distpath',type=Path,default=root/'dist')
args=parser.parse_args()
distpath=args.distpath.resolve()
os.chdir(root)
os.environ['PATH']=os.environ.get('PATH','')+os.pathsep+os.pathsep.join(['/usr/bin','/bin','/usr/sbin','/sbin'])
os.environ.setdefault('PYINSTALLER_CONFIG_DIR',str(root/'.build_cache'))
cmd=[sys.executable,'-m','PyInstaller','--noconfirm','--clean','--windowed','--onedir',
     '--name','MouseLimbLab','--distpath',str(distpath),'--osx-bundle-identifier','org.msi1c.mouselimblab',
     '--collect-data','matplotlib','--add-data',str(root/'LICENSE')+os.pathsep+'.',
     '--hidden-import','matplotlib.backends.backend_pdf','--hidden-import','matplotlib.backends.backend_agg',
     '--add-data',str(root/'mouselimblab/dlc_worker.py')+os.pathsep+'mouselimblab',
     '--add-data',str(root/'mouselimblab/dlc_project.py')+os.pathsep+'mouselimblab',
     '--add-data',str(root/'mouselimblab/video.py')+os.pathsep+'mouselimblab',
     '--add-data',str(root/'LICENSES')+os.pathsep+'LICENSES',
     '--add-data',str(root/'mouselimblab/assets')+os.pathsep+'mouselimblab/assets',
     '--add-data',str(root/'THIRD_PARTY_NOTICES.md')+os.pathsep+'.','launch.py']
subprocess.run(cmd,check=True)
if sys.platform=='darwin':
    bundle=distpath/'MouseLimbLab.app'
    for file in [bundle,*bundle.rglob('*')]:
        if not file.is_symlink():os.chflags(file,file.stat().st_flags & ~stat.UF_HIDDEN)
    for attribute in ('com.apple.FinderInfo','com.apple.ResourceFork'):
        subprocess.run(['/usr/bin/xattr','-dr',attribute,str(bundle)],capture_output=True)
    subprocess.run(['/usr/bin/codesign','--force','--deep','--sign','-',str(bundle)],check=True)
