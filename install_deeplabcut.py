"""Install optional training dependencies in a separate user-owned environment."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import venv

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,default=Path.home()/'MouseLimbLab_DLC')
    parser.add_argument('--cuda-index',help='Optional official PyTorch CUDA wheel index, e.g. https://download.pytorch.org/whl/cu126')
    args=parser.parse_args()
    if not (3,10)<=sys.version_info[:2]<=(3,12):
        raise SystemExit('Please install Python 3.10–3.12 (3.12 recommended).')
    target=args.directory.expanduser().resolve()
    if target.exists() and not (target/'pyvenv.cfg').is_file():
        raise SystemExit('Destination is not a virtual environment; refusing to modify it.')
    if not target.exists():venv.EnvBuilder(with_pip=True).create(target)
    python=target/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if args.cuda_index:
        if not args.cuda_index.startswith('https://download.pytorch.org/whl/cu'):
            raise SystemExit('Use an official download.pytorch.org CUDA wheel index.')
        subprocess.run([str(python),'-m','pip','install','torch','torchvision','--index-url',args.cuda_index],check=True)
    subprocess.run([str(python),'-m','pip','install','deeplabcut==3.0.2','numpy==1.26.4','opencv-python-headless==4.11.0.86','socksio'],check=True)
    env=dict(os.environ,DLClight='True',MPLBACKEND='Agg',PYTHONUTF8='1')
    subprocess.run([str(python),'-c',"import deeplabcut, torch; print('DLC',deeplabcut.__version__,'Torch',torch.__version__,'CUDA',torch.cuda.is_available(),'MPS',torch.backends.mps.is_available())"],check=True,env=env)
    freeze=subprocess.check_output([str(python),'-m','pip','freeze'],text=True)
    (target/'installed-packages.txt').write_text(freeze,encoding='utf-8')
    print('\nSUCCESS. Select this Python in MouseLimbLab:\n'+str(python))

if __name__=='__main__':main()
