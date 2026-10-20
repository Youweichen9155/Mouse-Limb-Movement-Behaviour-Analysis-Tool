"""Build distributable native/source ZIPs from the checked-out public source."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mouselimblab import __version__

def source_archive(target):
    subprocess.run(['git','diff','--quiet','HEAD','--'],cwd=ROOT,check=True)
    subprocess.run(['git','archive','--format=zip','--prefix=MouseLimbLab_Source/',
                    '-o',str(target),'HEAD'],cwd=ROOT,check=True)

def collect_licenses(destination):
    dependencies={}
    for dist in metadata.distributions():
        name=dist.metadata.get('Name','unknown');dependencies[name]=dist.version
        for file in dist.files or []:
            # Copy license files, never arbitrary site-package or environment data.
            if any(word in str(file).lower() for word in ('license','copying','notice')):
                origin=Path(dist.locate_file(file))
                if not origin.is_file() or origin.suffix.lower() in ('.py','.pyc','.so','.dll','.pyd'):continue
                relative=Path(*[part for part in Path(file).parts if part not in ('..','.','/')])
                target=destination/name/relative
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(origin,target)
    return dependencies

def main():
    p=argparse.ArgumentParser();p.add_argument('--dist',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    smoke=json.loads(a.smoke.read_text(encoding='utf-8'))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert smoke['source_commit']==commit and smoke['native_gui']=='passed' and smoke['native_analysis']=='passed'
    assert smoke['version']==__version__
    label='Mac_arm64' if sys.platform=='darwin' else 'Windows_x64'
    expected=('arm64','aarch64') if sys.platform=='darwin' else ('amd64','x86_64')
    assert platform.machine().lower() in expected,platform.machine()
    source=out/f'MouseLimbLab_Source_{__version__}.zip';source_archive(source)
    with tempfile.TemporaryDirectory(prefix='mouselimblab_release_') as temporary:
        folder=Path(temporary)/f'MouseLimbLab_{label}';folder.mkdir()
        if sys.platform=='darwin':
            subprocess.run(['/usr/bin/ditto','--norsrc','--noextattr',str(a.dist.resolve()/'MouseLimbLab.app'),str(folder/'MouseLimbLab.app')],check=True)
        else:
            shutil.copytree(a.dist.resolve()/'MouseLimbLab',folder,dirs_exist_ok=True)
        for name in ('README.md','开始使用.txt','LICENSE','THIRD_PARTY_NOTICES.md','install_deeplabcut.py',
                     'Install_DeepLabCut_Mac.command','Install_DeepLabCut_Windows.bat'):
            shutil.copyfile(ROOT/name,folder/name)
        (folder/'Install_DeepLabCut_Mac.command').chmod(0o755)
        shutil.copytree(ROOT/'docs',folder/'docs',copy_function=shutil.copyfile)
        shutil.copytree(ROOT/'LICENSES',folder/'LICENSES',copy_function=shutil.copyfile)
        shutil.copyfile(source,folder/source.name)
        shutil.copyfile(a.smoke,folder/'SMOKE_TEST.json')
        dependencies=collect_licenses(folder/'LICENSES/dependencies')
        info={'version':__version__,'source_commit':commit,'platform':platform.platform(),
              'machine':platform.machine(),'python':platform.python_version(),'dependencies':dependencies,
              'source_url':'https://github.com/Youweichen9155/Mouse-Limb-Movement-Behaviour-Analysis-Tool',
              'native_smoke_test':smoke,'research_preview':True}
        (folder/'BUILD_INFO.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
        target=out/f'MouseLimbLab_{label}_{__version__}.zip'
        if sys.platform=='darwin':
            subprocess.run(['/usr/bin/codesign','--verify','--deep','--strict',str(folder/'MouseLimbLab.app')],check=True)
            subprocess.run(['/usr/bin/ditto','-c','-k','--keepParent','--norsrc',str(folder),str(target)],check=True)
        else:
            with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                for file in sorted(folder.rglob('*')):
                    if file.is_file():z.write(file,Path(folder.name)/file.relative_to(folder))
        with zipfile.ZipFile(target) as z:assert z.testzip() is None
        (out/f'BUILD_INFO_{label}.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
    sums=[]
    for file in sorted(out.glob('*.zip')):sums.append(hashlib.sha256(file.read_bytes()).hexdigest()+'  '+file.name)
    (out/f'SHA256SUMS_{label}.txt').write_text('\n'.join(sums)+'\n',encoding='utf-8')
    print(target)

if __name__=='__main__':main()
