"""Publish only complete native packages from a successful, merged CI commit."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tomllib
import zipfile


def source_inventory(archive):
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        inventory={}
        for name in z.namelist():
            if name.endswith('/'):continue
            data=z.read(name)
            # Git for Windows can export text with CRLF. Permit that conversion
            # only for UTF-8 text; all binary files must remain byte-identical.
            if b'\0' not in data:
                try:data=data.decode('utf-8').replace('\r\n','\n').encode('utf-8')
                except UnicodeDecodeError:pass
            inventory[name]=hashlib.sha256(data).hexdigest()
        return inventory


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--run-id',required=True);p.add_argument('--commit',required=True)
    a=p.parse_args()
    assert a.run_id.isdigit() and re.fullmatch('[0-9a-f]{40}',a.commit)
    repo=os.environ['GITHUB_REPOSITORY']
    run=json.loads(subprocess.check_output(['gh','api',f'repos/{repo}/actions/runs/{a.run_id}']))
    assert run['conclusion']=='success' and run['head_sha']==a.commit
    assert run['path']=='.github/workflows/desktop.yml'
    subprocess.run(['git','merge-base','--is-ancestor',a.commit,'origin/main'],check=True)
    version=tomllib.loads(Path('pyproject.toml').read_text())['project']['version']
    assert re.fullmatch(r'\d+\.\d+\.\d+',version)
    out=Path('publish-assets');out.mkdir(exist_ok=False)
    def one(name):
        matches=list(a.artifacts.rglob(name));assert len(matches)==1,(name,matches)
        return matches[0]
    for label in ('Mac_arm64','Windows_x64'):
        meta=one(f'BUILD_INFO_{label}.json');info=json.loads(meta.read_text())
        assert info['version']==version and info['source_commit']==a.commit
        smoke=info['native_smoke_test']
        assert smoke['native_analysis']==smoke['native_gui']==smoke['chinese_font_glyphs']=='passed'
        assert smoke['source_commit']==a.commit
        archive=one(f'MouseLimbLab_{label}_{version}.zip')
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            embedded=json.loads(z.read(f'MouseLimbLab_{label}/BUILD_INFO.json'))
            assert embedded==info
            assert z.read(f'MouseLimbLab_{label}/LICENSES/NotoSansSC-OFL.txt')
        shutil.copyfile(archive,out/archive.name);shutil.copyfile(meta,out/meta.name)
    sources=list(a.artifacts.rglob(f'MouseLimbLab_Source_{version}.zip'))
    assert len(sources)==2
    canonical=out/sources[0].name
    subprocess.run(['git','archive','--format=zip','--prefix=MouseLimbLab_Source/',
                    '-o',str(canonical),a.commit],check=True)
    expected=source_inventory(canonical)
    for source in sources:
        actual=source_inventory(source)
        differences=[name for name in sorted(set(actual)|set(expected))
                     if actual.get(name)!=expected.get(name)]
        assert not differences, ('Source differs from tested commit',str(source),differences)
    sums=[hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.name for f in sorted(out.iterdir())]
    (out/'SHA256SUMS.txt').write_text('\n'.join(sums)+'\n',encoding='utf-8')
    notes=Path(f'docs/RELEASE_v{version}.md').read_text(encoding='utf-8')
    notes+=f'\nBuild source: `{a.commit}`. [Passed Mac / Windows checks](https://github.com/{repo}/actions/runs/{a.run_id}).\n'
    Path('release-notes.md').write_text(notes,encoding='utf-8')
    tag='v'+version
    subprocess.run(['gh','release','create',tag,*map(str,sorted(out.iterdir())),
                    '--repo',repo,'--target',a.commit,'--draft','--prerelease',
                    '--title',f'MouseLimbLab {tag} — Mac / Windows research preview',
                    '--notes-file','release-notes.md'],check=True)
    subprocess.run(['gh','release','edit',tag,'--repo',repo,'--draft=false'],check=True)


if __name__=='__main__':main()
