"""Handle macOS synced folders that mark installed Qt plugins as hidden."""
from pathlib import Path
import os
import shutil
import sys
import tempfile

_plugin_directory = None

def prepare():
    global _plugin_directory
    if sys.platform != 'darwin':return
    from PySide6.QtCore import QLibraryInfo,QDir,QCoreApplication
    original=Path(QLibraryInfo.path(QLibraryInfo.PluginsPath))/'platforms'
    files=list(original.glob('*.dylib'))
    visible=QDir(str(original)).entryList(QDir.Files)
    if files and not any(f.name in visible for f in files):
        # Copy bytes only, deliberately excluding Finder flags/xattrs. This
        # changes no installed package, source data or OS security settings.
        _plugin_directory=tempfile.TemporaryDirectory(prefix='mouselimblab_qt_')
        base=Path(_plugin_directory.name);target=base/'platforms';target.mkdir()
        for f in files:shutil.copyfile(f,target/f.name)
        QCoreApplication.addLibraryPath(str(base))
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=str(target)
