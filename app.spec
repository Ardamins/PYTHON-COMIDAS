# app.spec
# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['app.py'],
    pathex=['C:\\Proy4'],
    binaries=[],
    datas=[
        ('C:\\Proy4\\modelo_comidas_peruanas.tflite', '.'),
        ('C:\\Proy4\\comidas', 'comidas'),
    ],
    hiddenimports=[
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'PyQt5.sip',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.backends.backend_qt5',
        'PIL',
        'PIL.Image',
        'numpy',
        'fpdf',
        'tensorflow',
        'tensorflow.lite',
        'tensorflow.lite.python',
        'tensorflow.lite.python.interpreter',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ComidasPeruanas',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    onefile=True,
)