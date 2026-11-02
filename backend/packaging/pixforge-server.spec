# PyInstaller spec for the Pixforge backend (one folder).
# Build from backend/:  .venv/Scripts/pyinstaller packaging/pixforge-server.spec --noconfirm
# Output: backend/dist/pixforge-server/  (copied into the installer by electron-builder)

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

datas, binaries, hiddenimports = [], [], []
for pkg in ("onnxruntime", "rawpy", "pi_heif", "imagecodecs", "truststore"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
datas += collect_data_files("pixforge", includes=["**/*.json"])
hiddenimports += collect_submodules("pixforge.server")
hiddenimports += collect_submodules("uvicorn")
hiddenimports += ["pixforge.backends.onnx_backend"]

a = Analysis(
    ["entry.py"],
    pathex=["../src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=[
        # only needed to convert weights on the developer machine
        "torch", "torchvision", "onnx", "onnxscript", "gdown",
        "pixforge.models.export_onnx", "pixforge.models.build",
        "tkinter", "matplotlib", "IPython", "pytest", "PyQt5", "PySide6",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="pixforge-server",
    # A console build keeps stdin/stdout (READY handshake, exit when the app closes); the app
    # starts it with windowsHide, so no window appears. Logs: %LOCALAPPDATA%\Pixforge\logs
    console=True,
    icon="../../design/pixforge.ico",
    version=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="pixforge-server")
