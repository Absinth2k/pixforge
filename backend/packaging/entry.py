"""Entry point of the frozen backend (PyInstaller): ``pixforge-server.exe``."""

import sys

from pixforge.server.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
