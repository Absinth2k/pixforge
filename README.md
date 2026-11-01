# Pixforge

Lokaler KI-Upscaler für Fotos auf Basis von **HAT** (Hybrid Attention Transformer).
Bild hineinziehen, Modell wählen, fertig. Alles läuft auf dem eigenen Rechner, ohne Konto
und ohne Telemetrie.

> **Status:** in Entwicklung (Etappe 2 von 6: Bildpipeline und Kommandozeile).
> Einen Installer gibt es noch nicht. Screenshot folgt mit der Oberfläche in Etappe 4.

## Für Anwenderinnen und Anwender

Kommt mit dem ersten Release: Windows-Installer (ohne Adminrechte) und portable Version.
Pixforge sucht sich die beste Grafikkarte selbst (NVIDIA, AMD oder Intel über DirectML) und
rechnet ohne passende Grafikkarte auf dem Prozessor, dann allerdings deutlich langsamer.

## Für Entwickler

Voraussetzungen: Windows 10/11, [uv](https://docs.astral.sh/uv/), Git, Node.js (ab Etappe 4).

```powershell
cd backend
uv sync --extra export --extra raw   # Python 3.12 venv inkl. PyTorch (CPU) für den Export
uv run pytest
```

Hinter einem HTTPS-prüfenden Virenscanner (z. B. Avast) vorher `UV_SYSTEM_CERTS=1` setzen.

### Kommandozeile

```powershell
uv run pixforge foto.jpg -o out                      # 4x mit „Fotos (Standard)“
uv run pixforge C:\Bilder --width 6000 --faces       # ganzer Ordner, Gesichter verbessern
uv run pixforge scan.tif --model clean --scale 2 --format tiff --bits 16
uv run pixforge --list-models                        # installierte Modelle
uv run pixforge --list-devices                       # gefundene GPUs
```

Weitere Optionen: `--denoise auto|off|on|0-100`, `--jpeg`, `--sharpen`, `--detail`,
`--grain`, `--face-model face-gfpgan`, `--pattern "{name}_{scale}_{model}"`,
`--exists rename|overwrite|skip`, `--device cpu|gpu:0`, siehe `pixforge --help`.

### Modelle konvertieren

Alle Modelle aus `backend/src/pixforge/models/models.json` auf einmal erzeugen (landen in
`%LOCALAPPDATA%\Pixforge\models`, Hashes werden in `models.json` eingetragen):

```powershell
uv run python -m pixforge.models.build --weights ..\weights
```

Einzelne Modelle:

Die Original-Gewichte (siehe `THIRD_PARTY.md`) liegen nicht im Repository. Konvertierung nach
ONNX, als FP32 oder gemischtes FP16 (für GPUs):

```powershell
uv run python -m pixforge.models.export_onnx hat ..\weights\hat\Real_HAT_GAN_SRx4.pth out\real_hat_gan_x4_fp16.onnx --fp16
uv run python -m pixforge.models.export_onnx scunet ..\weights\denoise\scunet_color_real_psnr.pth out\scunet_fp16.onnx --fp16
```

Prüfen und messen: `bench/verify_hat_onnx.py`, `bench/verify_restore_onnx.py`,
`bench/bench_hat.py`, `bench/bench_denoise.py`. Ergebnisse und Entscheidungen stehen in
[`docs/backend.md`](docs/backend.md).

## Aufbau

```
backend/   Python (Bildverarbeitung, ONNX Runtime, später FastAPI-Server und CLI)
app/       Electron + React + TypeScript (ab Etappe 4)
bench/     Mess- und Prüfskripte
docs/      Messungen, Design
```

## Lizenz

Code: Apache-2.0. Modelle und Bibliotheken von Dritten: siehe [`THIRD_PARTY.md`](THIRD_PARTY.md).
