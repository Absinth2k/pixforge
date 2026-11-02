# Inferenz-Backend: Messungen und Entscheidungen

Stand: Etappe 1, 01.11.2026. Entwicklungsrechner: Windows 11 Pro, Intel Core Ultra 9 285K,
AMD Radeon RX 9070 XT (RDNA4, 16 GB, Treiber 32.0.32015 vom 18.10.2026),
ONNX Runtime 1.24.4 (DirectML), PyTorch 2.15 (CPU, nur für den Export).

Skripte: `bench/verify_hat_onnx.py`, `bench/verify_restore_onnx.py`, `bench/bench_hat.py`,
`bench/bench_denoise.py`. Rohdaten landen in `bench/out/` (nicht im Repo).

## 1. Entscheidungen auf einen Blick

| Thema | Entscheidung | Begründung |
|---|---|---|
| Laufzeit für Endnutzer | **ONNX Runtime + DirectML**, CPU als Rückfall | läuft auf NVIDIA/AMD/Intel ohne Zusatzinstallation |
| Präzision | **gemischtes FP16** auf der GPU, **FP32** auf der CPU | GPU 1,7× schneller bei 65–69 dB Abweichung; auf der CPU ist FP16 langsamer |
| Kachelgröße | **dynamisch**, eine ONNX-Datei pro Modell; Größe aus freiem VRAM (Formel unten) | Durchsatz ab 256 px praktisch konstant |
| Entrauschen | **SCUNet (real, PSNR)**, gemischtes FP16 | klar bestes Ergebnis; NAFNet verfälscht Farben und ist instabil |
| Entrauschen anwenden | nur oberhalb einer Rauschschwelle, als Mischung mit Detailschutz | bei wenig Rauschen glättet jedes Modell echte Struktur weg |
| JPEG-Artefakte | **FBCNN** exportiert und verifiziert; Bewertung mit echten JPEGs in Etappe 2 | eigene Aufgabe, an RAWs nicht sinnvoll messbar |
| Modell-Verteilung | vorkonvertierte ONNX-Dateien als **GitHub-Release** mit SHA256 | alle Lizenzen erlauben Weitergabe (Abschnitt 6) |

## 2. HAT nach ONNX

Die HAT-Architektur ist neu geschrieben (`models/hat_arch.py`), ohne `basicsr`/`einops`, und
liefert **bitgenau** dieselben Ausgaben wie der Originalcode (max. Abweichung 0,0).

Zwei Probleme beim Export und ihre Lösung:

1. **Aufmerksamkeitsmaske als Konstante.** Der naive Export backt die Maske der verschobenen
   Fenster 18× als 64-MB-Konstante ein, die Datei hat dann 1,3 GB. Jetzt ist die Maske ein
   **Modell-Eingang** (`attn_mask`), den Pixforge in NumPy berechnet (`models/hat_mask.py`).
   Nebeneffekt: Höhe und Breite sind **dynamisch** (Vielfache von 16), eine Datei mit 83 MB
   reicht für alle Kachelgrößen.
2. **FP16-Überlauf.** Der Residual-Strom in der letzten HAT-Gruppe erreicht ~1,8·10⁶, viele
   Kanal-Attention-Zwischenwerte liegen bei 30 000–51 000 (FP16-Grenze 65 504). Lösung:
   Kalibrierung mit Testmustern; jeder Knoten, der einen Tensor über 1/8 des FP16-Bereichs liest
   oder schreibt, bleibt FP32 (HAT: 162 von 10 857 Knoten). Zusätzlich entfernt der Export
   FP32→FP16→FP32-Cast-Paare, die der ONNX-Runtime-Konverter zwischen zwei FP32-Knoten einfügt
   und an denen der Wert sonst doch überläuft.

Parität gegen PyTorch (synthetisches Testbild, PSNR, Ziel > 50 dB):

| Modell | CPU | DirectML |
|---|---|---|
| Real_HAT_GAN_SRx4, FP32 | 131,5 dB | 131,0 dB |
| Real_HAT_GAN_SRx4, gemischtes FP16 | 76,4 dB | 67,1 dB |
| HAT-L_SRx4, gemischtes FP16 | 79,2 dB | 66,4 dB |

Dateigrößen: HAT 83 MB (FP32) / 47 MB (FP16), HAT-L 161 MB / 86 MB.

## 3. Geschwindigkeit und VRAM (HAT ×4)

Pro Konfiguration ein frischer Prozess. Median aus 3 Läufen nach dem Aufwärmen. „s/MP“ heißt
Sekunden pro Megapixel **Eingabe**, bereits mit 32 px Kachelüberlappung gerechnet. VRAM ist der
Spitzenwert des Prozesses (Windows-Leistungsindikator „GPU Process Memory“).

### RX 9070 XT, DirectML

| Modell | Präzision | Kachel | ms/Kachel | s/MP | VRAM |
|---|---|---:|---:|---:|---:|
| HAT (Real GAN) | FP32 | 256 | 874 | 17,4 | 4,6 GB |
| HAT (Real GAN) | FP32 | 384 | 1 902 | 15,3 | 9,1 GB |
| HAT (Real GAN) | FP32 | 512 | 37 338 | 162 | 12,8 GB ⚠ |
| HAT (Real GAN) | **FP16 gemischt** | 192 | 276 | 10,8 | 1,4 GB |
| HAT (Real GAN) | **FP16 gemischt** | 256 | 501 | **10,0** | 2,6 GB |
| HAT (Real GAN) | **FP16 gemischt** | 384 | 1 238 | **10,0** | 5,3 GB |
| HAT (Real GAN) | **FP16 gemischt** | 512 | 2 415 | 10,5 | 9,9 GB |
| HAT-L | FP32 | 384 | 3 699 | 29,9 | 9,3 GB |
| HAT-L | FP32 | 512 | 73 354 | 318 | 13,0 GB ⚠ |
| HAT-L | **FP16 gemischt** | 256 | 964 | **19,2** | 2,7 GB |
| HAT-L | **FP16 gemischt** | 384 | 2 426 | 19,6 | 5,4 GB |

⚠ Ab ~12,8 GB lagert der Treiber in den geteilten Systemspeicher aus, die Laufzeit steigt um das
20-Fache. Die automatische Kachelwahl muss davor klar Abstand halten.

Erster Lauf pro Kachelgröße: +0,5–1,5 s (Shader-Kompilierung). Deshalb bleibt die Kachelgröße
innerhalb eines Bildes konstant, Randkacheln werden aufgefüllt.

**Was das praktisch heißt** (HAT, FP16, 4×): 1 MP → 16 MP in ~10 s; 12-MP-Handyfoto → 48 MP in
~2 Minuten; HAT-L doppelt so lange.

### Intel Core Ultra 9 285K, CPU

| Modell | Präzision | Kachel | ms/Kachel | s/MP |
|---|---|---:|---:|---:|
| HAT (Real GAN) | FP32 | 128 | 2 729 | 296 |
| HAT (Real GAN) | FP32 | 256 | 15 618 | 311 |
| HAT (Real GAN) | FP16 gemischt | 256 | 18 036 | 360 |

Die CPU ist ~30× langsamer als die GPU: Ein 12-MP-Foto braucht rund **eine Stunde**. Die
Oberfläche muss das vorher ehrlich anzeigen und für CPU-Nutzer eine kleinere Zielgröße oder
2× vorschlagen. Auf der CPU wird das FP32-Modell geladen.

### Kachelgröße aus dem VRAM

Der VRAM-Bedarf wächst linear mit der Kachelfläche:
FP16 ≈ 0,5 GB + 38 KB × Pixel, FP32 ≈ 0,9 GB + 56 KB × Pixel (konservativ, HAT und HAT-L).
Regel: größte Kachel aus {128, 192, 256, 384}, deren Schätzung höchstens **50 % des
freien VRAM** belegt.
Beispiele: 16 GB → 384; 8 GB → 256; 4 GB → 192; 2 GB → 128. Bei einem
Speicherfehler halbiert Pixforge die Kachel und rechnet die Kachel neu.

### Geräteerkennung

ONNX Runtime 1.24 meldet über `get_ep_devices()` Name, VRAM, „diskret“ und den DXGI-Index jeder
DirectML-GPU. Pixforge sortiert: diskrete GPU vor integrierter, dann nach VRAM. Auf diesem
Rechner erscheint nur die RX 9070 XT. Die Intel-iGPU taucht nicht als DirectML-Gerät auf
(wahrscheinlich wegen ihres Treibers von 11/2024); wäre sie gelistet, käme sie hinter der
Radeon.

## 4. Entrauschen vor dem Upscale

### Kandidaten

| Modell | Herkunft | Lizenz | ONNX FP32 / FP16 | Parität DirectML (FP16) |
|---|---|---|---|---|
| SCUNet color real (PSNR) | cszn/SCUNet | Apache-2.0 | 72 / 38 MB | 68,7 dB |
| NAFNet SIDD width64 | megvii-research/NAFNet | MIT | 443 / 232 MB | 64,7 dB |
| FBCNN color | jiaxi-jiang/FBCNN | Apache-2.0 | 274 / 137 MB | 69,2 dB |
| OpenCV fastNlMeans | OpenCV | Apache-2.0 | – | – |

Alle drei Netze sind neu geschrieben und liefern bitgenau die Originalausgaben. Für DirectML
musste SCUNet angepasst werden: 5-dimensionale MatMuls lehnt DirectML ab, die Köpfe werden in
die Batch-Dimension gefaltet.

### Testmaterial

5 Canon-CR3-Aufnahmen von Andre (nicht im Repo), entwickelt mit LibRaw (rawpy) ohne eigene
Rauschminderung. Zwei davon sind stark verrauscht (Luma-σ ≈ 5 in 8-Bit-Einheiten), drei
leicht (σ ≈ 1–1,8).

**A. Echte Fotos** (je 3 Ausschnitte 768², kein rauschfreies Original vorhanden): Restrauschen
und Detail-Erhalt (Laplace-Energie im Verhältnis zum Original, auf halber Auflösung).
**B. Synthetisch**: Foto 4× verkleinert (nahezu rauschfrei), dann signalabhängiges Rauschen
(Schrot + Ausleserauschen in linearem Licht) in drei Stufen, PSNR/SSIM gegen das saubere Bild.

### Ergebnisse B: synthetisch (PSNR dB / SSIM, Mittel über 4 Fotos)

| Stufe | verrauscht | nlmeans | **SCUNet** | NAFNet |
|---|---|---|---|---|
| leicht | 31,6 / 0,80 | 29,9 / 0,79 | **33,0 / 0,88** | 28,3 / 0,80 |
| mittel | 26,7 / 0,61 | 26,8 / 0,64 | **32,4 / 0,87** | 26,3 / 0,72 |
| stark | 22,6 / 0,42 | 24,9 / 0,53 | **31,6 / 0,87** | 18,6 / 0,44 ⚠ |

⚠ NAFNet bricht bei einem Foto komplett ein (4,1 dB). Auch ohne diesen Ausreißer liegt es
überall hinter SCUNet.

### Ergebnisse A: echte Fotos

- Stark verrauschte Aufnahmen (σ ≈ 5): SCUNet entfernt das Rauschen nahezu vollständig
  (σ < 0,15) und erhält dabei Farben, Kontrast und feine Kanten (Seile, Gitter). NAFNet
  verschiebt die Farben sichtbar: blasser, flauer, Rottöne kippen. Der Grund ist vermutlich,
  dass SIDD nur aus Handyfotos besteht. nlmeans lässt viel Farbrauschen stehen.
- Leicht verrauschte Aufnahmen (σ ≈ 1–1,8): Alle Verfahren entfernen mehr als nötig. SCUNet
  erhält nur 56–78 % der Detail-Energie, abgeschabte Flächen und Stoff wirken wachsartig.

### Folgerungen für die Pipeline (Etappe 2)

1. **SCUNet ist das Standardmodell.** NAFNet fliegt raus. Als Alternative mit mehr Textur
   prüfe ich in Etappe 2 die GAN-Variante von SCUNet (gleiche Lizenz, gleiche Größe).
2. **Automatik in drei Fällen** (kalibriert an 5 unbearbeiteten RAWs und 3 DNGs aus DxO
   PureRAW / DeepPRIME 3):

   Farbrauschen wird auf **halber Auflösung** gemessen (×2 zurückgerechnet): JPEGs speichern
   Farbe nur in halber Auflösung (4:2:0), auf dem feinsten Raster ist dort nie Farbrauschen
   messbar. Gemessen an 5 unbearbeiteten RAWs, 3 DxO-DNGs und 6 JPEGs eines Pixel 8 Pro:

   | Herkunft | Luma-σ | Chroma-σ | Chroma/Luma |
   |---|---|---|---|
   | unbearbeitete Canon-RAWs (LibRaw) | 1,0–5,2 | 1,8–8,9 | 1,7–1,9 |
   | dieselben RAWs als JPEG 4:2:0, Q92 | 0,7–5,7 | 0,45–4,45 | 0,5–0,9 |
   | DxO DeepPRIME 3 | 0,3–5,7 | 0,07–0,33 | 0,05–0,43 |
   | Pixel 8 Pro, Tageslicht-JPEG | 0,0–3,5¹ | 0,00 | 0 |
   | iPhone 12 Pro / iPad Air / Samsung A32, HEIC | 0,0–0,7 | 0,00 | 0 |

   ¹ Die Pixel-Fotos haben keine glatten Flächen; Luma misst dort Putz- und Steinstruktur.
   In wirklich ruhigen Blöcken hat Handy-Software plus JPEG jedes feine Detail entfernt.

   - **Bereits entrauscht:** Chroma-σ < 0,6 und Chroma/Luma < 0,6 → Regler startet auf 0,
     Hinweis „Bild ist bereits entrauscht“. Ein KI-Entrauscher entfernt Farbrauschen fast
     vollständig, lässt aber oft bewusst feines Helligkeitskorn stehen (ein DxO-Bild hat
     Luma-σ 5,7 bei Chroma 0,16). Das Korn ist gewollt und wird nicht weggerechnet.
   - **Kein Rauschen:** Luma-σ < 1,5 → Regler auf 0, Hinweis „kein Rauschen erkannt“.
   - **Schwarz-Weiß-Bilder** haben nie Farbrauschen; für sie gilt nur die Luma-Regel
     (Farbigkeit |Cb|+|Cr| < 2 erkennt sie).
   - **Kein Rauschen:** Luma-σ < 1,5 → Regler auf 0, Hinweis „kein Rauschen erkannt“.
   - **Sonst:** Startwert nach Punkt 3, frei überschreibbar.
   Damit werden Andres DxO-Fotos nicht doppelt entrauscht. Der Regler bleibt in allen Fällen
   bedienbar, und die Vorschau zeigt „Original / nur entrauscht / Endergebnis“.
3. **Restrauschen wird beim Hochskalieren verstärkt** (Etappe 2, Test mit einem Ausschnitt
   aus IMG_3629, Luma-σ 6,5, siehe Vergleichsbilder unten). Die Stärke ist eine Mischung
   Original ↔ SCUNet, und was davon an Rauschen übrig bleibt, machen beide Modellfamilien zu
   „Würmern“ oder grobem Korn, sobald es über σ ≈ 1,5 liegt. Deshalb:
   - **Foto-Modelle (Real-HAT-GAN):** Sie entfernen sogar starkes Sensorrauschen selbst und
     sauber. Ohne Vorentrauschen war das Ergebnis genauso gut wie mit 100 % SCUNet, jede
     Teilmischung war schlechter. Die Automatik bleibt dort **aus** (Hinweis „das Fotomodell
     entfernt es selbst“), lässt sich aber einschalten.
   - **Saubere Bilder / Maximale Qualität:** Ohne Entrauschen wird Rauschen massiv verstärkt.
     Die Automatik wählt die Stärke so, dass rund **σ 0,8 Restrauschen** bleibt:
     Stärke = 1 − 0,8/σ (σ 6,5 → 88 %, σ 2 → 60 %). Das wirkt natürlicher als 100 %, das
     Flächen wachsartig glättet.
   - **Detailschutz** (Kantenmaske) nur bei leichtem Rauschen: voll bis σ 1,5, aus ab σ 3.
     Bei starkem Rauschen hätte er genau an den Kanten das Rauschen konserviert, wo der
     Upscaler am empfindlichsten ist.
   - **Farbrauschen** wird immer 1,5× stärker entfernt als Helligkeitsrauschen.
4. Kosten: SCUNet FP16 braucht **0,34 s/MP**, NAFNet FP16 0,22 s/MP, nlmeans (CPU) 0,44 s/MP.
   Neben HAT (10 s/MP) fällt das nicht ins Gewicht.

## 4b. Gesichter (Etappe 2)

- **Erkennung:** YuNet (MIT, 230 KB) über OpenCV. YuNet ist auf kleine bis mittlere Gesichter
  trainiert; ein bildfüllendes Porträt findet es nur verkleinert, winzige Gesichter einer
  Gruppe nur groß. Pixforge sucht deshalb auf **640 und 1600 px** und führt die Treffer per NMS
  zusammen. Auf dem Gruppenfoto der Solvay-Konferenz 1927 (900 px breit, 29 Personen) findet
  es alle 29 Gesichter.
- **Fehlalarme:** Eine Fensternische erreichte Score 0,72, echte Gesichter 0,86–0,92. Ein
  Restaurierer malt in Fehlalarme Haare und Augen hinein. Darum Schwelle 0,8 und eine
  Plausibilitätsprüfung der Merkmalspunkte (Nase zwischen Augen und Mund, Augenabstand
  20–80 % der Boxbreite).
- **Restaurierung:** Ausrichtung auf die FFHQ-Vorlage (512 px), Modell, Rückprojektion mit
  weicher ovaler Maske (Stirn bis Kinn). Gesichter, die im Ergebnis größer als 512 px sind,
  würden durch das 512-px-Modell weicher statt schärfer: Die Wirkung blendet zwischen 512 und
  1024 px Gesichtsgröße linear aus.
- **Modelle:** RestoreFormer++ (Standard, Apache-2.0) wirkt schärfer und kontrastreicher,
  GFPGAN 1.4 glatter („Beauty“-Look). Beide verbessern Augen und Mund kleiner Gesichter
  deutlich gegenüber HAT allein. GFPGAN ist nur ein optionaler Download mit Hinweis auf die
  nicht-kommerzielle NVIDIA-Klausel.
- **Parität:** RestoreFormer++ ONNX vs. PyTorch 104 dB (CPU) / 107 dB (DirectML), GFPGAN
  130 / 129 dB. GFPGAN läuft mit festem statt zufälligem Rauscheingang, damit Vorschau und
  Export identisch sind.

## 4c. Vorschau: schnelle Näherung und echte Kacheln (Etappe 4)

Ein Modell auf einer stark verkleinerten Kopie des Bildes ist keine brauchbare Vorschau. Bei
„Einpassen“ lief HAT auf etwa 275 × 180 px und erfand Strukturen. Gemessen bei 35 % Zoom
(Pixel-8-Foto): Mauerstruktur und Stoffdetails gehen verloren. Bei einem verrauschten CR3 war
die Näherung dagegen *schärfer* als das echte Ergebnis, weil das Verkleinern das Rauschen
schon entfernt hatte. Sie war also in beide Richtungen unzuverlässig.

- **Schnelle Vorschau** (`server/preview.py`): Ab 100 % Zoom laufen die echten Quellpixel
  durch das Modell, das Ergebnis ist exakt. Zwischen 50 % und 100 % ist es eine Näherung
  (Modell bei nativem Faktor auf leicht verkleinerter Quelle). Darunter wird das Original
  gezeigt, gekennzeichnet in der Statuszeile.
- **Echte Kacheln** (`server/tiles.py`): Ein Hintergrund-Thread rechnet das Bild in einem
  festen Raster wie der Export: Rasterweite = Modellkachel − 2 × 32 px Kontext, also 320 px
  bei 16 GB. Gesichter, deren 512er-Ausrichtungsquadrat eine Kachel berührt, werden ganz
  mitgerechnet, deshalb gibt es keine Naht durch Gesichter. Jede Kachel wird als
  JPEG-Pyramide gespeichert (1, 1/2, 1/4 …), die App zeichnet die Stufe, die zum Bildschirm
  passt. Reihenfolge: sichtbarer Bereich zuerst. Priorität im GPU-Gate: Vorschau > Export >
  Kacheln. Pro Bild und Einstellung gibt es eine Sitzung; der Ausschnitt begrenzt nur, welche
  Kacheln gerechnet werden. Speichergrenze für alle Sitzungen: 1 GB.
- **Messung** (20-MP-CR3, „Fotos“ 4×, RX 9070 XT): etwa 1,3 s pro Kachel, das ganze Bild
  nach rund 4,5 min. Eine 100-%-Vorschau wartet daneben höchstens eine Modellkachel
  (1,5–1,6 s statt 0,8–1,5 s). Pyramide pro Kachel: 145 KB + 30 KB + 10 KB + …
- Eine JPEG-Qualitätsschätzung pro Bild (`Settings.jpeg_qf`), damit Ausschnitte und Kacheln
  dieselbe FBCNN-Stärke wie der Export bekommen.

## 5. PyTorch mit ROCm unter Windows (optional, nur Entwicklung)

AMD bietet offizielle Windows-Wheels an: PyTorch 2.9.1 + ROCm 7.2.1, nur Python 3.12,
RX 9070 XT (gfx1201) unterstützt, Treiber ≥ 26.2.2 nötig. Der installierte Treiber vom
18.10.2026 ist neuer. **Noch nicht gemessen:** Die Installation (mehrere GB von
repo.radeon.com) wurde im automatischen Modus nicht freigegeben und wartet auf Andres OK.
Für das Endnutzer-Paket ist ROCm ohnehin keine Option: Es läuft nur auf neueren AMD-Karten und
wäre um ein Vielfaches größer. Interessant ist es nur, wenn es deutlich schneller ist, etwa für
eigene Experimente.

## 6. Lizenzen und Bereitstellung der Modelle

HAT (Apache-2.0), SCUNet (Apache-2.0), FBCNN (Apache-2.0) und NAFNet (MIT) erlauben Änderung
und Weitergabe, auch als konvertierte Dateien, wenn Lizenztext und Herkunftsvermerk beiliegen.
Plan: Pixforge lädt die vorkonvertierten ONNX-Dateien aus einem eigenen GitHub-Release
(`models-v1`) mit SHA256-Prüfung; im Release liegen die Lizenztexte und `THIRD_PARTY.md`.
Lokales Konvertieren ist damit nicht nötig und bleibt nur als Entwickler-Weg
(`python -m pixforge.models.export_onnx`).

Offen für Etappe 2: Lizenz der Gesichtserkennungs-Gewichte (facexlib/RetinaFace) und von
GFPGAN v1.4. Das GFPGAN-Repo hat eine gemischte LICENSE-Datei, die ich vor dem Einbinden
genau prüfe.

## 7. Netzwerk-Hinweis

Auf dem Entwicklungsrechner prüft Avast HTTPS-Verbindungen mit einem eigenen Root-Zertifikat.
Pixforge lädt Modelle deshalb über `truststore`, also mit dem Windows-Zertifikatsspeicher.
Damit funktioniert der Download auch bei Endnutzern mit solcher Software.
