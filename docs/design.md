# Design (Etappe 3, Entwurf zur Freigabe)

Interaktiver Entwurf: Design-Leinwand „Pixforge Designentwurf“ (privat, Link im Projektverlauf).
Hauptfenster und Einfach-Modus sind bedienbar: Regler, Modellwahl, Ansicht, Vorher/Nachher-
Trenner; Dunkel/Hell und Akzentfarbe sind als Tweaks umschaltbar.

Quellen im Repo: `design/tokens.json` (Farben, Schrift, Maße), `design/pixforge-*.svg`,
`design/pixforge.ico` (16–256 px), erzeugt mit `design/make_icons.py`.

## Leitidee

**Neutrales Grau, ein Akzent: Glut.** Fotos sollen wirken, nicht die Oberfläche. Die Grautöne
haben keinen Farbstich (R = G ≈ B), damit Hauttöne und Himmel nicht verfälscht wahrgenommen
werden. Der Akzent ist ein warmes Bernstein (#F2A541, „Glut“, passend zu *forge*), bewusst kein
Allerweltsblau. Er erscheint nur dort, wo Aufmerksamkeit hingehört: Auswahl, Hauptaktion
(„Exportieren“, „Verbessern“), Fortschritt, Gesichtsrahmen.

Feine 1-px-Linien statt Rahmen, Schatten nur für schwebende Elemente (Vorschau-Overlays,
Popover), großzügige Innenabstände, kurze Animationen ohne Federn.

## Logo

Eine glühende Scheibe. Die linke Hälfte ist eine grobe Treppe auf dem 4-px-Raster, die rechte
Hälfte die glatte Kurve: grob wird fein. Ein abgelöstes Pixel oben links ist der Funke.

- App-Icon: dunkles abgerundetes Quadrat (#141416, Radius 13,5/64), Scheibe #F2A541.
- Unter 48 px entfällt der Funke, die Scheibe wird größer gezeichnet
  (`pixforge-icon-small.svg`), damit die Treppe lesbar bleibt.
- Wortmarke: Inter SemiBold, Laufweite −3,5 %; auf hellem Grund Scheibe in #E8962E.
- Installer-Variante invertiert (dunkle Scheibe auf Bernstein).

## Farben

| Rolle | Dunkel (Standard) | Hell |
|---|---|---|
| Vorschau-Fläche | #0E0E10 | #E4E4E8 |
| Fenster | #141416 | #F7F7F8 |
| Panel | #19191C | #FFFFFF |
| Erhöht / Eingabe | #222226 | #F2F2F4 |
| Hover | #2B2B30 | #E8E8EC |
| Linie | #2A2A2F | #E2E2E7 |
| Text / Text 2 / Text 3 | #EDEDF0 / #A7A7B0 / #8C8C96 | #17171A / #4F4F58 / #676770 |
| Akzent (Fläche) | #F2A541, Text darauf #1C1206 | #E8962E, Text darauf #1C1206 |
| Akzent als Text | #F2A541 | #8F5200 |
| Fertig / Fehler / Hinweis | #56C68F / #F26D6D / #8AB4F8 | #177A4B / #C42B2B / #1F5FBF |

Kontraste (WCAG, nachgerechnet): Text 15,0:1 (dunkel) / 17,9:1 (hell), Text 2 7,4 / 8,1,
Text 3 5,3 / 5,6, Akzent-Text 8,6 / 6,2, Text auf Akzent 9,0 / 7,8. Alle ≥ 4,5:1.
Fertig und Fehler unterscheiden sich auch in der Helligkeit, nicht nur im Farbton.

## Schrift

Inter (SIL Open Font License), wird mit der App ausgeliefert. Ziffern mit fester Breite
(`tnum`), damit Auflösungen, Prozente und Restzeiten beim Zählen nicht springen.
Profi-Modus: Text 13 px, Panel-Überschriften 14/600, Werte 14/600, kleinste Schrift 11 px.
Einfach-Modus eine Stufe größer: Text 14–15 px, Hauptknopf 17/700, Schalter 44×26 px.

## Fenster und Layout (1440 × 900, rahmenloses Electron-Fenster)

```
┌ Titelleiste 36 ─ Logo · Pixforge · Dateiname ····················· _ □ × ┐
├ Werkzeugleiste 52 ─ Öffnen + | Einzeln Split Nebeneinander | Einpassen 100 % 200 % ·· Einfach|Profi  [Exportieren] ┤
│ Vorschau (flexibel)                                       │ Einstellungen 340 │
│  Vorher/Nachher mit ziehbarem Trenner,                    │  Modell (4 Kacheln)│
│  Gesichtsrahmen (Klick schließt aus)                      │  Größe 1× 2× 4× 6× Frei│
│  unten: Original | Nur entrauscht | Ergebnis   Gesichter 52 %│  Rauschminderung  │
│                                                           │  Feinschliff      │
│                                                           │  Gesichter        │
├ Warteschlange 112 ─ Miniaturen mit Status, Gesichts-Symbol, Fortschritt · Pausieren [Alle verarbeiten] ┤
└ Statusleiste 26 ─ GPU · DirectML · FP16 · VRAM · Kachel ············ Bild 3 von 5 · Restzeit ┘
```

- **Modell-Kacheln** mit Kurzbeschreibung und verfügbaren Faktoren: Fotos (Standard),
  Fotos schärfer, Grafiken, Grafiken groß (die beiden letzten sind auf saubere, verkleinerte
  Bilder trainiert und machen aus Kamerafotos Mosaik-Muster, daher der Hinweis bei Fotos).
- **Größe**: Faktor als Segmente, darunter Ein- → Ausgabe-Auflösung, Megapixel und eine
  ehrliche Zeitschätzung für das gewählte Gerät.
- **Rauschminderung**: Regler mit „Auto“-Marke und Begründung in Klartext („das Fotomodell
  entfernt es selbst“, „Bild ist bereits entrauscht“, …), siehe docs/backend.md Abschnitt 4.
- **Gesichter**: Schalter, Anzahl erkannt/ausgeschlossen, Stärke, Modell
  (RestoreFormer++ / GFPGAN mit Hinweis „nicht-kommerziell“), „Nur das größte Gesicht“.
- **Vorschau**: rechnet nur den sichtbaren Ausschnitt, aktualisiert ~300 ms nach einer
  Regleränderung (Etappe 4).
- Ab schmalen Fenstern rutscht das Panel unter die Vorschau, die Werkzeugleiste bricht um.

## Einfach-Modus

Für Laien: Vorher/Nachher groß, rechts nur „Wie groß soll das Bild werden?“ (2× / 4× als große
Karten mit Zielauflösung und Verwendungszweck), zwei Schalter (Gesichter verbessern mit Anzahl,
Rauschen entfernen „automatisch, nur wenn nötig“), ein großer Knopf **Verbessern**, darunter
wohin gespeichert wird. Link „Profi-Einstellungen anzeigen“.

## Startbildschirm

Große gestrichelte Ablagefläche „Bilder hierher ziehen“ mit Bildmarke, „Bilder auswählen“ und
„Ordner öffnen“, Liste der Formate in Laiensprache („HEIC (Handy)“, „RAW (Kamera)“). Darunter
beim ersten Start die Karte **Einmalige Einrichtung** mit Fortschritt (Modelle ~380 MB:
Fotos 47 MB, Rauschminderung 40 MB, Gesichter 294 MB, Erkennung 0,2 MB), danach ohne Internet.
Die Statusleiste meldet die gefundene Grafikkarte.

## Barrierearmut

Echte Knöpfe, Schalter (`role="switch"`), Regler (`input type=range`) und Beschriftungen,
sichtbarer Fokusrahmen in Akzentfarbe, Tastaturbedienung für alle Aktionen (Shortcuts in
Etappe 4), Mindestschrift 11 px, Kontraste siehe oben, „Bewegung reduzieren“ von Windows wird
respektiert. Oberfläche auf Deutsch und Englisch, Sprache nach System.

## Aufgabe, Größe und Ausschnitt (nach Rückmeldung ergänzt)

- **Aufgabe:** Umschalter „Vergrößern | Nur entrauschen“ oben im Größen-Abschnitt.
  *Nur entrauschen* behält die Größe, nutzt kein Hochskalier-Modell (die Modell-Kacheln werden
  ausgegraut), dauert Sekunden (≈ 0,4 s/MP) und erfindet keine Details. Ist das Bild schon
  entrauscht, sagt Pixforge das, statt eine unveränderte Kopie zu speichern.
- **1× = KI-Schärfen:** rechnet intern 4× und verkleinert wieder. Damit niemand sich über die
  Dauer wundert, erscheint bei 1× ein Hinweiskasten („dauert so lange wie 4×, nur Rauschen
  entfernen geht mit ‚Nur entrauschen‘ in Sekunden“) und die Zeitschätzung trägt „(wie 4×)“.
- **6×:** im Profi-Modus immer wählbar, mit Hinweis „Details werden teilweise neu erzeugt,
  originalgetreu höchstens 4×“ sowie Warnung zu Zeit, Arbeitsspeicher und Dateigröße.
  Im Einfach-Modus nur, wenn das Ergebnis unter etwa 100 MP bleibt (alte Fotos, kleine Bilder).
  Ein 12-MP-Handyfoto ergäbe bei 6× rund 440 MP und bräuchte 6–7 Minuten.
- **Einfach-Modus**, Frage „Was soll passieren?“: Karten *Gleiche Größe* (nur Rauschen weg),
  *2×*, *4×*, *6×* (nur wenn sinnvoll).
- **Ausschnitt:** Werkzeug „Ausschnitt“ in der Werkzeugleiste. Rahmen mit Drittel-Raster und
  Anfassern, abgedunkelte Umgebung, Seitenverhältnisse Frei, 1:1, 4:3, 3:2, 16:9, 10×15,
  Anzeige „Ausschnitt 234 × 299 px → 936 × 1.196 px“. Gerechnet wird nur der Ausschnitt plus
  32 px echter Umgebung (danach abgeschnitten, keine Randartefakte). Ergebnis ist ein **neues
  Bild** `…_pixforge_ausschnitt.jpg`, das Original bleibt unberührt. Im Einfach-Modus als Link
  „Nur einen Ausschnitt“. Backend: `Settings.crop`, CLI `--crop X,Y,B,H`.

## Copyright

„© 2026 Andre Strahl“ steht in der Statusleiste des Hauptfensters (rechts außen), auf dem
Startbildschirm neben der Versionsnummer, im Einfach-Modus unter den Links und im Info-Dialog
(Etappe 4); dazu im Installer und in `LICENSE`/`NOTICE` des Repos.

## Freigabe

Freigegeben am 01.11.2026: Akzent Glut (#F2A541), Logo „glühende Scheibe mit Pixeltreppe“,
Dunkel als Standard, Hell als Option. Ergänzt um „Nur entrauschen“, gekennzeichnetes
1×-KI-Schärfen, 6×-Regel, Ausschnitt-Werkzeug und Copyright.
