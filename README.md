# Cuphead RL – Prototyp

Früher Prototyp für meine Bachelorarbeit über Reinforcement Learning in Videospiel-Bosskämpfen.
Ziel war es, das **originale Spiel Cuphead** als Trainingsumgebung nutzbar zu machen. Das Skript liest dafür den Bildschirm aus, erkennt den Spielzustand per Bildverarbeitung und steuert das Spiel über simulierte Tastatureingaben.

**Stack:** Python · OpenCV · NumPy · mss (Screen Capture) · pydirectinput

---

## Was der Prototyp kann

- **Screen Capture in Echtzeit**: Einzelne Bildschirmregionen (HP-Anzeige, Super-Meter, Fortschrittsbalken) werden pro Schritt mit `mss` erfasst.
- **HP-Erkennung**: Template Matching (`cv2.matchTemplate`) gegen Vorlagen der HP-Anzeige, inklusive der blinkenden Zustände bei 1 HP und einer Mindest-Konfidenz gegen Fehlerkennungen.
- **Schaden als Belohnungssignal**: Das Spiel zeigt den verursachten Schaden nicht direkt an. Der `DamageTracker` leitet ihn deshalb aus dem Füllstand der Super-Meter-Karten ab, über eine HSV-Farbmaske. Zusätzlich erkennt er, wann der Super-Angriff (Ult) bereit ist.
- **Steuerung**: Tasten werden über `pydirectinput` gedrückt und gehalten, sodass Bewegen, Ducken, Schießen und Dashen kombinierbar sind. Episoden starten und enden automatisch.
- **Logging**: Jede Aktion landet mit HP, Schadens-Reward und Ereignissen in einer CSV-Datei, dazu kommen Frames und Screenshots des Fortschrittsbalkens pro Episode.

Die Aktionen wählt in diesem Stand noch ein Zufallsagent. Der Prototyp diente dazu, Beobachtung, Belohnung und Steuerung zuverlässig zu machen, bevor ein lernender Agent angebunden wird.

## Aufbau

```
.
├── main.py                 # Episoden-Loop: Capture → Zustand erkennen → Aktion → Logging
├── damage_tracker.py       # Schadens- und Ult-Erkennung über das Super-Meter
├── config.yaml             # Run-Name, Schrittweite, Episodenanzahl
├── templates/              # Vorlagen für HP- und Karten-Erkennung
└── *_test.py               # Einzeltests für Screen Capture, Input und Control Loop
```

## Weiterentwicklung

Mit diesem Ansatz ist das Training an die Echtzeit gebunden und nicht parallelisierbar. Für die Bachelorarbeit wurden die Bosskämpfe deshalb später in Godot als eigene Umgebungen nachgebaut. Dort lassen sich mehrere Agenten parallel und deutlich schneller trainieren.

---

*Cuphead ist eine Marke von StudioMDHR. Dieses Projekt steht in keiner Verbindung zu StudioMDHR. Die Bildvorlagen in `templates/` sind kleine Ausschnitte der Spieloberfläche und dienen ausschließlich der Bilderkennung.*
