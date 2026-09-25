# Starter-Kit: Lokale KI, sparsam betrieben

Fertige Dateien zum Workflow in [`../lokale-ki-low-power-workflow.md`](../lokale-ki-low-power-workflow.md).

| Datei | Wofür |
|---|---|
| [`llama-server.service`](llama-server.service) + [`llama-server.env`](llama-server.env) | llama.cpp als Linux-Dienst im **Router-Modus**. Mehrere Modelle, immer nur eins im Speicher, Entladen nach 5 min Leerlauf, API-Key. |
| [`models.ini`](models.ini) | Modell-Presets: Allrounder (Qwen3.6-35B-A3B), Sprachassistent (Gemma 4 26B-A4B ohne Thinking), Coding (Qwen3.8-27B). Mit KV-Cache-Quantisierung, `n-cpu-moe` und MTP als Vorlage. |
| [`gpu-powerlimit.service`](gpu-powerlimit.service) | NVIDIA-Power-Limit beim Booten setzen (z. B. RTX 3090 → 260 W). |
| [`ki-energie-bench.py`](ki-energie-bench.py) | **Mess-Skript:** tok/s, Watt unter Last, Wh und Cent pro 1.000 Tokens, Leerlauf-Watt und Jahreskosten. Liest Shelly, Tasmota oder einen Home-Assistant-Sensor. |
| [`home-assistant/ki_server.yaml`](home-assistant/ki_server.yaml) | HA-Package: hochgerechnete Jahreskosten, Wake-on-LAN-Skript, Warnung „Server schläft nicht“. |

---

## Schnellstart (Linux-Server)

### 1. llama.cpp installieren

Zwei Wege:

- **Fertige Binärdateien:** Von den [GitHub-Releases](https://github.com/ggml-org/llama.cpp/releases)
  die passende Variante laden (CUDA, Vulkan, ROCm, SYCL oder CPU). `llama-server` danach nach
  `/usr/local/bin/` kopieren.
- **Selbst bauen:** Anleitung in der
  [Build-Doku](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).

Auf dem Mac geht es mit `brew install llama.cpp`. Die `models.ini` und das Mess-Skript
funktionieren dort genauso. Nur die systemd-Dateien entfallen.

### 2. Modelle herunterladen

GGUF-Dateien von Hugging Face nach `/var/lib/llama-server/models/` legen, z. B. mit
`hf download <repo> <datei.gguf> --local-dir /var/lib/llama-server/models`.

- Als Quantisierung **Q4_K_M** oder **UD-Q4_K_XL** nehmen.
- Danach die Dateinamen in `models.ini` anpassen.
- Für MTP brauchst du eine GGUF, die die MTP-Köpfe enthält. Das steht auf der Modellkarte.

### 3. Dienst einrichten

```bash
sudo useradd --system --home /var/lib/llama-server --create-home llama
sudo mkdir -p /etc/llama-server /var/lib/llama-server/models
sudo cp llama-server.env models.ini /etc/llama-server/
sudo chmod 600 /etc/llama-server/llama-server.env   # enthält den API-Key
sudoedit /etc/llama-server/llama-server.env          # LLAMA_API_KEY setzen!
sudo cp llama-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llama-server
journalctl -u llama-server -f                        # Log ansehen
```

Den API-Key erzeugst du z. B. mit `openssl rand -hex 24`.

### 4. Testen

```bash
curl -s http://ki-server:8080/v1/models -H "Authorization: Bearer $LLAMA_API_KEY"

curl -s http://ki-server:8080/v1/chat/completions \
  -H "Authorization: Bearer $LLAMA_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "qwen3.6-35b-a3b", "messages": [{"role": "user", "content": "Hallo!"}]}'
```

Der Wert für `"model"` ist der Abschnittsname aus `models.ini`. Die erste Anfrage lädt das
Modell, deshalb dauert sie länger.

### 5. Optional: Power-Limit (NVIDIA)

```bash
nvidia-smi -q -d POWER | grep -i "power limit"   # erlaubten Bereich ansehen
sudo cp gpu-powerlimit.service /etc/systemd/system/
sudoedit /etc/systemd/system/gpu-powerlimit.service   # Wert anpassen (Standard 260 W)
sudo systemctl daemon-reload && sudo systemctl enable --now gpu-powerlimit
```

---

## Messen mit `ki-energie-bench.py`

Du brauchst nur Python 3.8 oder neuer, keine weiteren Pakete. Das Skript läuft auf jedem
Rechner im Netz, nicht nur auf dem Server.

```bash
# Nur Tempo
./ki-energie-bench.py --server http://ki-server:8080 --model qwen3.6-35b-a3b

# Tempo + Strom über eine Shelly-Steckdose (Plus/Pro/Gen3/Gen4) vor dem Server
./ki-energie-bench.py --server http://ki-server:8080 --model qwen3.6-35b-a3b \
    --power shelly:http://192.168.1.50 --csv messungen.csv --label "Q4_K_M, PL 260 W"

# Strom über einen beliebigen Home-Assistant-Sensor (Zigbee, Matter, …)
read -rsp "HA-Token: " HA_TOKEN && export HA_TOKEN   # Long-Lived Access Token, landet nicht in der History
./ki-energie-bench.py --server http://ki-server:8080 --model ha-assist --reasoning none \
    --power ha:http://homeassistant.local:8123/sensor.ki_server_leistung

# Ollama (OpenAI-kompatible API unter /v1)
./ki-energie-bench.py --server http://ki-server:11434 --model qwen3:8b
```

**Messquellen für `--power`:**

- `shelly:` für Shelly Gen2 und neuer
- `shelly1:` für Shelly Plug / Plug S der ersten Generation
- `tasmota:` für Tasmota-Steckdosen
- `ha:` für Home-Assistant-Sensoren, Token in `HA_TOKEN`
- `none` misst nur das Tempo

Beispielausgabe:

```
=== Ergebnis (Median über alle Läufe) ===
Tempo:                52,4 tok/s
Prompt-Verarbeitung:  610 tok/s
Leistung unter Last:  148 W
Energie:              0,79 Wh pro 1.000 Tokens = 0,022 ct bei 0,28 €/kWh
Leerlauf vorher: 9,8 W → 24 € pro Jahr bei 24/7
Leerlauf nachher, Modell geladen: 31,5 W → 77 € pro Jahr bei 24/7
```

*(Beispielwerte zur Illustration)*

**So liest du das Ergebnis:**

- **„Leerlauf nachher“ deutlich höher als „vorher“:** Das geladene Modell oder die GPU hält
  den Rechner wach. Prüfe, ob `--sleep-idle-seconds` wirklich spart, und schau dir das offene
  llama.cpp-Issue #19318 an. Wenn nicht, hilft Wake-on-LAN mit Suspend (siehe unten).
- **Varianten vergleichen:** Nutze dieselbe `--csv`-Datei mit verschiedenen `--label`-Werten,
  z. B. Quantisierung, Power-Limit, `n-cpu-moe` oder MTP an/aus. Die CSV nutzt `;` und
  Dezimalkomma und öffnet sich direkt in Excel oder LibreOffice.
- **Genauigkeit:** Steckdosen melden etwa einmal pro Sekunde. Längere Antworten (Standard
  512 Tokens) und mehrere Läufe (`--runs`) mitteln das aus. Home-Assistant-Sensoren
  aktualisieren oft seltener, dann `--max-tokens 2048` nehmen.
- **Ohne llama.cpp-Timings** (z. B. bei Ollama) schätzt das Skript das Tempo aus der Wandzeit
  und markiert das in der Ausgabe.

---

## Home Assistant anbinden

1. **Package einspielen:**
   - `home-assistant/ki_server.yaml` nach `/config/packages/` kopieren.
   - Packages in `configuration.yaml` aktivieren (siehe Kopf der Datei).
   - Sensorname, MAC-Adresse und Strompreis anpassen.
   - Home Assistant neu starten.
2. **LLM als Konversationsagent:** Eine OpenAI-kompatible Integration einrichten, z. B.
   *Extended OpenAI Conversation* aus HACS. Dort eintragen:
   - Base-URL `http://ki-server:8080/v1`
   - API-Key aus `llama-server.env`
   - Modell `ha-assist`
3. **Tipps aus der Community:**
   - Nur 30–40 Entitäten für Assist freigeben.
   - Einen eigenen System-Prompt mit Beispielen schreiben.
   - Thinking ausschalten. Das übernimmt das Preset `ha-assist` mit `reasoning-budget = 0`.

---

## Wake-on-LAN: großen Rechner schlafen legen

1. **Im BIOS/UEFI** Wake-on-LAN aktivieren. Danach unter Linux:

   ```bash
   sudo ethtool -s <netzwerkkarte> wol g   # sofort, bis zum Neustart
   # dauerhaft mit NetworkManager:
   sudo nmcli connection modify <verbindung> 802-3-ethernet.wake-on-lan magic
   ```

2. **Schlafen legen** mit `systemctl suspend`, manuell oder automatisch.
3. **Wecken** über das Home-Assistant-Skript `script.ki_server_wecken` oder per
   `wakeonlan <MAC>` von einem anderen Rechner.
4. **Komplett automatisch** (Anfrage kommt → wecken → weiterleiten → später wieder schlafen):
   - [sleepyllama](https://github.com/FarFetchd/sleepyllama)
   - [Wakezilla](https://guibeira.dev/wakezilla-en.html)

   Beide laufen auf einem sparsamen Always-on-Gerät wie einem Pi oder Mini-PC.

---

## Sicherheit

- **Mit `LLAMA_ARG_HOST=0.0.0.0`** ist der Server im ganzen LAN erreichbar. Setz deshalb
  **immer einen API-Key** und gib Port 8080 nicht ins Internet frei. Für den Zugriff von
  unterwegs ist ein VPN (z. B. WireGuard) die bessere Wahl.
- **Der systemd-Dienst läuft als eigener Benutzer `llama`** und darf nur nach
  `/var/lib/llama-server` schreiben.
- **Zugangsdaten gehören in Umgebungsvariablen**, nicht in die Kommandozeile: `HA_TOKEN` und
  `LLAMA_API_KEY` für das Mess-Skript. So landen sie nicht in der Shell-History.
