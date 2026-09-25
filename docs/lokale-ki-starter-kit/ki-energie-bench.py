#!/usr/bin/env python3
"""ki-energie-bench: Tempo und Stromverbrauch eines lokalen LLM-Servers messen.

Schickt Anfragen an eine OpenAI-kompatible API (llama-server, Ollama, LM Studio)
und misst dabei optional die Leistungsaufnahme über eine Mess-Steckdose
(Shelly Gen1/Gen2+, Tasmota) oder einen Home-Assistant-Sensor.

Ergebnis je Lauf: Tokens/s, mittlere Watt, Wh pro 1.000 Tokens, Cent pro
1.000 Tokens. Dazu Leerlauf-Watt vor und nach den Läufen samt Jahreskosten.

Nur Python-Standardbibliothek (ab Python 3.8), keine Installation nötig.

Beispiele:
  ./ki-energie-bench.py --server http://ki-server:8080 --model qwen3.6-35b-a3b \\
      --power shelly:http://192.168.1.50
  HA_TOKEN=... ./ki-energie-bench.py --server http://ki-server:8080 \\
      --power ha:http://homeassistant.local:8123/sensor.ki_server_leistung
"""

import argparse
import csv
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

DEFAULT_PROMPT = (
    "Erkläre in etwa 300 Wörtern, warum die Textgenerierung großer Sprachmodelle "
    "vor allem durch die Speicherbandbreite begrenzt ist und was Mixture-of-Experts daran ändert."
)
HOURS_PER_YEAR = 24 * 365


def http_json(url, data=None, headers=None, timeout=10.0):
    """GET (data=None) oder POST (JSON) und die Antwort als JSON zurückgeben."""
    hdrs = {"Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    body = None
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# --------------------------------------------------------------------------- Strommessung


class ShellyGen2:
    """Shelly Plus/Pro/Gen3/Gen4: /rpc/Switch.GetStatus → apower (W)."""

    def __init__(self, base, channel, timeout):
        self.url = f"{base.rstrip('/')}/rpc/Switch.GetStatus?id={channel}"
        self.timeout = timeout

    def read_watts(self):
        return float(http_json(self.url, timeout=self.timeout)["apower"])


class ShellyGen1:
    """Shelly Plug / Plug S (Gen1): /meter/<n> → power (W)."""

    def __init__(self, base, channel, timeout):
        self.url = f"{base.rstrip('/')}/meter/{channel}"
        self.timeout = timeout

    def read_watts(self):
        return float(http_json(self.url, timeout=self.timeout)["power"])


class Tasmota:
    """Tasmota-Steckdosen: Status 8 → StatusSNS.ENERGY.Power (W, ggf. Liste je Kanal)."""

    def __init__(self, base, channel, timeout):
        self.url = f"{base.rstrip('/')}/cm?cmnd=Status%208"
        self.timeout = timeout

    def read_watts(self):
        power = http_json(self.url, timeout=self.timeout)["StatusSNS"]["ENERGY"]["Power"]
        if isinstance(power, list):
            return float(sum(power))
        return float(power)


class HomeAssistantSensor:
    """Beliebiger Leistungssensor (W oder kW) über die Home-Assistant-REST-API."""

    def __init__(self, target, token, timeout):
        if not token:
            raise ValueError("Für --power ha:... wird ein Long-Lived Access Token in HA_TOKEN benötigt.")
        base, _, entity_id = target.rstrip("/").rpartition("/")
        if not base or "." not in entity_id:
            raise ValueError("Format: ha:http://homeassistant.local:8123/sensor.name")
        self.url = f"{base}/api/states/{urllib.parse.quote(entity_id)}"
        self.headers = {"Authorization": f"Bearer {token}"}
        self.timeout = timeout

    def read_watts(self):
        data = http_json(self.url, headers=self.headers, timeout=self.timeout)
        value = float(data["state"])  # "unavailable" & Co. lösen hier einen Fehler aus
        if data.get("attributes", {}).get("unit_of_measurement") == "kW":
            value *= 1000
        return value


def make_power_source(spec, channel, timeout):
    if not spec or spec == "none":
        return None
    kind, sep, target = spec.partition(":")
    if not sep or not target:
        raise ValueError(f"Ungültige --power-Angabe: {spec!r}")
    if kind == "shelly":
        return ShellyGen2(target, channel, timeout)
    if kind == "shelly1":
        return ShellyGen1(target, channel, timeout)
    if kind == "tasmota":
        return Tasmota(target, channel, timeout)
    if kind == "ha":
        return HomeAssistantSensor(target, os.environ.get("HA_TOKEN"), timeout)
    raise ValueError(f"Unbekannte Messquelle {kind!r} (erlaubt: shelly, shelly1, tasmota, ha, none)")


class PowerSampler(threading.Thread):
    """Liest die Leistung im Hintergrund in festen Abständen aus."""

    def __init__(self, source, interval):
        super().__init__(daemon=True)
        self.source = source
        self.interval = interval
        self.samples = []
        self.errors = 0
        self.last_error = None
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            started = time.monotonic()
            try:
                self.samples.append(self.source.read_watts())
            except Exception as exc:  # Messfehler sollen den Benchmark nicht abbrechen
                self.errors += 1
                self.last_error = exc
            self._halt.wait(max(0.0, self.interval - (time.monotonic() - started)))

    def stop(self):
        self._halt.set()
        self.join(timeout=10)

    def mean_watts(self):
        return statistics.fmean(self.samples) if self.samples else None


def measure_idle(source, seconds, interval):
    sampler = PowerSampler(source, interval)
    sampler.start()
    time.sleep(seconds)
    sampler.stop()
    if sampler.errors and not sampler.samples:
        raise RuntimeError(f"Strommessung fehlgeschlagen: {sampler.last_error}")
    return sampler.mean_watts()


# --------------------------------------------------------------------------- LLM-Anfragen


def chat(args, max_tokens):
    payload = {
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }
    if args.model:
        payload["model"] = args.model
    if args.reasoning:
        payload["reasoning_effort"] = args.reasoning
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else None
    url = f"{args.server.rstrip('/')}/v1/chat/completions"
    started = time.monotonic()
    resp = http_json(url, data=payload, headers=headers, timeout=args.timeout)
    wall = time.monotonic() - started

    timings = resp.get("timings") or {}
    usage = resp.get("usage") or {}
    tokens = timings.get("predicted_n") or usage.get("completion_tokens") or 0
    gen_tps = timings.get("predicted_per_second")
    # Server ohne llama.cpp-Timings (z. B. Ollama): Tempo über die Wandzeit schätzen.
    tps_from_wall = gen_tps is None
    if tps_from_wall:
        gen_tps = tokens / wall if wall > 0 else 0.0
    return {
        "tokens": int(tokens),
        "gen_tps": float(gen_tps),
        "prompt_tps": timings.get("prompt_per_second"),
        "wall_s": wall,
        "tps_from_wall": tps_from_wall,
    }


def run_measured(args, source):
    sampler = None
    if source:
        sampler = PowerSampler(source, args.interval)
        sampler.start()
    try:
        result = chat(args, args.max_tokens)
    finally:
        if sampler:
            sampler.stop()
    result["avg_w"] = sampler.mean_watts() if sampler else None
    result["samples"] = len(sampler.samples) if sampler else 0
    if result["avg_w"] is not None and result["tokens"] > 0:
        result["wh"] = result["avg_w"] * result["wall_s"] / 3600
        result["wh_per_1k"] = result["wh"] / result["tokens"] * 1000
        result["ct_per_1k"] = result["wh_per_1k"] / 1000 * args.price * 100
    else:
        result["wh"] = result["wh_per_1k"] = result["ct_per_1k"] = None
    return result


# --------------------------------------------------------------------------- Ausgabe


def fmt(value, digits=1, unit=""):
    if value is None:
        return "–"
    return f"{value:.{digits}f}{unit}".replace(".", ",")


def yearly_cost(watts, price):
    return None if watts is None else watts * HOURS_PER_YEAR / 1000 * price


def csv_number(value, digits):
    """Zahl mit Dezimalkomma, passend zum Semikolon-CSV für deutsches Excel/LibreOffice."""
    return "" if value is None else f"{value:.{digits}f}".replace(".", ",")


def write_csv(path, rows):
    fields = [
        "zeitpunkt", "label", "server", "modell", "lauf", "tokens", "tok_s", "prompt_tok_s",
        "wandzeit_s", "watt_mittel", "wh", "wh_pro_1k", "ct_pro_1k", "leerlauf_vorher_w", "leerlauf_nachher_w",
    ]
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter=";")
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Tempo (tok/s) und Stromverbrauch (W, Wh pro 1.000 Tokens) eines lokalen LLM-Servers messen.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Messquellen für --power:\n"
            "  shelly:http://IP     Shelly Plus/Pro/Gen3/Gen4 (Gen2-API)\n"
            "  shelly1:http://IP    Shelly Plug / Plug S (Gen1)\n"
            "  tasmota:http://IP    Tasmota-Steckdose\n"
            "  ha:http://HA:8123/sensor.name   Home-Assistant-Sensor (Token in HA_TOKEN)\n"
            "  none                 nur Tempo messen\n"
        ),
    )
    p.add_argument("--server", required=True, help="Basis-URL, z. B. http://ki-server:8080 (Ollama: http://host:11434)")
    p.add_argument("--model", help="Modellname (Router-Modus: Abschnitt aus models.ini)")
    p.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY"), help="API-Key (Standard: $LLAMA_API_KEY)")
    p.add_argument("--power", default="none", help="Messquelle, siehe unten (Standard: none)")
    p.add_argument("--power-channel", type=int, default=0, help="Kanal der Mess-Steckdose (Standard: 0)")
    p.add_argument("--interval", type=float, default=0.5, help="Messintervall in Sekunden (Standard: 0,5)")
    p.add_argument("--idle-seconds", type=float, default=30,
                   help="Leerlauf-Messdauer vorher/nachher, 0 = aus (Standard: 30)")
    p.add_argument("--runs", type=int, default=3, help="Anzahl gemessener Läufe (Standard: 3)")
    p.add_argument("--max-tokens", type=int, default=512, help="Maximale Antwortlänge (Standard: 512)")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="Eigener Prompt")
    p.add_argument("--reasoning", choices=["none", "low", "medium", "high"],
                   help="reasoning_effort an den Server schicken (none = Thinking aus)")
    p.add_argument("--price", type=float, default=0.28, help="Strompreis in €/kWh (Standard: 0,28)")
    p.add_argument("--timeout", type=float, default=900, help="Timeout pro Anfrage in Sekunden (Standard: 900)")
    p.add_argument("--csv", help="Ergebnisse an diese CSV-Datei anhängen (Trennzeichen ;)")
    p.add_argument("--label", default="", help="Freitext für die CSV, z. B. 'RTX3060 PL170 Q4_K_M'")
    args = p.parse_args(argv)
    if args.runs < 1 or args.max_tokens < 1 or args.interval <= 0 or args.idle_seconds < 0:
        p.error("--runs, --max-tokens und --interval müssen positiv sein, --idle-seconds >= 0")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        source = make_power_source(args.power, args.power_channel, timeout=min(5.0, args.interval * 4 + 1))
        if source:
            source.read_watts()  # früh scheitern, wenn die Steckdose nicht erreichbar ist
    except Exception as exc:
        print(f"Fehler bei der Strommessung ({args.power}): {exc}", file=sys.stderr)
        return 2

    print(f"Server: {args.server}  Modell: {args.model or '(Standard)'}  Messquelle: {args.power}")
    try:
        idle_before = None
        if source and args.idle_seconds > 0:
            print(f"Leerlauf messen ({fmt(args.idle_seconds, 0)} s) …", flush=True)
            idle_before = measure_idle(source, args.idle_seconds, args.interval)

        print("Aufwärmen (lädt das Modell, falls nötig) …", flush=True)
        warm = chat(args, max_tokens=8)
        print(f"  erste Antwort nach {fmt(warm['wall_s'])} s (inkl. Laden/Aufwachen)")

        results = []
        for i in range(1, args.runs + 1):
            r = run_measured(args, source)
            results.append(r)
            line = f"  Lauf {i}: {r['tokens']} Tokens, {fmt(r['gen_tps'])} tok/s"
            if r["tps_from_wall"]:
                line += " (aus Wandzeit geschätzt)"
            if source:
                line += f", {fmt(r['avg_w'], 0, ' W')}, {fmt(r['wh_per_1k'], 2, ' Wh')} pro 1.000 Tokens"
            print(line, flush=True)
            if source and r["samples"] < 3:
                print("    Hinweis: wenige Messpunkte – längere Antworten (--max-tokens) messen genauer.")

        idle_after = None
        if source and args.idle_seconds > 0:
            print(f"Leerlauf nach den Läufen messen ({fmt(args.idle_seconds, 0)} s, Modell noch geladen) …", flush=True)
            idle_after = measure_idle(source, args.idle_seconds, args.interval)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        print(f"Server-Fehler {exc.code}: {detail}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1

    def median(key):
        values = [r[key] for r in results if r[key] is not None]
        return statistics.median(values) if values else None

    print("\n=== Ergebnis (Median über alle Läufe) ===")
    print(f"Tempo:                {fmt(median('gen_tps'))} tok/s")
    prompt_tps = median("prompt_tps")
    print(f"Prompt-Verarbeitung:  {fmt(prompt_tps, 0) + ' tok/s' if prompt_tps else 'vom Server nicht gemeldet'}")
    if source:
        print(f"Leistung unter Last:  {fmt(median('avg_w'), 0)} W")
        print(f"Energie:              {fmt(median('wh_per_1k'), 2)} Wh pro 1.000 Tokens "
              f"= {fmt(median('ct_per_1k'), 3)} ct bei {fmt(args.price, 2)} €/kWh")
        for name, watts in (("vorher", idle_before), ("nachher, Modell geladen", idle_after)):
            if watts is not None:
                cost = fmt(yearly_cost(watts, args.price), 0)
                print(f"Leerlauf {name}: {fmt(watts, 1)} W → {cost} € pro Jahr bei 24/7")

    if args.csv:
        stamp = datetime.now().isoformat(timespec="seconds")
        rows = [
            {
                "zeitpunkt": stamp, "label": args.label, "server": args.server, "modell": args.model or "",
                "lauf": i, "tokens": r["tokens"], "tok_s": csv_number(r["gen_tps"], 2),
                "prompt_tok_s": csv_number(r["prompt_tps"], 1),
                "wandzeit_s": csv_number(r["wall_s"], 2),
                "watt_mittel": csv_number(r["avg_w"], 1),
                "wh": csv_number(r["wh"], 4),
                "wh_pro_1k": csv_number(r["wh_per_1k"], 3),
                "ct_pro_1k": csv_number(r["ct_per_1k"], 4),
                "leerlauf_vorher_w": csv_number(idle_before, 1),
                "leerlauf_nachher_w": csv_number(idle_after, 1),
            }
            for i, r in enumerate(results, 1)
        ]
        write_csv(args.csv, rows)
        print(f"\nErgebnisse angehängt an {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
