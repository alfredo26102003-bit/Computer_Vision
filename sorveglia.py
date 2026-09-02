"""
INFRASTRUTTURA — ferma un comando se la GPU sfora potenza o temperatura.

Sorveglianza dei consumi: lancia un comando e lo FERMA se la GPU sfora.

PERCHE' NON BASTA IL CLOCK LOCK. `limiti_hw.ps1` mette un tetto al clock, e
quello e' il limite vero: la potenza non arriva piu' ai picchi che spengono
la macchina. Ma il tetto va CALIBRATO - 1500 MHz e' un punto di partenza
prudente, non una misura - e finche' non e' calibrato serve qualcuno che
guardi il tachimetro. Questo e' il tachimetro, con l'aggiunta che tira il
freno da solo.

PERCHE' FERMARE E' SICURO. Tutti gli esperimenti salvano in modo
incrementale e ripartono da dove erano: `exp_controlli.py` e
`exp_pooling.py` rileggono il loro JSON e saltano le celle gia' misurate,
la griglia scrive una riga alla volta. Interrompere costa al massimo la
cella in corso. Uno spegnimento improvviso invece costa tutto quello che
sta solo in memoria - ed e' gia' successo, due volte in un giorno.

Quindi la scelta e' fra perdere una cella e perdere tutto. Non e' una
scelta.

COSA SORVEGLIA
  power.draw     il consumo istantaneo: e' la grandezza che conta, perche'
                 quello che spegne la macchina e' il transitorio, non la
                 media
  temperature    di contorno: a 88 C questa macchina si e' spenta, ma i
                 Kernel-Power 41 sono arrivati anche a 71 C, quindi la
                 temperatura da sola non e' un buon allarme
  clocks.sm      per verificare che il tetto di limiti_hw.ps1 sia davvero
                 attivo: se qui vedi 3000 MHz, il lock non e' stato
                 applicato o e' caduto con un riavvio

USO
    python sorveglia.py --tetto 95 -- python exp_diversita.py --tag _casuale
    python sorveglia.py --tetto 95 --carico 85 -- python exp_controlli.py

Tutto quello che sta dopo `--` e' il comando da eseguire, passato di peso.
"""

import argparse
import csv
import os
import subprocess
import sys
import threading
import time

INTERVALLO = 2.0        # secondi fra una lettura e l'altra
CONSECUTIVE = 3         # letture sopra il tetto prima di fermare: una sola
                        # puo' essere un picco di misura di nvidia-smi

CAMPI = "power.draw,temperature.gpu,clocks.sm,utilization.gpu"


def leggi():
    """Una lettura di nvidia-smi. None se la scheda non risponde."""
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={CAMPI}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return None
        v = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        return {"potenza": float(v[0]), "temp": float(v[1]),
                "clock": float(v[2]), "uso": float(v[3])}
    except (subprocess.TimeoutExpired, ValueError, IndexError, OSError):
        return None


class Guardia(threading.Thread):
    def __init__(self, proc, tetto, tetto_temp, registro):
        super().__init__(daemon=True)
        self.proc = proc
        self.tetto = tetto
        self.tetto_temp = tetto_temp
        self.registro = registro
        self.picco = 0.0
        self.picco_temp = 0.0
        self.clock_max = 0.0
        self.letture = 0
        self.sopra = 0
        self.motivo = None
        self.ferma = threading.Event()

    def run(self):
        with open(self.registro, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "potenza_W", "temp_C", "clock_MHz", "uso_%"])
            t0 = time.time()
            while not self.ferma.is_set() and self.proc.poll() is None:
                m = leggi()
                if m is None:
                    time.sleep(INTERVALLO)
                    continue

                self.letture += 1
                self.picco = max(self.picco, m["potenza"])
                self.picco_temp = max(self.picco_temp, m["temp"])
                self.clock_max = max(self.clock_max, m["clock"])
                w.writerow([f"{time.time() - t0:.1f}", m["potenza"], m["temp"],
                            m["clock"], m["uso"]])
                f.flush()

                sfora = (m["potenza"] > self.tetto
                         or m["temp"] > self.tetto_temp)
                # Il conteggio si azzera appena una lettura rientra: si ferma
                # per un carico SOSTENUTO sopra il tetto, non per un picco
                # isolato che nvidia-smi puo' anche aver misurato male.
                self.sopra = self.sopra + 1 if sfora else 0

                if self.sopra >= CONSECUTIVE:
                    self.motivo = (
                        f"{m['potenza']:.0f} W (tetto {self.tetto}) "
                        f"a {m['temp']:.0f} C, {CONSECUTIVE} letture di fila")
                    self._abbatti()
                    return

                time.sleep(INTERVALLO)

    def _abbatti(self):
        print(f"\n{'!' * 70}", flush=True)
        print(f"SFORAMENTO: {self.motivo}", flush=True)
        print("Fermo il comando. Gli esperimenti ripartono da dove erano:",
              flush=True)
        print("rilancia lo stesso comando con un tetto di clock piu' basso.",
              flush=True)
        print(f"{'!' * 70}\n", flush=True)
        try:
            self.proc.terminate()
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Lancia un comando sorvegliando i consumi della GPU.")
    ap.add_argument("--tetto", type=float, default=95.0,
                    help="potenza massima tollerata in W (predefinito 95)")
    ap.add_argument("--tetto-temp", type=float, default=80.0,
                    help="temperatura massima tollerata in C")
    ap.add_argument("--registro", default=None,
                    help="dove scrivere il CSV delle letture")
    ap.add_argument("comando", nargs=argparse.REMAINDER,
                    help="dopo -- : il comando da eseguire")
    a = ap.parse_args()

    cmd = a.comando[1:] if a.comando and a.comando[0] == "--" else a.comando
    if not cmd:
        ap.error("manca il comando. Esempio:\n"
                 "  python sorveglia.py --tetto 95 -- python exp_diversita.py")

    prima = leggi()
    if prima is None:
        sys.exit("nvidia-smi non risponde: senza letture non sorveglio niente.")

    registro = a.registro or os.path.join(
        "runs", f"consumi_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    os.makedirs(os.path.dirname(registro) or ".", exist_ok=True)

    print(f"tetto      : {a.tetto:.0f} W, {a.tetto_temp:.0f} C")
    print(f"a riposo   : {prima['potenza']:.0f} W, {prima['temp']:.0f} C, "
          f"clock {prima['clock']:.0f} MHz")
    if prima["clock"] > 2000:
        print("            ATTENZIONE: clock alto da fermo. Se hai applicato")
        print("            limiti_hw.ps1, controlla che non sia caduto con un")
        print("            riavvio - il lock non sopravvive al riavvio.")
    print(f"registro   : {registro}")
    print(f"comando    : {' '.join(cmd)}")
    print("-" * 70, flush=True)

    t0 = time.time()
    proc = subprocess.Popen(cmd)
    guardia = Guardia(proc, a.tetto, a.tetto_temp, registro)
    guardia.start()

    try:
        codice = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        codice = 130
    guardia.ferma.set()
    guardia.join(timeout=5)

    durata = time.time() - t0
    print("-" * 70)
    print(f"durata     : {durata / 60:.1f} min")
    print(f"letture    : {guardia.letture}")
    print(f"picco      : {guardia.picco:.0f} W   (tetto {a.tetto:.0f})")
    print(f"temp max   : {guardia.picco_temp:.0f} C")
    print(f"clock max  : {guardia.clock_max:.0f} MHz")
    print(f"registro   : {registro}")

    if guardia.motivo:
        print(f"\nFERMATO per sforamento: {guardia.motivo}")
        sys.exit(2)

    # La calibrazione si fa qui: il picco misurato dice se c'e' margine per
    # alzare il clock, che e' l'unico modo di recuperare tempo senza
    # rimettersi nella condizione di prima.
    margine = a.tetto - guardia.picco
    if codice == 0 and margine > 20:
        print(f"\n{margine:.0f} W di margine sotto il tetto. C'e' spazio per")
        print("alzare il clock e finire prima:")
        print("    .\\limiti_hw.ps1 -Clock 1800      (da amministratore)")
    elif codice == 0:
        print(f"\nPicco a {margine:.0f} W dal tetto: la taratura va bene cosi'.")
    sys.exit(codice)
