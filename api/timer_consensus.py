"""
Timer Consensus — Conseil des Remoras — Blade Runner V6
========================================================

Convoque le conseil toutes les N minutes avec une deadline stricte.
Lit  : cerveau/esprit_etat.json      (ecrit par ESPRIT.py)
Ecrit: cerveau/remora_consensus.json  (lu   par ESPRIT.py)

Si la deadline expire -> votes recus + fallback 0.1 pour les muets.

Modes :
  python api/timer_consensus.py                  # boucle 5 min
  python api/timer_consensus.py --interval 120   # boucle 2 min
  python api/timer_consensus.py --once           # cycle unique
  python api/timer_consensus.py --url http://localhost:8001

Import depuis ESPRIT.py :
  from api.timer_consensus import TimerConsensus
  timer = TimerConsensus()
  timer.start()
  biais = timer.get_biais()  # float -1.0 -> +1.0

Variables .env :
  REMORA_INTERVAL_SEC=300
  REMORA_API_URL=http://localhost:8000
"""

import os
import sys
import json
import time
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

_BASE = Path(__file__).parent
load_dotenv(dotenv_path=_BASE / ".env")
load_dotenv(dotenv_path=_BASE.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TIMER-REMORAS] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("timer_consensus")

INTERVAL_DEFAUT = int(os.environ.get("REMORA_INTERVAL_SEC", 300))
API_URL_DEFAUT  = os.environ.get("REMORA_API_URL", "http://localhost:8000")
FICHIER_ESPRIT  = _BASE.parent / "cerveau" / "esprit_etat.json"
FICHIER_RESULT  = _BASE.parent / "cerveau" / "remora_consensus.json"


def _lire_contexte() -> dict:
    """
    Extrait le contexte public depuis esprit_etat.json.
    Seules les donnees lisibles par les IA externes sont transmises.
    Les donnees internes (magic, slots, feedbacks) restent privees.
    """
    ctx = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "magic_99999_actif": False,
        "deadline_sec":    25.0,
    }
    if not FICHIER_ESPRIT.exists():
        log.warning("esprit_etat.json introuvable — contexte minimal")
        return ctx
    try:
        with open(FICHIER_ESPRIT, "r", encoding="utf-8") as f:
            etat = json.load(f)

        xau = etat.get("symboles", {}).get("XAUUSD", {})
        if xau:
            if "prix"      in xau: ctx["prix"]      = float(xau["prix"])
            if "regime"    in xau: ctx["regime"]    = str(xau["regime"])
            if "pct_range" in xau: ctx["pct_range"] = float(xau["pct_range"])

        ctx["biais_short"]       = float(etat.get("biais_short", 0.0))
        ctx["biais_long"]        = float(etat.get("biais_long",  0.0))
        ctx["drawdown_flottant"] = float(etat.get("drawdown_flottant", 0))
        ctx["nb_positions"]      = int(etat.get("nb_positions_total", 0))
        ctx["magic_99999_actif"] = bool(etat.get("magic_99999_actif", False))
        ctx["deadline_sec"]      = float(etat.get("remora_deadline_sec", 25.0))
    except Exception as e:
        log.error(f"Erreur lecture esprit_etat.json : {e}")
    return ctx


class TimerConsensus:
    """
    Appelle POST /vote a intervalle regulier.
    Expose get_biais() pour integration directe dans ESPRIT.
    """

    def __init__(self, interval_sec: int = INTERVAL_DEFAUT, url: str = API_URL_DEFAUT):
        self.interval_sec = interval_sec
        self.url_vote     = url.rstrip("/") + "/vote"
        self._stop        = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nb_cycles   = 0
        self._dernier: Optional[dict] = None

    def start(self, daemon: bool = True) -> None:
        if self._thread and self._thread.is_alive():
            log.warning("Timer deja actif.")
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._boucle, name="timer-remoras", daemon=daemon
        )
        self._thread.start()
        log.info(f"Timer remoras demarre | intervalle={self.interval_sec}s | {self.url_vote}")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=15)
        log.info(f"Timer arrete apres {self._nb_cycles} cycles.")

    def forcer_cycle(self) -> Optional[dict]:
        """Cycle immediat bloquant. Utile pour tests / ESPRIT."""
        return self._executer_cycle()

    def get_consensus(self) -> Optional[dict]:
        if self._dernier:
            return self._dernier
        try:
            if FICHIER_RESULT.exists():
                with open(FICHIER_RESULT, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def get_biais(self) -> float:
        """
        Biais directif des remoras en float.
          +1.0 = fort consensus LONG
          -1.0 = fort consensus SHORT
           0.0 = NEUTRE / veto / pas de donnees

        Pret a etre mixe dans les curseurs de l'ESPRIT.
        """
        c = self.get_consensus()
        if not c or c.get("veto_99999"):
            return 0.0
        score = float(c.get("score", 0.0))
        force = c.get("force", "NEUTRE")
        mult  = {"FORT": 1.0, "MOYEN": 0.7, "FAIBLE": 0.4}.get(force, 0.0)
        return round(score * mult, 3)

    def _boucle(self) -> None:
        self._executer_cycle()
        while not self._stop.wait(self.interval_sec):
            self._executer_cycle()
        log.info("Boucle terminee.")

    def _executer_cycle(self) -> Optional[dict]:
        self._nb_cycles += 1
        heure = datetime.now().strftime("%H:%M:%S")
        log.info(f"-- Cycle #{self._nb_cycles} [{heure}] --")
        try:
            ctx      = _lire_contexte()
            deadline = ctx.get("deadline_sec", 25.0)
            resp     = requests.post(self.url_vote, json=ctx, timeout=deadline + 5)
            resp.raise_for_status()
            c = resp.json()
            self._dernier = c

            if c.get("veto_99999"):
                log.warning("VETO Magic 99999 — conseil annule par le bus")
            else:
                biais = self.get_biais()
                orient = (
                    f"FAVORISE LONG ({biais:+.2f})"  if biais >  0.1 else
                    f"FAVORISE SHORT ({biais:+.2f})" if biais < -0.1 else
                    "NEUTRE"
                )
                log.info(
                    f"VERDICT -> {c.get('vote_final')} | "
                    f"score={c.get('score', 0):+.3f} | "
                    f"force={c.get('force')} | {c.get('nb_valides')}/{c.get('nb_valides')} | {orient}"
                )
            return c

        except requests.exceptions.ConnectionError:
            log.error(f"remora_bus non joignable. Lance : uvicorn api.remora_bus:app --port 8000")
        except requests.exceptions.Timeout:
            log.error("Deadline depassee — cycle ignore.")
        except Exception as e:
            log.error(f"Erreur cycle #{self._nb_cycles} : {e}", exc_info=True)
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Timer Conseil des Remoras — Blade Runner V6")
    parser.add_argument("--interval", type=int, default=INTERVAL_DEFAUT, metavar="SEC")
    parser.add_argument("--url",      type=str, default=API_URL_DEFAUT)
    parser.add_argument("--once",     action="store_true", help="Un seul cycle puis quitter")
    args = parser.parse_args()

    timer = TimerConsensus(interval_sec=args.interval, url=args.url)

    if args.once:
        result = timer.forcer_cycle()
        print(json.dumps(result or {"erreur": "aucun resultat"}, ensure_ascii=False, indent=2))
        sys.exit(0)

    log.info("Ctrl+C pour arreter.")
    timer.start(daemon=False)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        timer.stop()
        sys.exit(0)
