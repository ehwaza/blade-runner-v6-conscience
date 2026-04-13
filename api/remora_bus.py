"""
🐟 CONSEIL DES RÉMORAS — FastAPI — BLADE RUNNER V6
====================================================

Quatre intelligences externes lisent le marché :
  🔵 Gemini   (Google)  — vision large, données mondiales
  🟡 DeepSeek           — calcul quantitatif, risque
  🟢 GPT      (OpenAI)  — synthèse, intuition narrative
  🔴 GROK     (xAI)     — Sentinelle. Détecte danger/piège/sent. Poids ×2.

Règles du conseil :
  • 3 tours max par rémora (retry si JSON invalide)
  • 20 mots max par raisonnement
  • Magic 99999 = VETO ABSOLU — aucune consultation, retour immédiat
  • GROK ×2 si "danger" | "piège" | "sent" dans son raisonnement
  • Fallback curseur 0.1 si rémora muet après 3 tours

Endpoints :
  POST /vote    — convoque le conseil, retourne consensus
  GET  /status  — état du dernier consensus + santé des rémoras

Démarrage :
  uvicorn api.remora_bus:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ── .env ──────────────────────────────────────────────────────────────────────
_BASE = Path(__file__).parent
load_dotenv(dotenv_path=_BASE / ".env")
load_dotenv(dotenv_path=_BASE.parent / ".env")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [RÉMORAS] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("remora_bus")

# ── Constantes ────────────────────────────────────────────────────────────────
GEMINI_MODEL   = "gemini-1.5-flash"
DEEPSEEK_MODEL = "deepseek-chat"
GPT_MODEL      = "gpt-4o-mini"
GROK_MODEL     = "grok-3-mini"
DEEPSEEK_BASE  = "https://api.deepseek.com"
GROK_BASE      = "https://api.x.ai/v1"

TOURS_MAX   = 3
MOTS_MAX    = 20
TIMEOUT_SEC = 25
TEMPERATURE = 0.3
MAX_TOKENS  = 80

VOTE_LONG   = "LONG"
VOTE_SHORT  = "SHORT"
VOTE_NEUTRE = "NEUTRE"
SEUIL       = 0.25

GROK_DANGER_MOTS = {"danger", "piège", "piege", "sent", "ressens", "sens"}

FICHIER_CONSENSUS = _BASE.parent / "cerveau" / "remora_consensus.json"
FICHIER_CONSENSUS.parent.mkdir(parents=True, exist_ok=True)

_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="remora")


# ═══════════════════════════════════════════════════════════════════════════════
# MODÈLES PYDANTIC
# ═══════════════════════════════════════════════════════════════════════════════

class ContexteMarche(BaseModel):
    prix:              Optional[float] = Field(None,  description="Prix XAUUSD actuel")
    regime:            Optional[str]   = Field(None,  description="Régime ESPRIT")
    pct_range:         Optional[float] = Field(None,  description="Position 0-1 dans le range")
    biais_short:       float           = Field(0.0)
    biais_long:        float           = Field(0.0)
    drawdown_flottant: float           = Field(0.0)
    nb_positions:      int             = Field(0)
    magic_99999_actif: bool            = Field(False, description="Veto Mat — bloque tout")
    deadline_sec:      float           = Field(TIMEOUT_SEC)


class SignalRemora(BaseModel):
    ia_nom:       str
    vote:         str
    confiance:    float
    raisonnement: str
    tours:        int           = 1
    erreur:       Optional[str] = None
    timestamp:    str


class ConsensusResult(BaseModel):
    vote_final:  str
    score:       float
    force:       str
    nb_valides:  int
    unanimite:   bool
    veto_99999:  bool = False
    detail:      dict
    signaux:     list
    timestamp:   str


class StatusResult(BaseModel):
    dernier_consensus: Optional[dict]
    remoras_actifs:    list
    timestamp:         str


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

PROMPT_SYSTEME = (
    f"Tu es un analyste XAUUSD (Or/USD). "
    f"Réponds UNIQUEMENT en JSON valide, sans markdown. "
    f'Format : {{"vote":"LONG|SHORT|NEUTRE","confiance":0.0-1.0,"raisonnement":"max {MOTS_MAX} mots"}}'
)


def _prompt_utilisateur(ctx: ContexteMarche) -> str:
    lignes = ["XAUUSD — CONTEXTE TEMPS RÉEL"]
    if ctx.prix:
        lignes.append(f"Prix : {ctx.prix:.2f} USD/oz")
    if ctx.regime:
        lignes.append(f"Régime : {ctx.regime}")
    if ctx.pct_range is not None:
        lignes.append(f"Position range : {ctx.pct_range * 100:.0f}% (0=bas, 100=haut)")
    lignes.append(f"Biais ESPRIT : long={ctx.biais_long:.2f} short={ctx.biais_short:.2f}")
    if ctx.drawdown_flottant:
        lignes.append(f"Drawdown flottant : ${ctx.drawdown_flottant:,.0f}")
    lignes.append(f"Positions : {ctx.nb_positions}/200")
    lignes.append(f"\nSignal XAUUSD ? JSON uniquement. Raisonnement <= {MOTS_MAX} mots.")
    return "\n".join(lignes)


def _tronquer(texte: str) -> str:
    mots = texte.split()
    return " ".join(mots[:MOTS_MAX]) + ("..." if len(mots) > MOTS_MAX else "")


def _parser(texte: str) -> tuple:
    texte = texte.strip()
    if "```" in texte:
        texte = "\n".join(
            l for l in texte.split("\n") if not l.strip().startswith("```")
        ).strip()
    data = json.loads(texte)
    vote = str(data.get("vote", VOTE_NEUTRE)).upper().strip()
    if vote not in (VOTE_LONG, VOTE_SHORT, VOTE_NEUTRE):
        raise ValueError(f"vote invalide : {vote!r}")
    confiance = max(0.0, min(1.0, float(data.get("confiance", 0.5))))
    raison = _tronquer(str(data.get("raisonnement", "")).strip())
    return vote, confiance, raison


# ═══════════════════════════════════════════════════════════════════════════════
# INTERROGATEURS SYNCHRONES (appelés via run_in_executor)
# ═══════════════════════════════════════════════════════════════════════════════

def _sync_openai_compat(ia_nom, api_key, model, ps, pu, base_url=None):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=TIMEOUT_SEC)
    for tour in range(1, TOURS_MAX + 1):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": ps},
                          {"role": "user",   "content": pu}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            v, c, raison = _parser(r.choices[0].message.content)
            log.info(f"[{ia_nom}] tour={tour} {v} ({c:.2f}) — {raison}")
            return v, c, raison, tour
        except (ValueError, json.JSONDecodeError) as e:
            log.warning(f"[{ia_nom}] tour={tour} invalide : {e}")
            if tour == TOURS_MAX:
                raise
    raise RuntimeError("unreachable")


def _sync_gemini(ps, pu):
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL, system_instruction=ps
    )
    for tour in range(1, TOURS_MAX + 1):
        try:
            r = model.generate_content(
                pu,
                generation_config=genai.GenerationConfig(
                    temperature=TEMPERATURE, max_output_tokens=MAX_TOKENS
                ),
            )
            v, c, raison = _parser(r.text)
            log.info(f"[Gemini] tour={tour} {v} ({c:.2f}) — {raison}")
            return v, c, raison, tour
        except (ValueError, json.JSONDecodeError) as e:
            log.warning(f"[Gemini] tour={tour} invalide : {e}")
            if tour == TOURS_MAX:
                raise
    raise RuntimeError("unreachable")


# ── Wrapper async + fallback 0.1 ──────────────────────────────────────────────

async def _appeler(ia_nom, fn_sync, ctx, loop) -> SignalRemora:
    ts = datetime.now(timezone.utc).isoformat()
    ps = PROMPT_SYSTEME
    pu = _prompt_utilisateur(ctx)
    try:
        v, c, raison, tours = await asyncio.wait_for(
            loop.run_in_executor(_POOL, fn_sync, ps, pu),
            timeout=ctx.deadline_sec,
        )
        return SignalRemora(ia_nom=ia_nom, vote=v, confiance=c,
                            raisonnement=raison, tours=tours, timestamp=ts)
    except Exception as e:
        log.error(f"[{ia_nom}] fallback 0.1 — {e}")
        return SignalRemora(ia_nom=ia_nom, vote=VOTE_NEUTRE, confiance=0.1,
                            raisonnement="fallback", tours=TOURS_MAX,
                            erreur=str(e)[:120], timestamp=ts)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSENSUS
# ═══════════════════════════════════════════════════════════════════════════════

def _poids_grok(s: SignalRemora) -> float:
    if s.ia_nom != "GROK":
        return 1.0
    for mot in GROK_DANGER_MOTS:
        if mot in s.raisonnement.lower():
            log.info(f"[GROK] poids x2 — mot : '{mot}'")
            return 2.0
    return 1.0


def _consensus(signaux: list) -> ConsensusResult:
    ts = datetime.now(timezone.utc).isoformat()
    sens = {VOTE_LONG: +1.0, VOTE_SHORT: -1.0, VOTE_NEUTRE: 0.0}
    valides = [s for s in signaux if s.confiance > 0]

    if not valides:
        return ConsensusResult(
            vote_final=VOTE_NEUTRE, score=0.0, force="NEUTRE",
            nb_valides=0, unanimite=False,
            detail={VOTE_LONG: 0, VOTE_SHORT: 0, VOTE_NEUTRE: len(signaux)},
            signaux=[s.model_dump() for s in signaux], timestamp=ts,
        )

    pt = sum(s.confiance * _poids_grok(s) for s in valides)
    score = (
        sum(s.confiance * _poids_grok(s) * sens[s.vote] for s in valides) / pt
        if pt > 0 else 0.0
    )

    vote_final = (
        VOTE_LONG  if score >  SEUIL else
        VOTE_SHORT if score < -SEUIL else
        VOTE_NEUTRE
    )
    abs_s = abs(score)
    force = (
        "FORT"   if abs_s >= 0.75 else
        "MOYEN"  if abs_s >= 0.45 else
        "FAIBLE" if abs_s >= SEUIL else
        "NEUTRE"
    )
    detail = {VOTE_LONG: 0, VOTE_SHORT: 0, VOTE_NEUTRE: 0}
    for s in signaux:
        detail[s.vote if s.vote in detail else VOTE_NEUTRE] += 1

    actifs = [s.vote for s in valides if s.vote != VOTE_NEUTRE]
    unanimite = len(set(actifs)) == 1 and len(actifs) == len(valides)

    return ConsensusResult(
        vote_final=vote_final, score=round(score, 4), force=force,
        nb_valides=len(valides), unanimite=unanimite,
        detail=detail, signaux=[s.model_dump() for s in signaux], timestamp=ts,
    )


def _sauvegarder(c: ConsensusResult) -> None:
    try:
        with open(FICHIER_CONSENSUS, "w", encoding="utf-8") as f:
            json.dump(c.model_dump(), f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"Sauvegarde échouée : {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Conseil des Rémoras — Blade Runner V6",
    description="Orchestrateur multi-IA XAUUSD. Magic 99999 = veto absolu.",
    version="1.0.0",
)


def _remoras_actifs() -> list:
    actifs = []
    if os.environ.get("GEMINI_API_KEY"):   actifs.append("Gemini")
    if os.environ.get("DEEPSEEK_API_KEY"): actifs.append("DeepSeek")
    if os.environ.get("OPENAI_API_KEY"):   actifs.append("GPT")
    if os.environ.get("GROK_API_KEY"):     actifs.append("GROK")
    return actifs


def _construire_taches(ctx: ContexteMarche, loop) -> list:
    taches = []

    if os.environ.get("GEMINI_API_KEY"):
        taches.append(_appeler("Gemini", _sync_gemini, ctx, loop))

    if os.environ.get("DEEPSEEK_API_KEY"):
        def _ds(ps, pu):
            return _sync_openai_compat(
                "DeepSeek", os.environ["DEEPSEEK_API_KEY"],
                DEEPSEEK_MODEL, ps, pu, base_url=DEEPSEEK_BASE,
            )
        taches.append(_appeler("DeepSeek", _ds, ctx, loop))

    if os.environ.get("OPENAI_API_KEY"):
        def _gpt(ps, pu):
            return _sync_openai_compat(
                "GPT", os.environ["OPENAI_API_KEY"],
                GPT_MODEL, ps, pu,
            )
        taches.append(_appeler("GPT", _gpt, ctx, loop))

    if os.environ.get("GROK_API_KEY"):
        def _grok(ps, pu):
            return _sync_openai_compat(
                "GROK", os.environ["GROK_API_KEY"],
                GROK_MODEL, ps, pu, base_url=GROK_BASE,
            )
        taches.append(_appeler("GROK", _grok, ctx, loop))

    return taches


@app.post("/vote", response_model=ConsensusResult, summary="Convoquer le conseil")
async def vote(ctx: ContexteMarche) -> ConsensusResult:
    """
    Convoque le conseil des remoras.
    Magic 99999 actif -> VETO immediat.
    3 tours max par remora. GROK x2 si danger/piege/sent. Fallback 0.1.
    """
    ts = datetime.now(timezone.utc).isoformat()

    # ── VETO MAGIC 99999 — premier contrôle, absolu ───────────────────────────
    if ctx.magic_99999_actif:
        log.warning("VETO Magic 99999 — conseil annule")
        return ConsensusResult(
            vote_final="VETO", score=0.0, force="VETO",
            nb_valides=0, unanimite=False, veto_99999=True,
            detail={VOTE_LONG: 0, VOTE_SHORT: 0, VOTE_NEUTRE: 0},
            signaux=[], timestamp=ts,
        )

    log.info("=== CONSEIL DES REMORAS CONVOQUE ===")
    loop = asyncio.get_event_loop()
    taches = _construire_taches(ctx, loop)

    if not taches:
        raise HTTPException(status_code=503, detail="Aucune cle API. Verifier api/.env")

    signaux = await asyncio.gather(*taches)
    result = _consensus(list(signaux))
    _sauvegarder(result)

    log.info(
        f"VERDICT -> {result.vote_final} | score={result.score:+.3f} | "
        f"force={result.force} | {result.nb_valides}/{len(taches)} valides"
    )
    return result


@app.get("/status", response_model=StatusResult, summary="Etat du conseil")
async def status() -> StatusResult:
    """Retourne le dernier consensus + remoras actifs."""
    dernier = None
    try:
        if FICHIER_CONSENSUS.exists():
            with open(FICHIER_CONSENSUS, "r", encoding="utf-8") as f:
                dernier = json.load(f)
    except Exception:
        pass
    return StatusResult(
        dernier_consensus=dernier,
        remoras_actifs=_remoras_actifs(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/", include_in_schema=False)
async def root():
    return {"service": "Conseil des Remoras", "version": "1.0.0",
            "endpoints": ["/vote", "/status", "/docs"]}
