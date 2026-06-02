"""Research memory: registry of truths/hypotheses with auto-generated docs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REGISTRY_PATH = ROOT / "research_phases" / "memory_registry.json"

Category = Literal["CONFIRMED", "REJECTED", "PARTIAL EVIDENCE", "UNTESTED"]
HypothesisStatus = Literal["CONFIRMED", "REJECTED", "PARTIAL EVIDENCE", "UNTESTED", "IN PROGRESS"]


def _default_registry() -> dict[str, Any]:
    return {
        "version": 1,
        "updated": None,
        "findings": [],
        "hypotheses": [],
        "roadmap": {
            "current_frontier": "",
            "highest_value_unknowns": [],
            "recommended_experiments": [],
        },
    }


def load_registry() -> dict[str, Any]:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return _default_registry()


def save_registry(registry: dict[str, Any]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    sync_docs(registry)


def record_finding(
    category: Category,
    statement: str,
    confidence: str,
    evidence: str,
    source: str,
    *,
    replace_if_statement_contains: str | None = None,
) -> None:
    registry = load_registry()
    findings = registry["findings"]
    if replace_if_statement_contains:
        findings = [f for f in findings if replace_if_statement_contains not in f.get("statement", "")]
    entry = {
        "category": category,
        "statement": statement,
        "confidence": confidence,
        "evidence": evidence,
        "source": source,
        "recorded": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    findings.append(entry)
    registry["findings"] = findings
    save_registry(registry)


def update_hypothesis(
    hid: str,
    status: HypothesisStatus,
    summary: str,
    evidence: str,
    source: str,
) -> None:
    registry = load_registry()
    hypotheses = registry["hypotheses"]
    found = False
    for h in hypotheses:
        if h["id"] == hid:
            h["status"] = status
            h["summary"] = summary
            h["evidence"] = evidence
            h["source"] = source
            h["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            found = True
            break
    if not found:
        hypotheses.append(
            {
                "id": hid,
                "status": status,
                "summary": summary,
                "evidence": evidence,
                "source": source,
                "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
    registry["hypotheses"] = sorted(hypotheses, key=lambda x: x["id"])
    save_registry(registry)


def update_roadmap(
    current_frontier: str,
    highest_value_unknowns: list[str],
    recommended_experiments: list[str],
) -> None:
    registry = load_registry()
    registry["roadmap"] = {
        "current_frontier": current_frontier,
        "highest_value_unknowns": highest_value_unknowns,
        "recommended_experiments": recommended_experiments,
    }
    save_registry(registry)


def _format_finding(f: dict[str, Any]) -> str:
    return (
        f"### {f['statement']}\n\n"
        f"- **Confidence:** {f['confidence']}\n"
        f"- **Evidence:** {f['evidence']}\n"
        f"- **Source:** {f['source']}\n"
    )


def sync_docs(registry: dict[str, Any] | None = None) -> None:
    registry = registry or load_registry()
    DOCS.mkdir(parents=True, exist_ok=True)

    by_cat: dict[str, list[dict]] = {
        "CONFIRMED": [],
        "REJECTED": [],
        "PARTIAL EVIDENCE": [],
        "UNTESTED": [],
    }
    for f in registry["findings"]:
        cat = f.get("category", "UNTESTED")
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(f)

    truth_lines = [
        "# Current Truth — Perry Research Memory",
        "",
        f"**Last updated:** {registry.get('updated', 'never')}",
        "",
        "Auto-maintained from `research_phases/memory_registry.json`. "
        "Do not edit by hand; use `Model/research_memory.py` or frontier pipeline.",
        "",
    ]
    for cat in ["CONFIRMED", "REJECTED", "PARTIAL EVIDENCE", "UNTESTED"]:
        truth_lines.append(f"## {cat}\n")
        items = by_cat.get(cat, [])
        if not items:
            truth_lines.append("_None recorded._\n")
        else:
            for f in items:
                truth_lines.append(_format_finding(f))
                truth_lines.append("")

    (DOCS / "current_truth.md").write_text("\n".join(truth_lines), encoding="utf-8")

    hyp_lines = [
        "# Hypotheses — Perry",
        "",
        f"**Last updated:** {registry.get('updated', 'never')}",
        "",
        "| ID | Status | Summary | Source |",
        "| --- | --- | --- | --- |",
    ]
    for h in registry.get("hypotheses", []):
        summary = h.get("summary", "").replace("|", "/")[:120]
        hyp_lines.append(f"| {h['id']} | {h['status']} | {summary} | {h.get('source', '')} |")
    hyp_lines.append("\n## Detail\n")
    for h in registry.get("hypotheses", []):
        hyp_lines.append(f"### {h['id']} — {h['status']}\n")
        hyp_lines.append(f"{h.get('summary', '')}\n")
        hyp_lines.append(f"- **Evidence:** {h.get('evidence', '')}\n")
        hyp_lines.append(f"- **Updated:** {h.get('updated', '')}\n")

    (DOCS / "hypotheses.md").write_text("\n".join(hyp_lines), encoding="utf-8")

    rm = registry.get("roadmap", {})
    road_lines = [
        "# Research Roadmap — Perry",
        "",
        f"**Last updated:** {registry.get('updated', 'never')}",
        "",
        "## Current frontier",
        "",
        rm.get("current_frontier", "_Not set_"),
        "",
        "## Highest-value unknowns",
        "",
    ]
    for item in rm.get("highest_value_unknowns", []):
        road_lines.append(f"- {item}")
    road_lines.extend(["", "## Recommended next experiments", ""])
    for item in rm.get("recommended_experiments", []):
        road_lines.append(f"- {item}")
    road_lines.append("")

    (DOCS / "research_roadmap.md").write_text("\n".join(road_lines), encoding="utf-8")


def bootstrap_from_reports() -> None:
    """Seed registry from prior research if empty."""
    registry = load_registry()
    if registry["findings"]:
        sync_docs(registry)
        return

    seeds: list[tuple[Category, str, str, str, str]] = [
        ("CONFIRMED", "Volatility-expansion (move) signal exists on 15m crypto OHLCV.", "High", "Macro F1 ~0.60, bootstrap CI above chance", "research_report.md"),
        ("CONFIRMED", "Move signal survives on BTC, ETH, SOL, BNB, XRP individually.", "High", "5/5 assets pass falsification thresholds", "falsification_report.md"),
        ("CONFIRMED", "Best target is move/no-move at 0.5% over 12 candles.", "High", "Dominates direction/multiclass in target screen", "research_report.md"),
        ("CONFIRMED", "ATR, range, and return-variance features dominate predictors.", "High", "CatBoost importance + MI analysis", "information_theory_report.md"),
        ("CONFIRMED", "Information bottleneck exceeds model bottleneck on OHLCV.", "High", "Optuna plateau; MI flat; direction fails with same model class", "grand_research_report.md"),
        ("REJECTED", "TSFresh materially improves move detection.", "High", "Fold-local TSFresh below compact causal set", "research_report.md"),
        ("REJECTED", "Additional CatBoost tuning yields meaningful gains.", "High", "Optuna +0.00001 macro F1", "research_log.md"),
        ("REJECTED", "Current directional strategy is profitable after 5 bps.", "High", "Mean net −5.19 bps", "research_report.md"),
        ("REJECTED", "Current hierarchical move→direction strategy is profitable.", "High", "Mean net −1.63 bps; Phase 5 all thresholds negative", "conditional_direction_report.md"),
        ("REJECTED", "Stable market regime labels persist over calendar time.", "High", "KMeans ARI ≈ 0 early vs late", "market_state_report.md"),
        ("PARTIAL EVIDENCE", "Universal pooled model transfers to all assets including BTC.", "Medium", "BTC LOO fails (0.545) without BTC in train", "cross_asset_report.md"),
        ("PARTIAL EVIDENCE", "State transition structure contains tradeable information.", "Low", "Descriptive move rates by state; ARI unstable", "state_transition_report.md"),
        ("UNTESTED", "Funding rates improve move or direction prediction.", "N/A", "Not yet integrated in walk-forward", "data_expansion_report.md"),
        ("UNTESTED", "Open interest improves move or direction prediction.", "N/A", "Not yet integrated", "data_expansion_report.md"),
        ("UNTESTED", "Liquidations improve prediction.", "N/A", "Not yet integrated", "data_expansion_report.md"),
        ("UNTESTED", "Order book imbalance improves prediction.", "N/A", "No historical book data", "data_expansion_report.md"),
    ]
    for cat, stmt, conf, ev, src in seeds:
        registry["findings"].append(
            {
                "category": cat,
                "statement": stmt,
                "confidence": conf,
                "evidence": ev,
                "source": src,
                "recorded": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )

    hypothesis_seeds = [
        ("H001", "CONFIRMED", "Move/vol-expansion is predictable from causal OHLCV.", "research_report.md"),
        ("H002", "REJECTED", "TSFresh adds OOS move signal.", "research_report.md"),
        ("H003", "REJECTED", "Direction is profitable after fees with current features.", "conditional_direction_report.md"),
        ("H004", "PARTIAL EVIDENCE", "BTC leads altcoin move/vol (cross-asset flow).", "grand_research_report.md"),
        ("H005", "REJECTED", "Discrete regime labels are stable over time.", "market_state_report.md"),
        ("H006", "UNTESTED", "Funding rate extremes predict vol expansion.", "data_expansion_report.md"),
        ("H007", "UNTESTED", "OI changes predict move magnitude.", "data_expansion_report.md"),
        ("H008", "UNTESTED", "Cross-asset lead-lag features improve BTC move OOS.", "research_roadmap.md"),
        ("H009", "UNTESTED", "NSE/macro internals transfer to crypto.", "data_expansion_report.md"),
        ("H010", "UNTESTED", "Options IV/skew improves move timing.", "data_expansion_report.md"),
    ]
    for hid, status, summary, source in hypothesis_seeds:
        registry["hypotheses"].append(
            {"id": hid, "status": status, "summary": summary, "evidence": "", "source": source, "updated": None}
        )

    registry["roadmap"] = {
        "current_frontier": "Information Perry cannot see: derivatives positioning, cross-asset lead-lag, microstructure.",
        "highest_value_unknowns": [
            "Does BTC vol/move lead alt vol at 15m–3h horizons?",
            "Do funding/OI add incremental OOS lift over OHLCV?",
            "What causal cross-asset features unlock direction?",
        ],
        "recommended_experiments": [
            "Phase 1 cross-asset flow matrix + causal feature ablation",
            "Phase 2 Binance funding + OI integration",
            "Phase 3 enhanced walk-forward vs frozen baseline",
        ],
    }
    save_registry(registry)


if __name__ == "__main__":
    bootstrap_from_reports()
    print(f"Synced docs under {DOCS}")
