import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Criterion:
    scope: str
    category: str
    criterion_id: str
    criterion: str
    weight_points: float
    evidence_required: str
    hard_gate: bool


_LEVEL_FACTORS: dict[str, float] = {
    "exceeds": 1.0,
    "meets": 0.75,
    "approaching": 0.5,
    "not_met": 0.0,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_criteria(csv_path: Path) -> list[Criterion]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        criteria: list[Criterion] = []
        for row in reader:
            criteria.append(
                Criterion(
                    scope=row["scope"].strip(),
                    category=row["category"].strip(),
                    criterion_id=row["criterion_id"].strip(),
                    criterion=row["criterion"].strip(),
                    weight_points=float(row["weight_points"]),
                    evidence_required=row.get("evidence_required", "").strip(),
                    hard_gate=row.get("hard_gate", "false").strip().lower() == "true",
                )
            )
        return criteria


def _prompt_level(criterion: Criterion) -> str:
    while True:
        raw = (
            input(
                f"{criterion.criterion_id} ({criterion.weight_points:g} pts) - "
                f"{criterion.criterion}\n"
                f"Evidencia: {criterion.evidence_required}\n"
                "Nivel [exceeds/meets/approaching/not_met]: "
            )
            .strip()
            .lower()
        )
        if raw in _LEVEL_FACTORS:
            return raw
        print("Nivel inválido. Usa: exceeds | meets | approaching | not_met")


def _prompt_pb_score(pb_scope: str) -> float:
    while True:
        raw = input(f"Puntaje {pb_scope} (0-100): ").strip()
        try:
            val = float(raw)
        except ValueError:
            print("Número inválido.")
            continue
        if 0 <= val <= 100:
            return val
        print("Debe estar entre 0 y 100")


def _pb_level(pb_scope: str, score: float) -> str:
    if pb_scope in {"PB8", "PB16"}:
        if score >= 85:
            return "exceeds"
        if score >= 75:
            return "meets"
        if score >= 65:
            return "approaching"
        return "not_met"
    if pb_scope == "PB23":
        if score >= 90:
            return "exceeds"
        if score >= 80:
            return "meets"
        if score >= 70:
            return "approaching"
        return "not_met"
    if score >= 85:
        return "exceeds"
    if score >= 75:
        return "meets"
    if score >= 65:
        return "approaching"
    return "not_met"


def run_full(criteria: list[Criterion]) -> int:
    selected = [c for c in criteria if c.scope == "GLOBAL"]
    selected = [c for c in selected if c.weight_points > 0]

    total = 0.0
    print("\n== Evaluación FULL (GLOBAL) ==\n")
    for c in selected:
        level = _prompt_level(c)
        total += c.weight_points * _LEVEL_FACTORS[level]

    pb23 = _prompt_pb_score("PB23")
    pb23_ok = pb23 >= 80

    print("\n== Resultado ==")
    print(f"TOTAL: {total:.1f}/100")
    print(f"PB-23: {pb23:.1f}/100")
    if pb23_ok:
        print("ESTADO: Listo para admisión (gate PB-23 OK)")
        return 0

    print("ESTADO: Aún no listo (PB-23 < 80)")
    return 2


def run_module(criteria: list[Criterion], scope: str) -> int:
    selected = [c for c in criteria if c.scope == scope and c.weight_points > 0]
    if not selected:
        print(f"No hay criterios para scope={scope}")
        return 2

    total_weight = sum(c.weight_points for c in selected)
    total = 0.0

    print(f"\n== Evaluación MÓDULO ({scope}) ==\n")
    for c in selected:
        level = _prompt_level(c)
        total += c.weight_points * _LEVEL_FACTORS[level]

    print("\n== Resultado ==")
    print(f"TOTAL: {total:.1f}/{total_weight:.1f}")
    return 0


def run_pb(criteria: list[Criterion], scope: str) -> int:
    selected = [c for c in criteria if c.scope == scope]
    if not selected:
        print(f"No hay criterios para scope={scope}")
        return 2

    c = selected[0]
    score = _prompt_pb_score(scope)
    level = _pb_level(scope, score)
    points = c.weight_points * _LEVEL_FACTORS[level]

    print("\n== Resultado ==")
    print(f"Nivel: {level}")
    print(f"Puntos: {points:.1f}/{c.weight_points:.1f}")

    if scope == "PB23" and score < 80:
        print("GATE: PB-23 < 80 => Aún no listo")
        return 2

    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template",
        choices={"full", "module", "pb"},
        default="full",
        help="Tipo de scoring a ejecutar",
    )
    parser.add_argument(
        "--scope",
        default="",
        help="Scope del módulo o PB (ej: M05, PB8, PB16, PB23). Requerido en template=module/pb",
    )
    args = parser.parse_args()

    csv_path = _repo_root() / "rubrica.csv"
    criteria = load_criteria(csv_path)

    if args.template == "full":
        return run_full(criteria)

    if not args.scope:
        print("Falta --scope")
        return 2

    if args.template == "module":
        return run_module(criteria, args.scope)

    return run_pb(criteria, args.scope)


if __name__ == "__main__":
    raise SystemExit(main())
