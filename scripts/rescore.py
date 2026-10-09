"""Derive current scores from stored raw evidence without network calls or row changes."""

import json
from collections import Counter, defaultdict

from counterfact.config import Settings
from counterfact.reference import answer
from counterfact.score import SCORER_VERSION, score_response
from counterfact.store import Store
from counterfact.suites import HardSuite


def rescore_stored(store: Store, suite: HardSuite) -> dict:
    families = {family.family_id: family for family in suite.families}
    derived = {}
    groups = defaultdict(list)
    with store.connect() as connection:
        rows = connection.execute(
            "SELECT cr.case_id,cr.side,o.id,o.slot,o.raw_text,o.group_key "
            "FROM observation o JOIN case_result cr ON cr.group_key=o.group_key "
            "ORDER BY o.acquired_at,o.id"
        ).fetchall()
    for row in rows:
        if row["id"] in derived:
            continue
        spec = families[row["case_id"]].spec
        chart = spec.chart if row["side"] == "original" else spec.transformed()
        score = score_response(row["raw_text"], spec.question.type, answer(chart, spec.question))
        derived[row["id"]] = {
            "id": row["id"],
            "case_id": row["case_id"],
            "side": row["side"],
            "slot": row["slot"],
            "scorer_version": SCORER_VERSION,
            **score,
        }
        groups[(row["case_id"], row["side"], row["group_key"])].append(score["status"])
    verdicts = Counter()
    for statuses in groups.values():
        verdict = "inconclusive"
        if len(statuses) == 3:
            verdict = (
                "fails"
                if statuses.count("incorrect") >= 2
                else ("passes" if statuses.count("correct") >= 2 else "mixed")
            )
        verdicts[verdict] += 1
    return {
        "scorer_version": SCORER_VERSION,
        "network_calls": 0,
        "observation_counts": dict(Counter(o["status"] for o in derived.values())),
        "verdict_counts": dict(verdicts),
        "observations": list(derived.values()),
    }


def main() -> int:
    settings = Settings.from_env()
    suite = HardSuite.model_validate_json(
        (settings.data_dir / "suites" / "hard_v1.json").read_text(encoding="utf-8")
    )
    result = rescore_stored(Store(settings.data_dir), suite)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
