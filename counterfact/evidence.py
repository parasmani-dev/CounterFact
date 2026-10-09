"""Resolve stored failures using strict scores of their original raw observations."""

import json

from counterfact.reference import answer
from counterfact.score import score_response
from counterfact.suites import HardSuite


def case_observations(connection, row):
    ids = json.loads(row["budget_json"]).get("observation_ids", {}).get(row["id"])
    if ids is not None:
        observations = [
            connection.execute("SELECT * FROM observation WHERE id=?", (oid,)).fetchone()
            for oid in ids
        ]
        observations = [o for o in observations if o is not None]
    else:
        # Legacy runs lack exact ID links. Reconstruct the slot snapshot as of
        # their completion, never substituting a newer acquisition into history.
        candidates = connection.execute(
            "SELECT * FROM observation WHERE (group_key=? OR group_key LIKE ?) "
            "AND acquired_at<=? ORDER BY acquired_at DESC,id DESC",
            (row["group_key"], row["group_key"] + ":%", row["finished_at"] or row["created_at"]),
        ).fetchall()
        slots = {}
        for observation in candidates:
            slots.setdefault(observation["slot"], observation)
        observations = list(slots.values())
    return observations


def stored_failure(store, suite: HardSuite, *, run_id=None, case_id=None, side=None):
    families = {f.family_id: f for f in suite.families}
    with store.connect() as connection:
        rows = connection.execute(
            "SELECT cr.*,r.created_at,r.finished_at,r.budget_json "
            "FROM case_result cr JOIN run r ON r.id=cr.run_id "
            "ORDER BY r.created_at DESC,cr.case_id,cr.side"
        ).fetchall()
        groups = {}
        for row in rows:
            if run_id and row["run_id"] != run_id:
                continue
            if case_id and row["case_id"] != case_id:
                continue
            family = families.get(row["case_id"])
            if family is None:
                continue
            observations = case_observations(connection, row)
            chart = family.spec.chart if row["side"] == "original" else family.spec.transformed()
            statuses = [
                score_response(
                    o["raw_text"], family.spec.question.type, answer(chart, family.spec.question)
                )["status"]
                for o in observations
            ]
            verdict = "inconclusive"
            if len(statuses) == 3:
                if statuses.count("incorrect") >= 2:
                    verdict = "fails"
                elif statuses.count("correct") >= 2:
                    verdict = "passes"
                else:
                    verdict = "mixed"
            groups.setdefault((row["run_id"], row["case_id"]), {})[row["side"]] = verdict
    singles = []
    for (rid, cid), sides in groups.items():
        spec = families[cid].spec.model_dump()
        if (
            side is None
            and sides.get("original") == "passes"
            and sides.get("transformed") == "fails"
        ):
            return {
                "case": {"spec": spec, "side": "transformed"},
                "predicate": "correct_to_incorrect",
                "run_id": rid,
                "case_id": cid,
            }
        for which in ("original", "transformed"):
            if sides.get(which) == "fails" and (side is None or which == side):
                singles.append(
                    {
                        "case": {"spec": spec, "side": which},
                        "predicate": "wrong_answer",
                        "run_id": rid,
                        "case_id": cid,
                    }
                )
    return singles[0] if singles else None
