"""Read-only projections for recorded UI evidence; never dispatch inference."""

import json

from counterfact.evidence import case_observations
from counterfact.reference import answer, question_text
from counterfact.score import score_response
from counterfact.service import ServiceError


def runs_list(service):
    with service.store.connect() as connection:
        rows = connection.execute("SELECT * FROM run ORDER BY created_at DESC LIMIT 100").fetchall()
    result = []
    for row in rows:
        budget = json.loads(row["budget_json"])
        result.append(
            {
                "id": row["id"],
                "status": row["status"],
                "fresh": bool(row["fresh"]),
                "suite_id": row["suite_id"],
                "model_profile": row["profile"],
                "created_at": row["created_at"],
                "budget": budget,
                "live_count": budget.get("live_count"),
                "cached_count": budget.get("cached_count"),
            }
        )
    return result


def case_detail(service, run_id, case_id):
    family = next((f for f in service.suite.families if f.family_id == case_id), None)
    if family is None:
        raise ServiceError(404, "case_not_found", "Unknown case")
    with service.store.connect() as connection:
        run = connection.execute("SELECT * FROM run WHERE id=?", (run_id,)).fetchone()
        if run is None:
            raise ServiceError(404, "run_not_found", "Unknown run")
        records = connection.execute(
            "SELECT cr.*,r.created_at,r.finished_at,r.budget_json "
            "FROM case_result cr JOIN run r ON r.id=cr.run_id WHERE cr.run_id=? AND cr.case_id=?",
            (run_id, case_id),
        ).fetchall()
        if not records:
            raise ServiceError(404, "case_not_found", "Case has no observations in this run yet")
        pairs = connection.execute(
            "SELECT manifest FROM pairs WHERE spec=?",
            (json.dumps(family.spec.model_dump(), sort_keys=True, separators=(",", ":")),),
        ).fetchall()
        members = json.loads(pairs[0][0])["members"] if pairs else []
        sides = {}
        for which, chart in (
            ("original", family.spec.chart),
            ("transformed", family.spec.transformed()),
        ):
            record = next((r for r in records if r["side"] == which), None)
            observations = []
            if record:
                for o in case_observations(connection, record):
                    score = score_response(
                        o["raw_text"],
                        family.spec.question.type,
                        answer(chart, family.spec.question),
                    )
                    observations.append(
                        {
                            "id": o["id"],
                            "slot": o["slot"],
                            "raw_text": o["raw_text"],
                            "status": score["status"],
                            "model": o["model"],
                            "provider": o["provider"],
                            "acquired_at": o["acquired_at"],
                            "cached": o["acquired_at"] < run["created_at"],
                        }
                    )
            correct = sum(o["status"] == "correct" for o in observations)
            incorrect = sum(o["status"] == "incorrect" for o in observations)
            verdict = (
                "inconclusive"
                if len(observations) != 3
                else ("passes" if correct >= 2 else ("fails" if incorrect >= 2 else "mixed"))
            )
            member = next((m for m in members if m["variant"] == which), {})
            sides[which] = {
                "image_hash": member.get("image_hash"),
                "question": question_text(chart, family.spec.question),
                "expected_answer": answer(chart, family.spec.question),
                "observations": sorted(observations, key=lambda o: o["slot"]),
                "verdict": verdict,
                "correct_count": correct,
                "cached_count": record["cached_count"] if record else 0,
                "live_count": record["live_count"] if record else 0,
            }
        saved = connection.execute(
            "SELECT minimization_id,case_json FROM regression_case ORDER BY created_at DESC"
        ).fetchall()
    from counterfact.runner import pair_status

    eligible = run["status"] not in ("running", "queued") and any(
        s["verdict"] == "fails" for s in sides.values()
    )
    mid = next(
        (
            r["minimization_id"]
            for r in saved
            if json.loads(r["case_json"]).get("original_case") == family.spec.model_dump()
        ),
        None,
    )
    return {
        "case_id": case_id,
        "run_id": run_id,
        "question_type": family.spec.question.type,
        **sides,
        "pair_status": pair_status(sides["original"]["verdict"], sides["transformed"]["verdict"]),
        "eligible": eligible,
        "eligibility_reason": None
        if eligible
        else (
            "Wait for the run to finish"
            if run["status"] in ("running", "queued")
            else "No complete side has at least two incorrect observations"
        ),
        "minimization_id": mid,
    }


def cases_list(service, run_id):
    if service.store.get_run(run_id) is None:
        raise ServiceError(404, "run_not_found", "Unknown run")
    with service.store.connect() as connection:
        ids = connection.execute(
            "SELECT DISTINCT case_id FROM case_result WHERE run_id=? ORDER BY rowid", (run_id,)
        ).fetchall()
    return [case_detail(service, run_id, r[0]) for r in ids]
