"""POST /api/v1/assist/stream: the same work and the same answer as /assist, with real stage events."""

from __future__ import annotations

import json

from tests.unit.test_assist_api import GOAL, client, csv_file, stub  # noqa: F401  (fixtures)

URL = "/api/v1/assist/stream"


def parse(text: str):
    events = []
    for block in text.strip().split("\n\n"):
        lines = [l for l in block.splitlines() if not l.startswith(":")]
        if lines:
            events.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return events


class TestStream:
    def test_a_file_streams_its_real_stages_then_the_same_answer(self, client):
        r = client.post(URL, data={"message": "here is my data", "project_goal": "predict which customers will churn"},
                        files=csv_file())
        assert r.headers["content-type"].startswith("text/event-stream")
        events = parse(r.text)
        kinds = [k for k, _ in events]
        assert kinds[-1] == "result" and "error" not in kinds and "plan" in kinds
        done = {e["id"] for k, e in events if k == "step" and e["state"] == "done"}
        assert {"route", "read", "profile", "targets", "quality", "tasks", "signal", "ideas"} <= done
        order = [e["id"] for k, e in events if k == "step" and e["state"] == "done"]
        assert order.index("read") < order.index("profile") < order.index("targets") < order.index("quality")
        read = next(e for k, e in events if k == "step" and e["id"] == "read" and e["state"] == "done")
        assert "rows" in read["detail"]
        result = events[-1][1]
        assert result["status"] == "COMPLETED" and result["journey"] == "USER_B_DATASET_ANALYSIS"
        plan = next(e for k, e in events if k == "plan")
        assert [s["id"] for s in plan["steps"]][:2] == ["read", "profile"]

    def test_the_streamed_answer_matches_the_plain_endpoint(self, client):
        data = {"message": "here is my data"}
        plain = client.post("/api/v1/assist", data=data, files=csv_file()).json()
        streamed = parse(client.post(URL, data=data, files=csv_file()).text)[-1][1]
        for body in (plain, streamed):
            body.pop("request_id", None)
            body["analysis"].pop("request_id", None)
            for key in ("metadata",):
                body["analysis"]["analysis"].pop(key, None)
        assert streamed["headline"] == plain["headline"] and streamed["journey"] == plain["journey"]
        assert streamed["analysis"]["analysis"]["schema"] == plain["analysis"]["analysis"]["schema"]

    def test_an_ambiguous_message_streams_a_question_and_nothing_else(self, client):
        events = parse(client.post(URL, data={"message": "dataset"}).text)
        assert events[-1][0] == "result" and events[-1][1]["status"] in ("NEEDS_CLARIFICATION", "NEEDS_INPUT")
        assert any(k == "step" and e["id"] == "route" for k, e in events)
        assert not any(k == "plan" for k, _ in events)

    def test_a_refused_file_arrives_as_an_error_event_with_its_status(self, client):
        files = {"file": ("evil.exe", b"MZ....", "application/octet-stream")}
        events = parse(client.post(URL, data={"message": "analyze this"}, files=files).text)
        assert events[-1][0] == "error" and events[-1][1]["status"] == 415

    def test_no_user_text_or_file_content_appears_in_the_stage_events(self, client):
        secret = "ZEBRA-SECRET-PHRASE"
        r = client.post(URL, data={"message": f"analyze this {secret}", "project_goal": f"predict churn {secret}"},
                        files=csv_file())
        stage_lines = [e for k, e in parse(r.text) if k in ("plan", "step")]
        assert secret not in json.dumps(stage_lines) and "customer_id" not in json.dumps(stage_lines)
