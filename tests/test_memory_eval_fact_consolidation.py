import json

import pytest

from harness.memory_eval.fact_consolidation import load_normalized_jsonl


def write_rows(path, rows):
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def valid_rows():
    return [
        {
            "type": "record",
            "record_id": "r1",
            "entity": "Ada",
            "relation": "employer",
            "value": "A",
            "content": "FACT[Ada|employer]=A",
            "serial": 1,
        },
        {
            "type": "question",
            "query_id": "q1",
            "entity": "Ada",
            "relation": "employer",
            "question": "What is FACT[Ada|employer]?",
            "answer": "A",
        },
    ]


def test_loader_builds_records_and_questions_with_shared_scope(tmp_path):
    path = tmp_path / "normalized.jsonl"
    rows = valid_rows()
    rows[0]["source_id"] = "mab-fact-sh"
    write_rows(path, rows)

    corpus = load_normalized_jsonl(path)

    assert len(corpus.records) == len(corpus.questions) == 1
    assert corpus.records[0].scope == corpus.questions[0].scope
    assert corpus.records[0].scope == "Ada\x1femployer"
    assert corpus.records[0].event_time is None
    assert corpus.records[0].source_id == "mab-fact-sh"


def test_loader_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "duplicate.jsonl"
    rows = valid_rows()
    rows.insert(1, dict(rows[0]))
    write_rows(path, rows)

    with pytest.raises(ValueError, match="duplicate record_id"):
        load_normalized_jsonl(path)


def test_loader_rejects_invalid_serial_and_missing_answer(tmp_path):
    invalid_serial = tmp_path / "serial.jsonl"
    rows = valid_rows()
    rows[0]["serial"] = -1
    write_rows(invalid_serial, rows)
    with pytest.raises(ValueError, match="line 1"):
        load_normalized_jsonl(invalid_serial)

    missing_answer = tmp_path / "answer.jsonl"
    rows = valid_rows()
    del rows[1]["answer"]
    write_rows(missing_answer, rows)
    with pytest.raises(ValueError, match="line 2.*answer"):
        load_normalized_jsonl(missing_answer)


def test_loader_rejects_unknown_types_and_fields(tmp_path):
    unknown = tmp_path / "unknown.jsonl"
    rows = valid_rows()
    rows[0]["type"] = "chunk"
    write_rows(unknown, rows)
    with pytest.raises(ValueError, match="unknown row type"):
        load_normalized_jsonl(unknown)

    extra = tmp_path / "extra.jsonl"
    rows = valid_rows()
    rows[0]["surprise"] = True
    write_rows(extra, rows)
    with pytest.raises(ValueError, match="unexpected fields"):
        load_normalized_jsonl(extra)
