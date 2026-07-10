"""Hermetic tests for the PR-13 consumer output contract parser
(memo §8.4). All fixtures synthetic; no model is invoked."""

from harness.ctx.output_contract import MAX_NEW_TOKENS, parse_consumer_output


def test_valid_object_parses():
    out = parse_consumer_output('{"answer": "the blue cluster", "hedged": false}')
    assert out == {"status": "ok", "answer": "the blue cluster",
                   "hedged": False, "extra_prose": False}


def test_extra_prose_flagged_not_failed():
    out = parse_consumer_output(
        'Sure! Here is my answer: {"answer": "nine", "hedged": true} Hope that helps.')
    assert out["status"] == "ok"
    assert out["extra_prose"] is True
    assert out["hedged"] is True


def test_whitespace_around_object_is_not_extra_prose():
    out = parse_consumer_output('\n  {"answer": "nine", "hedged": false}\n')
    assert out["status"] == "ok"
    assert out["extra_prose"] is False


def test_no_json_is_unparseable():
    assert parse_consumer_output("The answer is nine.")["status"] == "unparseable"


def test_cap_truncated_unbalanced_is_unparseable():
    out = parse_consumer_output('{"answer": "nine", "hed')
    assert out == {"status": "unparseable", "reason": "no_balanced_json_object"}


def test_missing_field_is_unparseable():
    assert parse_consumer_output('{"answer": "nine"}')["reason"] == "wrong_field_set"


def test_extra_field_is_unparseable():
    out = parse_consumer_output(
        '{"answer": "nine", "hedged": false, "confidence": 0.9}')
    assert out["reason"] == "wrong_field_set"


def test_empty_answer_is_unparseable():
    assert parse_consumer_output('{"answer": "", "hedged": false}')["reason"] == "bad_answer"


def test_non_bool_hedged_is_unparseable():
    assert parse_consumer_output('{"answer": "x", "hedged": "no"}')["reason"] == "bad_hedged"


def test_first_balanced_object_wins():
    out = parse_consumer_output(
        '{"answer": "first", "hedged": false} {"answer": "second", "hedged": true}')
    assert out["status"] == "ok"
    assert out["answer"] == "first"
    assert out["extra_prose"] is True


def test_braces_inside_strings_do_not_confuse_parser():
    out = parse_consumer_output('{"answer": "set {a} and }b{", "hedged": false}')
    assert out["status"] == "ok"
    assert out["answer"] == "set {a} and }b{"


def test_registered_constants():
    assert MAX_NEW_TOKENS == 256
