import json

import pytest

from rsi_ml.agents import (
    BackendError,
    BudgetExceeded,
    DSPyBackend,
    MockBackend,
    ProviderUnavailable,
)
from rsi_ml.schemas import Limits, Plan, Policy, Recipe, Review, Usage


@pytest.fixture
def recipe():
    return Recipe(
        hidden_sizes=[8],
        activation="relu",
        normalization="none",
        dropout=0,
        optimizer="adamw",
        learning_rate=0.01,
        weight_decay=0,
        batch_size=16,
        epochs=2,
        schedule="constant",
        warmup_fraction=0,
        decay_fraction=0.2,
    )


@pytest.fixture
def policy():
    return Policy(
        policy_id="base",
        version=0,
        status="baseline",
        planner_instructions="Choose one controlled experiment from measured evidence.",
        developer_instructions="Implement the validated recipe with the trusted builder.",
        reviewer_instructions="Review the measured result and retain uncertainty.",
    )


@pytest.fixture
def limits():
    return Limits(
        max_trials=4,
        max_parameters=1000,
        max_epochs=2,
        per_trial_seconds=10,
        total_compute_seconds=40,
        max_api_calls=3,
        max_context_chars=64000,
        max_output_tokens=1024,
    )


@pytest.fixture
def context(recipe):
    return {
        "task": {"initial_recipe": recipe.model_dump()},
        "records": [],
        "knowledge": [],
        "remaining": {},
    }


def test_mock_roles_do_not_invent_scores_or_call_api(context, policy, limits):
    backend = MockBackend()
    limits.max_api_calls = 0
    plan = backend.call("planner", context, policy, limits)
    development = backend.call(
        "developer", {**context, "current_plan": plan.model_dump()}, policy, limits
    )
    assert development.recipe == plan.recipe
    result = {"status": "completed", "metrics": {"accuracy": 0.625}}
    review = backend.call("reviewer", {**context, "current_result": result}, policy, limits)
    assert "0.625" in review.interpretation
    assert review.prediction_supported == "uncertain"
    assert backend.usage.calls == 0
    assert backend.usage.cost_usd == 0


def test_mock_candidate_does_not_get_a_scripted_advantage(context, policy, limits):
    candidate = policy.model_copy(update={"version": 1, "status": "candidate"})
    baseline_plan = MockBackend().call("planner", context, policy, limits)
    candidate_plan = MockBackend().call("planner", context, candidate, limits)
    assert baseline_plan == candidate_plan


def test_mock_improvement_requires_evidence(context, policy, limits):
    backend = MockBackend()
    with pytest.raises(BackendError, match="evidence"):
        backend.call("improver", context, policy, limits)
    revision = backend.call(
        "improver",
        {**context, "development_runs": [{"run_id": "development-1"}]},
        policy,
        limits,
    )
    assert revision.evidence_run_ids == ["development-1"]
    assert "unvalidated" in revision.planner_instructions
    assert policy.version == 0


def test_missing_credentials_do_not_import_or_call_provider(monkeypatch):
    monkeypatch.delenv("RSI_MODEL", raising=False)
    monkeypatch.delenv("RSI_API_KEY", raising=False)
    with pytest.raises(BackendError, match="RSI_MODEL"):
        DSPyBackend()


@pytest.mark.parametrize(
    "url", ["https://key:secret@host/v1", "https://host/v1?key=secret", "file:///x"]
)
def test_endpoint_rejects_embedded_credentials(url):
    with pytest.raises(BackendError, match="endpoint"):
        DSPyBackend(model="test/model", api_key="secret", api_base=url)


def _dummy(answers):
    dspy = pytest.importorskip("dspy")
    from dspy.utils import DummyLM

    return DummyLM(answers, adapter=dspy.JSONAdapter())


def test_dspy_typed_output_and_call_limit_without_network(context, policy, limits):
    expected = MockBackend().call("planner", context, policy, limits)
    lm = _dummy([{"output": expected.model_dump()}])
    backend = DSPyBackend(lm=lm)
    limits.max_api_calls = 1
    result = backend.call("planner", context, policy, limits)
    assert isinstance(result, Plan)
    assert result == expected
    assert backend.usage.calls == 1
    assert len(lm.history) == 1
    assert "output" in json.dumps(lm.history[0]["outputs"])
    with pytest.raises(BudgetExceeded, match="budget"):
        backend.call("planner", context, policy, limits)
    assert len(lm.history) == 1


def test_dspy_malformed_output_counts_failed_call_once(context, policy, limits):
    lm = _dummy([{"output": {"invented_score": 1.0}}])
    backend = DSPyBackend(lm=lm)
    with pytest.raises(BackendError, match="invalid role response"):
        backend.call("planner", context, policy, limits)
    assert backend.usage.calls == 1
    assert len(lm.history) == 1


@pytest.mark.parametrize("role", ["planner", "reviewer"])
def test_dspy_rejects_diagnostic_request_when_tool_is_unavailable(context, policy, limits, role):
    if role == "planner":
        requested = MockBackend().call(role, context, policy, limits)
        requested.diagnostic_trial_id = "trial-0000"
    else:
        requested = Review(
            interpretation="Request report",
            prediction_supported="uncertain",
            next_step="Inspect errors",
            diagnostic_trial_id="trial-0000",
        )
    lm = _dummy([{"output": requested.model_dump()}])
    backend = DSPyBackend(lm=lm)
    with pytest.raises(BackendError, match="unavailable"):
        backend.call(role, {**context, "tools": {}}, policy, limits)
    assert backend.usage.calls == 1


def test_context_budget_prevents_any_api_call(context, policy, limits):
    backend = DSPyBackend(lm=_dummy([]))
    limits.max_context_chars = 4096
    with pytest.raises(BudgetExceeded, match="Context"):
        backend.call("planner", {**context, "large": "x" * 5000}, policy, limits)
    assert backend.usage.calls == 0


def test_provider_error_does_not_leak_secrets(context, policy, limits):
    dspy = pytest.importorskip("dspy")

    class FailingLM(dspy.BaseLM):
        def __init__(self):
            super().__init__("offline/test")

        def __call__(self, *args, **kwargs):
            raise RuntimeError("Authorization: Bearer secret-do-not-log")

    backend = DSPyBackend(lm=FailingLM())
    with pytest.raises(BackendError) as error:
        backend.call("planner", context, policy, limits)
    assert "secret-do-not-log" not in str(error.value)
    assert backend.usage.calls == 1
    assert backend.usage.tokens_known is False
    assert backend.usage.cost_usd is None


def test_unknown_evidence_is_not_a_failed_decision(context, policy, limits):
    # The workflow logs and strips it; see test_invalid_citation_is_logged_and_removed.
    expected = MockBackend().call("planner", context, policy, limits)
    expected.evidence_ids = ["made-up-paper"]
    backend = DSPyBackend(lm=_dummy([{"output": expected.model_dump()}]))
    assert backend.call("planner", context, policy, limits).evidence_ids == ["made-up-paper"]
    assert backend.usage.calls == 1


def test_resume_preserves_cumulative_cost_and_unknown_flags():
    backend = DSPyBackend(lm=_dummy([]))
    backend.restore_usage(Usage(calls=5, input_tokens=20, output_tokens=10, cost_usd=4.0))
    backend._record_usage([{"usage": {"input_tokens": 3, "output_tokens": 2}, "cost": 0.1}])
    assert backend.usage.calls == 5
    assert backend.usage.input_tokens == 23
    assert backend.usage.cost_usd == pytest.approx(4.1)
    backend.restore_usage(Usage(calls=6, cost_usd=None, tokens_known=False))
    backend._record_usage([{"usage": {"input_tokens": 1, "output_tokens": 1}, "cost": 0.1}])
    assert backend.usage.cost_usd is None
    assert backend.usage.tokens_known is False


def test_endpoint_identity_is_hashed_and_normalized():
    args = {"model": "test/model", "api_key": "never-log-this"}
    first = DSPyBackend(**args, api_base="https://EXAMPLE.com:443/v1")
    same = DSPyBackend(**args, api_base="https://example.com/v1")
    other = DSPyBackend(**args, api_base="https://another.example/v1")
    assert first.metadata["api_base_hash"] == same.metadata["api_base_hash"]
    assert first.metadata["api_base_hash"] != other.metadata["api_base_hash"]
    assert "example.com" not in json.dumps(first.metadata)
    assert "never-log-this" not in json.dumps(first.metadata)


def test_shared_memory_accepts_run_and_qualified_trial_citations(context, policy, limits):
    shared = {
        **context,
        "prior_evidence": [
            {"summary": {"run_id": "development-1"}, "records": [{"trial_id": "trial-0000"}]},
        ],
    }
    expected = MockBackend().call("planner", context, policy, limits)
    expected.evidence_ids = ["development-1", "development-1:trial-0000"]
    backend = DSPyBackend(lm=_dummy([{"output": expected.model_dump()}]))
    assert backend.call("planner", shared, policy, limits) == expected


def _status_lm(dspy, failures):
    """Raise the given DSPy errors in order, then answer like the provider."""
    from dspy.utils import DummyLM

    class Flaky(DummyLM):
        def __call__(self, *args, **kwargs):
            if failures:
                raise failures.pop(0)
            return super().__call__(*args, **kwargs)

    return Flaky


def test_transient_provider_faults_retry_without_consuming_the_decision_budget(
    context, policy, limits
):
    dspy = pytest.importorskip("dspy")
    expected = MockBackend().call("planner", context, policy, limits)
    errors = dspy.clients.errors
    failures = [errors.LMServerError("500"), errors.LMRateLimitError("429")]
    lm = _status_lm(dspy, failures)([{"output": expected.model_dump()}], adapter=dspy.JSONAdapter())
    backend = DSPyBackend(lm=lm, retry_backoff_seconds=0, max_transient_retries=2)
    assert backend.call("planner", context, policy, limits) == expected
    assert backend.usage.calls == 1
    assert backend.usage.transient_retries == 2


@pytest.mark.parametrize("name", ["LMAuthError", "LMBillingError", "LMInvalidRequestError"])
def test_configuration_faults_stop_the_run_instead_of_costing_a_trial(
    name, context, policy, limits
):
    dspy = pytest.importorskip("dspy")
    failures = [getattr(dspy.clients.errors, name)("Bearer secret-do-not-log")]
    lm = _status_lm(dspy, failures)([], adapter=dspy.JSONAdapter())
    backend = DSPyBackend(lm=lm, retry_backoff_seconds=0)
    with pytest.raises(ProviderUnavailable) as error:
        backend.call("planner", context, policy, limits)
    assert not isinstance(error.value, BackendError)
    assert "secret-do-not-log" not in str(error.value)


def test_exhausted_retries_stop_the_run(context, policy, limits):
    dspy = pytest.importorskip("dspy")
    failures = [dspy.clients.errors.LMServerError("503") for _ in range(3)]
    lm = _status_lm(dspy, failures)([], adapter=dspy.JSONAdapter())
    backend = DSPyBackend(lm=lm, retry_backoff_seconds=0, max_transient_retries=2)
    with pytest.raises(ProviderUnavailable, match="3 attempts"):
        backend.call("planner", context, policy, limits)
    assert backend.usage.transient_retries == 2


def test_reasoning_effort_and_timeout_are_recorded_in_run_identity(monkeypatch):
    monkeypatch.setenv("RSI_REASONING_EFFORT", "low")
    monkeypatch.setenv("RSI_API_TIMEOUT", "300")
    backend = DSPyBackend(model="openrouter/test/model", api_key="never-log")
    assert backend.metadata["reasoning_effort"] == "low"
    assert backend.metadata["timeout_seconds"] == 300
    assert "never-log" not in json.dumps(backend.metadata)
