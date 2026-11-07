"""Queue selection contracts; no data preparation, model construction, or training."""

import json

import pytest

from rsi_ml.study import stage_jobs


def write_study(root, arms):
    study = {
        "stages": ["baselines", "agents"],
        "arms": arms,
        "task_order": ["covertype"],
        "tasks": {
            "covertype": {
                "device": "cuda",
                "seeds": [1, 2],
                "orders": {"1": ["A", "B", "C", "D"], "2": ["B", "C", "D", "A"]},
            }
        },
    }
    (root / "study.json").write_text(json.dumps(study))


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        (["D"], ["D"]),
        (["A", "C"], ["C", "A"]),
        (["random"], ["random"]),
        (["D", "random"], ["D", "random"]),
        (["A", "B", "C", "D"], ["B", "C", "D", "A"]),
        (["A", "B", "C", "D", "random"], ["B", "C", "D", "A", "random"]),
    ],
)
def test_agent_jobs_respect_declared_arms_and_seed_order(tmp_path, declared, expected):
    write_study(tmp_path, declared)

    jobs = stage_jobs(tmp_path, "agents", seeds={2})

    assert [job.arm for job in jobs] == expected
    assert all(job.seed == 2 and job.task_id == "covertype" for job in jobs)
    assert not (tmp_path / "runs").exists()


def test_arm_filter_cannot_add_an_undeclared_design(tmp_path):
    write_study(tmp_path, ["D"])

    assert stage_jobs(tmp_path, "agents", seeds={2}, arms={"A", "random"}) == []
    assert [job.arm for job in stage_jobs(tmp_path, "agents", seeds={2}, arms={"A", "D"})] == ["D"]


def test_single_design_still_gets_one_shared_baseline_per_selected_seed(tmp_path):
    write_study(tmp_path, ["D"])

    jobs = stage_jobs(tmp_path, "baselines", seeds={2}, arms={"D"})

    assert [(job.task_id, job.seed, job.arm) for job in jobs] == [("covertype", 2, "baseline")]
