"""The whole pipeline on tiny fixtures: real subprocess training, fake LLM, no network."""

import csv
import gzip
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
from fake_llm import FakeLLM

from rsi_ml.baselines import adopt_baseline, training_identity
from rsi_ml.cli import load_task, main
from rsi_ml.prepare import (
    COVERTYPE_CLASSES,
    COVERTYPE_FEATURES,
    _spambase_columns,
    fetch_sources,
    prepare_all,
    source_plan,
    stratified_indices,
)
from rsi_ml.storage import (
    RunLock,
    Store,
    comparable_environment,
    environment,
    read_events,
    read_json,
    training_fingerprint,
    write_json,
)
from rsi_ml.study import (
    build_study,
    job_action,
    load_api_key,
    run_dir,
    stage_jobs,
    study_status,
)

ROOT = Path(__file__).parents[1]
PROPOSED = ROOT / "configs" / "study" / "v1.proposed.json"
FEATURES = [f"word_freq_{i}" for i in range(57)]
SENTENCES = ["good movie", "bad movie", "good good", "bad bad", "movie good", "movie bad"]


def _names_file() -> str:
    return "| comment line: continuous.\n" + "".join(f"{n}:   continuous.\n" for n in FEATURES)


def write_fixture_downloads(root: Path) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    transformers = pytest.importorskip("transformers")
    safetensors = pytest.importorskip("safetensors.torch")
    downloads = root / "downloads"
    downloads.mkdir(parents=True)
    rng = np.random.default_rng(0)

    def cifar_records(count):
        labels = np.arange(count) % 5
        images = rng.integers(0, 256, size=(count, 3072), dtype=np.uint8)
        return np.column_stack([np.zeros(count, dtype=np.uint8), labels, images]).astype(np.uint8)

    with tarfile.open(downloads / "cifar-100-binary.tar.gz", "w:gz") as archive:
        members = {
            "train.bin": cifar_records(100).tobytes(),
            "test.bin": cifar_records(20).tobytes(),
            "fine_label_names.txt": "\n".join(f"class_{i}" for i in range(100)).encode(),
        }
        for name, data in members.items():
            info = tarfile.TarInfo(f"cifar-100-binary/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

    rows = rng.random((60, 57))
    labels = np.arange(60) % 2
    table = "\n".join(",".join(map(str, [*r, y])) for r, y in zip(rows, labels)) + "\n"
    with zipfile.ZipFile(downloads / "spambase.zip", "w") as archive:
        archive.writestr("spambase.data", table)
        archive.writestr("spambase.names", _names_file())

    forest = rng.integers(0, 100, size=(70, 54))
    table = "".join(",".join(map(str, [*r, i % 7 + 1])) + "\n" for i, r in enumerate(forest))
    with zipfile.ZipFile(downloads / "covertype.zip", "w") as archive:
        archive.writestr("covtype.data.gz", gzip.compress(table.encode()))
        archive.writestr("covtype.info", "Forest cover type notes\n")

    def write_parquet(name, records):
        parquet.write_table(pyarrow.Table.from_pylist(records), downloads / name)

    def sst(count, labeled=True):
        return [
            {
                "sentence": f"{SENTENCES[i % 6]} {i}",
                "label": (i % 2 if labeled else -1),
                "idx": i,
            }
            for i in range(count)
        ]

    write_parquet("glue-sst2-train.parquet", sst(24))
    write_parquet("glue-sst2-validation.parquet", sst(8))
    write_parquet("glue-sst2-test.parquet", sst(6, labeled=False))
    (downloads / "setfit-sst2-test.jsonl").write_text(
        "".join(
            json.dumps({"text": row["sentence"], "label": i % 2, "label_text": "x"}) + "\n"
            for i, row in enumerate(reversed(sst(6, labeled=False)))
        )
    )
    pairs = lambda count: [
        {
            "sentence1": SENTENCES[i % 6],
            "sentence2": 'a "quoted"\ttabbed ' + SENTENCES[(i + 1) % 6],
            "label": i % 2,
            "idx": i,
        }
        for i in range(count)
    ]
    write_parquet("glue-mrpc-train.parquet", pairs(24))
    write_parquet("glue-mrpc-validation.parquet", pairs(8))
    write_parquet("glue-mrpc-test.parquet", pairs(6))

    # Same file shape as the real BERT-Tiny: no tokenizer config, keys prefixed by `bert.`,
    # plus pretraining-head weights the encoder must ignore.
    vocabulary = (
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ngood\nbad\nmovie\na\nquoted\ntabbed\n0\n1\n2\n3\n"
    )
    (downloads / "bert-tiny-vocab.txt").write_text(vocabulary)
    config = transformers.BertConfig(
        vocab_size=15,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=512,  # As in the real BERT-Tiny.
    )
    (downloads / "bert-tiny-config.json").write_text(config.to_json_string())
    state = transformers.BertModel(config, add_pooling_layer=False).state_dict()
    tensors = {f"bert.{k}": v.clone() for k, v in state.items()}
    tensors["cls.predictions.bias"] = rng_tensor(15)
    safetensors.save_file(tensors, downloads / "bert-tiny-model.safetensors")


def rng_tensor(size):
    import torch

    return torch.zeros(size)


def tiny_protocol(path: Path, llm_url: str | None, *, targets_set: bool = True) -> Path:
    protocol = read_json(PROPOSED)
    protocol["status"] = "approved"
    protocol["policy"] = str(ROOT / "configs" / "policy.json")
    protocol["common_limits"].update(max_trials=3, max_context_chars=100000, max_output_tokens=2000)
    stage = protocol["agent_stage"]
    if targets_set:
        stage.update(model="openrouter/openai/gpt-6-luna", api_base=llm_url, api_timeout_seconds=30)
        for key in protocol["tasks"]:
            stage["targets"][key] = 0.2
            stage["per_trial_seconds"][key] = 300
            stage["total_compute_seconds"][key] = 900
    for key, task in protocol["tasks"].items():
        task["template"] = str(ROOT / "configs" / f"{key}.template.json")
        task["device"] = "cpu"
        task["seeds"] = [1]
        task["initial_recipe"].update(epochs=1, batch_size=8)
        task["max_epochs"] = 2
        # Small and stable, so the reference arm stays quick and always completes on CPU.
        task["random_search"].update(
            epochs={"int": [1, 1]},
            batch_size={"int": [4, 8]},
            learning_rate={"log_uniform": [0.001, 0.01]},
        )
    cnn = protocol["tasks"]["cifar100"]
    cnn["initial_recipe"].update(
        cnn_channels=[4, 8, 8, 8], cnn_blocks=[1, 1, 1, 1], learning_rate=0.03
    )
    cnn["random_search"].update(
        cnn_channels={"stage_widths": {"base": [4, 8], "stages": 4}},
        cnn_blocks={"int_list": {"length": 4, "range": [1, 1]}},
    )
    protocol["tasks"]["spambase"]["random_search"]["hidden_sizes"] = {
        "layers": {"depth": [1, 2], "width": [8, 32]}
    }
    for key in ("mrpc", "sst2"):
        text = protocol["tasks"][key]
        text["initial_recipe"].update(max_length=16)
        text["random_search"].update(
            max_length={"int": [8, 16]},
            hidden_sizes={"layers": {"depth": [0, 1], "width": [8, 16]}},
        )
    path.write_text(json.dumps(protocol, indent=1))
    return path


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    write_fixture_downloads(root)
    prepare_all(root, fetch=False)
    return root


def test_proposed_protocol_validates_and_stays_gated(tmp_path, data_root):
    protocol = read_json(PROPOSED)
    assert protocol["status"] == "proposed"
    with pytest.raises(ValueError, match="proposed"):
        build_study(PROPOSED, data_root, tmp_path / "refused")
    assert not (tmp_path / "refused").exists()


def test_agents_choose_freely_under_the_proposed_constraints(tmp_path, data_root):
    """Exactly the settings we would run: no menu, and limits that really bind."""
    from rsi_ml.baseline import RandomSearchBackend
    from rsi_ml.tasks.registry import parameter_count, validate_recipe

    protocol = read_json(PROPOSED)
    protocol["status"] = "approved"
    protocol["policy"] = str(ROOT / "configs" / "policy.json")
    for key, task in protocol["tasks"].items():
        task["template"] = str(ROOT / "configs" / f"{key}.template.json")
    path = tmp_path / "exact.json"
    path.write_text(json.dumps(protocol))
    study = tmp_path / "exact"
    build_study(path, data_root, study)
    tasks = {
        key: load_task(study / "tasks" / "baselines" / key / "seed-1.json")
        for key in protocol["tasks"]
    }
    for key, task in tasks.items():
        assert task.recipe_policy == "constraints" and not task.search_space
        # Numbers nobody listed anywhere are legal choices.
        unlisted = {
            "learning_rate": 0.0123,
            "dropout": 0.37,
            "batch_size": 77,
            "weight_decay": 0.0042,
            "epochs": 13,
            "warmup_fraction": 0.07,
        }
        validate_recipe(task, task.initial_recipe.model_copy(update=unlisted))
        # The reference arm's own ranges always yield recipes within the same limits.
        backend = RandomSearchBackend()
        backend.bind_validator(lambda recipe, task=task: validate_recipe(task, recipe))
        drawn = [
            backend._plan(
                {
                    "task": {
                        "seed": 1,
                        "initial_recipe": task.initial_recipe.model_dump(mode="json"),
                    },
                    "records": [],
                    "history_count": count,
                    "random_search": task.random_search,
                }
            ).recipe
            for count in range(10)
        ]
        for recipe in drawn:
            validate_recipe(task, recipe)
        assert len({recipe.model_dump_json() for recipe in drawn}) > 1, key

    def recipe_for(key, **changes):
        return tasks[key].initial_recipe.model_copy(update=changes)

    # The agent decides depth and width; only the parameter ceiling stops it.
    validate_recipe(tasks["spambase"], recipe_for("spambase", hidden_sizes=[37, 211, 5, 90, 12]))
    with pytest.raises(ValueError, match="parameters"):
        validate_recipe(tasks["spambase"], recipe_for("spambase", hidden_sizes=[1024, 1024]))
    five_stages = recipe_for(
        "cifar100", cnn_channels=[32, 64, 128, 256, 512], cnn_blocks=[1, 1, 1, 1, 1]
    )
    validate_recipe(tasks["cifar100"], five_stages)
    largest = recipe_for("cifar100", cnn_channels=[80, 160, 320, 640], cnn_blocks=[2, 2, 3, 2])
    assert validate_recipe(tasks["cifar100"], largest) <= tasks["cifar100"].limits.max_parameters
    with pytest.raises(ValueError, match="parameters"):
        validate_recipe(
            tasks["cifar100"], recipe_for("cifar100", cnn_channels=[128, 256, 512, 1024])
        )
    validate_recipe(tasks["mrpc"], recipe_for("mrpc", hidden_sizes=[317], max_length=77))
    # A linear model is the smallest legal choice, and a no-op activation change is refused.
    validate_recipe(tasks["spambase"], recipe_for("spambase", hidden_sizes=[]))
    with pytest.raises(ValueError, match="no effect"):
        validate_recipe(tasks["spambase"], recipe_for("spambase", hidden_sizes=[], dropout=0.3))
    with pytest.raises(ValueError, match="no effect"):
        validate_recipe(tasks["cifar100"], recipe_for("cifar100", activation="gelu"))
    # Fields a model family does not use are still not choices.
    with pytest.raises(ValueError, match="not an editable"):
        validate_recipe(tasks["cifar100"], recipe_for("cifar100", pooling="mean"))
    with pytest.raises(ValueError, match="epoch limit"):
        validate_recipe(tasks["spambase"], recipe_for("spambase", epochs=100000))

    counts = {key: parameter_count(task, task.initial_recipe) for key, task in tasks.items()}
    assert 11_000_000 < counts["cifar100"] < 11_500_000  # ResNet-18, CIFAR stem, 100 classes.
    assert counts["spambase"] == (57 * 128 + 128) + (128 * 64 + 64) + (64 * 2 + 2)


@pytest.mark.slow
def test_full_pipeline_through_all_stages(tmp_path, data_root, monkeypatch):
    # No backoff waits here; retries themselves are covered in test_agents.
    monkeypatch.setenv("RSI_API_MAX_RETRIES", "0")
    study_dir = tmp_path / "study"
    key_file = tmp_path / "api.env"
    key_file.write_text("OPENROUTER_API_KEY=sk-fake-never-logged\n")
    with FakeLLM() as llm:
        protocol = tiny_protocol(tmp_path / "protocol.json", llm.url)
        assert (
            main(
                [
                    "study-build",
                    "--protocol",
                    str(protocol),
                    "--data",
                    str(data_root),
                    "--out",
                    str(study_dir),
                    "--execute",
                ]
            )
            == 0
        )
        # Planning runs nothing.
        assert main(["study-run", "--study", str(study_dir), "--stage", "baselines"]) == 0
        assert not (study_dir / "runs").exists()
        # Stage 1: no API at all.
        assert (
            main(
                [
                    "study-run",
                    "--study",
                    str(study_dir),
                    "--stage",
                    "baselines",
                    "--cpu-slots",
                    "2",
                    "--execute",
                ]
            )
            == 0
        )
        assert not llm.contexts
        baselines = {
            job.task_id: read_json(job.out / "baseline.json")
            for job in stage_jobs(study_dir, "baselines")
        }
        assert all(b["result"]["status"] == "completed" for b in baselines.values())
        # A provider outage stops the run at a resumable point instead of costing a trial.
        outage = [
            "study-run",
            "--study",
            str(study_dir),
            "--stage",
            "agents",
            "--tasks",
            "spambase",
            "--arms",
            "A",
            "--env-file",
            str(key_file),
            "--execute",
        ]
        llm.failures_to_inject = 1
        assert main(outage) == 1
        interrupted = run_dir(study_dir, "spambase", 1, "A")
        assert [r["trial_id"] for r in read_json(interrupted / "state.json")["experiments"]] == [
            "trial-0000"
        ]
        assert (interrupted / "inflight.json").exists()
        assert not (interrupted / "summary.json").exists()
        assert (
            "Provider still failing"
            in (study_dir / "logs/agents/spambase.seed-1.A.log").read_text()
        )
        # Recovery is explicit: without the flag the run stays untouched.
        assert main(outage) == 1
        assert (interrupted / "inflight.json").exists()
        assert main([*outage, "--recover"]) == 0
        recovered = read_json(interrupted / "summary.json")
        assert recovered["total_trials"] == 3
        assert all(
            r["result"]["status"] == "completed"
            for r in read_json(interrupted / "state.json")["experiments"]
        )
        assert "interrupted_operation_recovered" in (interrupted / "events.jsonl").read_text()
        # Everything else, in parallel.
        assert (
            main(
                [
                    "study-run",
                    "--study",
                    str(study_dir),
                    "--stage",
                    "agents",
                    "--cpu-slots",
                    "2",
                    "--env-file",
                    str(key_file),
                    "--execute",
                ]
            )
            == 0
        )
        jobs = stage_jobs(study_dir, "agents")
        assert len(jobs) == 4 * 5
        assert all(job_action(job) == "done" for job in jobs)
        # Rerunning does nothing: a finished study is never repeated.
        contexts_before = len(llm.contexts)
        assert (
            main(
                [
                    "study-run",
                    "--study",
                    str(study_dir),
                    "--stage",
                    "agents",
                    "--env-file",
                    str(key_file),
                    "--execute",
                ]
            )
            == 0
        )
        assert len(llm.contexts) == contexts_before

    for job in jobs:
        summary = read_json(job.out / "summary.json")
        state = read_json(job.out / "state.json")
        baseline = baselines[job.task_id]
        # Every arm's trial 0 is the one shared baseline, charged at its measured cost.
        first = state["experiments"][0]
        assert first["result"]["metrics"] == baseline["result"]["metrics"]
        assert first["result"]["compute_seconds"] == baseline["result"]["compute_seconds"]
        assert (job.out / "trials/trial-0000/checkpoint.pt").stat().st_nlink >= 2
        assert summary["total_trials"] == 3
        assert all(r["result"]["status"] == "completed" for r in state["experiments"]), job.job_id
        assert summary["starting_model_hash"] == baseline["starting_model_hash"]
        # Stopping on target is off: runs keep the whole trajectory.
        assert summary["stop_reason"] == "trial_budget_exhausted"
        assert (summary["usage"]["calls"] > 0) == (job.arm != "random")
        if job.arm != "random":  # The provider's charged amount, not a price-table estimate.
            assert summary["usage"]["cost_usd"] == pytest.approx(0.001 * summary["usage"]["calls"])
        assert (
            job.arm == "random"
            or summary["usage"]["calls"] == {"A": 2, "B": 2, "C": 5, "D": 5}[job.arm]
        )

    # OpenRouter only returns its charged cost when usage accounting is requested.
    assert all(body["usage"] == {"include": True} for body in llm.bodies)

    # What each arm actually saw.
    for context in llm.contexts:
        serialized = json.dumps(context)
        assert "workflow" not in context and "policy" not in context
        # No menu reaches an agent: only the constraints, in every arm.
        assert not context["task"]["search_space"] and "random_search" not in context["task"]
        assert "random_search" not in context and context["recipe_rules"]["constraints"]
        assert "sk-fake" not in serialized
        assert context["task"]["dataset"].get("class_names")
    with_report = [c for c in llm.contexts if any("diagnostic_report" in r for r in c["records"])]
    assert with_report, "B/D must receive reports"
    for context in llm.contexts:
        for record in context["records"]:
            # Everyone gets ordinary training and validation curves.
            if record["result"]["status"] == "completed":
                assert record["result"]["curve_summary"]
                assert "validation_loss" in record["result"]["curve_summary"][-1]
                assert "train_accuracy" in record["result"]["curve_summary"][-1]
    assert not any("tools" in c and c["tools"] for c in llm.contexts)
    text_reports = [
        r["diagnostic_report"]
        for c in with_report
        if c["task"]["adapter"] == "text_classification"
        for r in c["records"]
        if "diagnostic_report" in r and r["diagnostic_report"]["confident_errors"]
    ]
    assert text_reports and "text" in text_reports[0]["confident_errors"][0]
    assert text_reports[0]["largest_confusions"][0]["true"] in {
        "negative",
        "positive",
        "not_paraphrase",
        "paraphrase",
    }
    tabular = [
        r["diagnostic_report"]
        for c in with_report
        if c["task"]["adapter"] == "mlp_classification"
        for r in c["records"]
        if "diagnostic_report" in r
    ]
    assert tabular and tabular[0]["features_distinguishing_errors"][0]["feature"].startswith(
        "word_freq"
    )

    # A/C reports never reach disk, even though raw outputs are kept for later analysis.
    for job in jobs:
        reports = list(job.out.rglob("diagnostic_report.json"))
        assert bool(reports) == (job.arm in ("B", "D")), job.job_id
        assert list(job.out.glob("trials/*/validation_outputs.npz"))

    # A citation slip was logged and did not reject the attempt.
    assert "invalid_evidence_citation" in (jobs[0].out / "events.jsonl").read_text()
    # The API key is nowhere in the study's records.
    for path in study_dir.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".log", ".csv"}:
            assert "sk-fake-never-logged" not in path.read_text(errors="ignore"), path

    status = study_status(study_dir)
    assert status["projection"]["spambase"]["agent_trials_remaining_at_most"] == 0
    assert main(["study-figures", "--study", str(study_dir)]) == 0
    figures = study_dir / "figures"
    for name in (
        "fig1_quality_vs_compute",
        "fig2_quality_vs_experiments",
        "fig3_cost_to_target",
        "fig4_experiment_outcomes",
        "fig5_agent_behavior",
        "fig6_score_vs_parameters",
        "timeline_cifar100_seed-1",
    ):
        assert (figures / f"{name}.svg").exists() and (figures / f"{name}.png").exists(), name
    receipt = read_json(figures / "figures.json")
    assert set(receipt["matched_cases"].values()) == {"matched"}
    assert not receipt["missing_or_unfinished"]
    contrasts = (figures / "contrasts.csv").read_text()
    for name in ("B_vs_A", "D_vs_C", "C_vs_A", "D_vs_B", "interaction"):
        assert name in contrasts
    assert "fields_changed_from_best" in (figures / "decisions.csv").read_text()

    # Final test scoring is separate, explicit, and selects by validation only.
    assert main(["study-score-test", "--study", str(study_dir)]) == 1
    assert main(["study-score-test", "--study", str(study_dir), "--execute"]) == 0
    final = read_json(study_dir / "final_test.json")
    assert len(final["scored"]) == len(jobs) and not final["skipped"]


def test_crash_recovery_is_explicit_and_keeps_partial_work(tmp_path):
    root = tmp_path / "run"
    (root / "trials" / "trial-0001").mkdir(parents=True)
    (root / "trials" / "trial-0001" / "partial.txt").write_text("interrupted")
    write_json(
        root / "state.json", {"experiments": [{"trial_id": "trial-0000"}], "next_node": "execute"}
    )
    write_json(root / "inflight.json", {"operation": "abcd1234", "node": "execute"})
    store = Store(root)
    with pytest.raises(RuntimeError, match="recover-interrupted"):
        store.resumable_state()
    assert (root / "trials/trial-0001/partial.txt").exists()
    state = store.resumable_state(recover=True)
    assert state["next_node"] == "execute"
    assert not (root / "trials/trial-0001").exists() and not (root / "inflight.json").exists()
    assert (root / "interrupted/trial-0001-abcd1234/partial.txt").read_text() == "interrupted"
    event = read_events(root / "events.jsonl")[-1]
    assert event["event"] == "interrupted_operation_recovered"


def test_stale_lock_is_cleared_only_when_its_process_is_gone(tmp_path):
    lock = tmp_path / ".run.lock"
    lock.write_text(json.dumps({"pid": 2**22 - 1}))
    with pytest.raises(RuntimeError, match="locked"), RunLock(tmp_path):
        pass
    with RunLock(tmp_path, recover=True):
        pass
    import os

    lock.write_text(json.dumps({"pid": os.getpid()}))
    with pytest.raises(RuntimeError, match="live process"), RunLock(tmp_path, recover=True):
        pass


def test_gpu_assignment_does_not_change_run_identity():
    env = environment()
    one = {
        **env,
        "hardware": {
            **env["hardware"],
            "cuda_devices": [{"name": "GPU", "memory": 8, "uuid": "x"}],
        },
    }
    two = {
        **env,
        "hardware": {
            **env["hardware"],
            "cuda_devices": [{"name": "GPU", "memory": 8, "uuid": "y"}],
        },
    }
    other = {
        **env,
        "hardware": {
            **env["hardware"],
            "cuda_devices": [{"name": "Other", "memory": 8, "uuid": "y"}],
        },
    }
    assert comparable_environment(one) == comparable_environment(two)
    assert comparable_environment(one) != comparable_environment(other)
    assert training_fingerprint(one) != training_fingerprint(other)
    changed_agent_code = {**one, "source_hashes": {**one["source_hashes"], "agents.py": "edited"}}
    assert training_fingerprint(changed_agent_code) == training_fingerprint(one)
    changed_training = {
        **one,
        "source_hashes": {**one["source_hashes"], "tasks/training.py": "edited"},
    }
    assert training_fingerprint(changed_training) != training_fingerprint(one)


def test_baseline_adoption_refuses_anything_that_changes_training(tmp_path, data_root):
    study = tmp_path / "study"
    protocol = tiny_protocol(tmp_path / "p.json", None, targets_set=False)
    build_study(protocol, data_root, study)
    task = load_task(study / "tasks/baselines/spambase/seed-1.json")
    changed_recipe = task.model_copy(
        update={"initial_recipe": task.initial_recipe.model_copy(update={"dropout": 0.3})}
    )
    assert training_identity(task) != training_identity(changed_recipe)
    assert training_identity(task) != training_identity(task.model_copy(update={"seed": 2}))
    # Targets and budgets only steer later decisions, so a pilot baseline stays reusable.
    assert training_identity(task) == training_identity(task.model_copy(update={"target": 0.5}))
    source = tmp_path / "baseline"
    source.mkdir()
    write_json(
        source / "baseline.json",
        {
            "training_identity": training_identity(task),
            "training_fingerprint": "from another machine",
            "starting_model_hash": "h",
            "dataset_hash": "d",
            "result": {"status": "completed", "compute_seconds": 1},
        },
    )
    with pytest.raises(ValueError, match="training code/environment"):
        adopt_baseline(
            task, source, tmp_path / "trial", {"starting_model_hash": "h", "dataset_hash": "d"}
        )


def test_cheap_tasks_run_first_despite_sorted_json_records(tmp_path, data_root):
    """Hours of CIFAR must not delay the first feedback from the cheap tasks."""
    study = tmp_path / "study"
    build_study(tiny_protocol(tmp_path / "p.json", None, targets_set=False), data_root, study)
    assert [job.task_id for job in stage_jobs(study, "baselines")] == [
        "spambase",
        "mrpc",
        "sst2",
        "cifar100",
    ]


def test_agent_stage_is_blocked_until_its_settings_are_chosen(tmp_path, data_root):
    study = tmp_path / "study"
    protocol = tiny_protocol(tmp_path / "p.json", None, targets_set=False)
    built = build_study(protocol, data_root, study)
    assert built["stages"] == ["baselines"]
    assert "agent_stage.targets.cifar100" in built["unresolved_for_agent_stage"]
    assert stage_jobs(study, "baselines")
    with pytest.raises(ValueError, match="not built"):
        stage_jobs(study, "agents")


def test_splits_are_deterministic_disjoint_and_stratified():
    labels = np.array([0] * 900 + [1] * 100)
    first = stratified_indices(labels, {"train": 0.6, "validation": 0.2, "test": 0.2}, 7)
    assert first == stratified_indices(labels, {"train": 0.6, "validation": 0.2, "test": 0.2}, 7)
    assert first != stratified_indices(labels, {"train": 0.6, "validation": 0.2, "test": 0.2}, 8)
    assert sorted(i for v in first.values() for i in v) == list(range(1000))
    assert labels[first["validation"]].mean() == pytest.approx(0.1)
    assert labels[first["test"]].mean() == pytest.approx(0.1)


def test_prepared_data_records_provenance_and_text_survives_round_trip(data_root):
    manifest = read_json(data_root / "selected/cifar100/manifest.json")
    assert len(manifest["class_names"]) == 100
    assert len(manifest["image_mean"]) == 3
    assert manifest["normalization_fitted_on"] == "selected train split only"
    assert read_json(data_root / "selected/spambase/manifest.json")["class_names"] == [
        "not_spam",
        "spam",
    ]
    row = json.loads((data_root / "selected/mrpc/train.jsonl").read_text().splitlines()[0])
    assert row["sentence2"].startswith('a "quoted"\ttabbed')
    # SST-2 test labels come from an exact-text join, not row order.
    test_rows = [
        json.loads(line)
        for line in (data_root / "selected/sst2/test.jsonl").read_text().splitlines()
    ]
    assert len(test_rows) == 6 and {r["label"] for r in test_rows} == {0, 1}
    pin = read_json(data_root / "models/bert-tiny.pin.json")
    assert set(pin["file_sha256"]) == {"config.json", "vocab.txt", "model.safetensors"}


def test_sst2_join_stops_on_any_unmatched_sentence(tmp_path):
    from rsi_ml.prepare import build_sst2

    write_fixture_downloads(tmp_path)
    (tmp_path / "downloads" / "setfit-sst2-test.jsonl").write_text(
        json.dumps({"text": "not one of them", "label": 0}) + "\n"
    )
    with pytest.raises(ValueError, match="no labeled match"):
        build_sst2(tmp_path)


def test_source_plan_downloads_nothing_and_spambase_names_parse(tmp_path):
    plan = source_plan()
    assert plan["downloaded"] is False and len(plan["sources"]) == 13
    assert all(
        s["sha256_pinned"] or s["name"].split(".")[-1] in {"gz", "zip", "jsonl", "json", "txt"}
        for s in plan["sources"]
    )
    assert _spambase_columns(_names_file()) == FEATURES
    with pytest.raises(ValueError, match="57"):
        _spambase_columns("only_one: continuous.")
    assert main(["prepare-sources", "--root", str(tmp_path / "unused")]) == 0
    assert not (tmp_path / "unused").exists()


def test_covertype_numbers_classes_from_zero_and_keeps_every_class(data_root):
    manifest = read_json(data_root / "selected" / "covertype" / "manifest.json")
    assert manifest["class_names"] == COVERTYPE_CLASSES
    assert len(COVERTYPE_FEATURES) == 54
    assert sum(split["rows"] for split in manifest["splits"].values()) == 70
    for split in manifest["splits"].values():
        with open(split["path"], newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert list(rows[0]) == [*COVERTYPE_FEATURES, "label"]
        assert {int(row["label"]) for row in rows} == set(range(7))


def test_covertype_study_changes_only_what_was_decided(tmp_path, data_root):
    path = ROOT / "configs" / "study" / "mlp-covertype.json"
    study, demo = read_json(path), read_json(ROOT / "configs" / "study" / "mlp-demo.json")
    for section in ("arms", "artifact_policy", "common_limits", "autopilot", "stop_on_target"):
        assert study[section] == demo[section]
    assert study["agent_stage"]["model"] == demo["agent_stage"]["model"]
    new, old = study["tasks"]["covertype"], demo["tasks"]["spambase"]
    assert new["initial_recipe"] == {**old["initial_recipe"], "epochs": 5}
    assert new["random_search"] == {**old["random_search"], "epochs": {"int": [3, 10]}}
    for key in ("seeds", "max_parameters", "max_epochs", "device", "recipe_policy"):
        assert new[key] == old[key]
    built = build_study(path, data_root, tmp_path / "study")
    assert built["tasks"]["covertype"]["initial_parameters"] == 15751


def test_the_600_second_study_changes_only_the_budget_and_random_scale(tmp_path, data_root):
    path = ROOT / "configs" / "study" / "mlp-covertype-600s.json"
    study, base = read_json(path), read_json(ROOT / "configs" / "study" / "mlp-covertype.json")
    for section in (
        "arms",
        "artifact_policy",
        "autopilot",
        "baseline_stage",
        "common_limits",
        "policy",
        "stop_on_target",
    ):
        assert study[section] == base[section]
    stage = study["agent_stage"]
    assert stage["total_compute_seconds"] == {"covertype": 600}
    assert stage["per_trial_seconds"] == {"covertype": 267}
    assert stage["targets"] == {"covertype": None}
    assert stage["model"] == base["agent_stage"]["model"]
    new, old = study["tasks"]["covertype"], base["tasks"]["covertype"]
    assert new["initial_recipe"] == old["initial_recipe"]
    assert new["random_search"] == {
        **old["random_search"],
        "epochs": {"int": [9, 36]},
        "max_steps_vs_baseline": 9.0,
    }
    for key in ("seeds", "max_parameters", "max_epochs", "device", "recipe_policy", "template"):
        assert new[key] == old[key]
    assert build_study(path, data_root, tmp_path / "study")["stages"] == ["baselines"]


def test_fetching_one_task_keeps_the_record_of_earlier_downloads(tmp_path):
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    for name in ("spambase.zip", "covertype.zip"):
        (downloads / name).write_bytes(b"already here")
    fetch_sources(tmp_path, {"spambase"})
    lock = fetch_sources(tmp_path, {"covertype"})
    assert set(lock["sources"]) == {"spambase.zip", "covertype.zip"}


def test_api_key_loader_reads_only_the_key(tmp_path):
    path = tmp_path / "env"
    path.write_text("# comment\nCS329Z_BUDGET_USD=200\nexport OPENROUTER_API_KEY='sk-value'\n")
    assert load_api_key(path) == {"RSI_API_KEY": "sk-value"}
    path.write_text("OTHER=1\n")
    with pytest.raises(ValueError, match="No RSI_API_KEY"):
        load_api_key(path)


def _autopilot_protocol(path: Path, llm_url: str, **rules) -> Path:
    """Two quick tasks, with targets and budgets left for the autopilot to derive."""
    tiny_protocol(path, llm_url)
    protocol = read_json(path)
    for key in ("cifar100", "sst2"):
        del protocol["tasks"][key]
    for group in ("targets", "per_trial_seconds", "total_compute_seconds"):
        protocol["agent_stage"][group] = {"spambase": None, "mrpc": None}
    # Tests must not depend on this machine's free disk; the guard has its own test.
    protocol["autopilot"].update(**{"seconds_between_passes": 0, "min_free_disk_gb": 0, **rules})
    path.write_text(json.dumps(protocol))
    return path


@pytest.mark.slow
def test_autopilot_runs_everything_unattended_through_an_outage(tmp_path, data_root, monkeypatch):
    from rsi_ml.autopilot import autopilot

    monkeypatch.setenv("RSI_API_MAX_RETRIES", "0")
    key_file = tmp_path / "api.env"
    key_file.write_text("OPENROUTER_API_KEY=sk-fake-never-logged\n")
    out = tmp_path / "study"
    with FakeLLM() as llm:
        protocol = _autopilot_protocol(tmp_path / "protocol.json", llm.url)
        # Past the preflight call: costs one pass, recovered with no one watching.
        llm.failures_after, llm.failures_to_inject = 1, 1
        result = autopilot(protocol, data_root, out, gpus=[], env_file=key_file)
        assert result["status"] == "complete", result
        passes = [e for e in result["log"] if e["step"] == "agents_pass"]
        assert len(passes) == 2 and passes[0]["failed"] and not passes[-1]["failed"]
        # Finished work is never repeated, and the recorded settings are reused.
        calls = len(llm.contexts)
        again = autopilot(protocol, data_root, out, gpus=[], env_file=key_file)
        assert again["status"] == "complete" and len(llm.contexts) == calls

    jobs = stage_jobs(out, "agents")
    assert len(jobs) == 2 * 5
    # Protocol order survives the sorted JSON records: cheap tasks first.
    assert read_json(out / "study.json")["task_order"] == ["spambase", "mrpc"]
    assert jobs[0].task_id == "spambase"
    preflight = read_json(out / "api_preflight.json")
    assert preflight["task"] == "mrpc" and preflight["recipe_valid_for_task"]
    assert preflight["context_characters"] > 20000  # Sized like a late D prompt.
    derived = read_json(out / "agent_stage.derived.json")
    for task_id, measured in derived["measured"].items():
        median = measured["median_score"]
        assert derived["agent_stage"]["targets"][task_id] == pytest.approx(
            min(0.999, median + 0.1 * (1 - median)), abs=1e-4
        )
        task = load_task(out / "tasks/agents" / task_id / "seed-1.json")
        assert task.target == derived["agent_stage"]["targets"][task_id]
        assert task.limits.per_trial_seconds >= max(measured["seconds"]) * 4
    for job in jobs:
        run = read_json(job.out / "summary.json")
        assert run["total_trials"] == 3
        assert all(
            r["result"]["status"] == "completed"
            for r in read_json(job.out / "state.json")["experiments"]
        ), job.job_id
    assert (out / "figures/fig1_quality_vs_compute.png").exists()
    assert len(read_json(out / "final_test.json")["scored"]) == len(jobs)
    assert "final_test" in {e["step"] for e in read_json(out / "autopilot.json")["log"]}


@pytest.mark.slow
def test_autopilot_stops_at_guardrails_and_continues_after(tmp_path, data_root):
    from rsi_ml.autopilot import autopilot

    key_file = tmp_path / "api.env"
    key_file.write_text("OPENROUTER_API_KEY=sk-fake\n")
    out = tmp_path / "study"
    with FakeLLM() as llm:
        # Low disk: nothing trains, nothing is deleted.
        protocol = _autopilot_protocol(tmp_path / "p.json", llm.url, min_free_disk_gb=10**9)
        stopped = autopilot(protocol, data_root, out, gpus=[], env_file=key_file)
        assert stopped["status"] == "stopped_at_guardrail" and "disk" in stopped["reason"]
        assert not list(out.glob("runs/*/*/baseline/baseline.json"))
        # The spending cap holds new agent runs once it is reached.
        protocol = _autopilot_protocol(tmp_path / "p.json", llm.url, max_api_usd=0.0001)
        stopped = autopilot(
            protocol, data_root, out, gpus=[], env_file=key_file, sleep=lambda s: None
        )
        assert stopped["status"] == "stopped_at_guardrail" and "cap" in stopped["reason"]
        assert stopped["held"] and (out / "figures").exists()
        # Raising the limit and rerunning the same command finishes the study.
        resolved = read_json(out / "protocol.resolved.json")
        resolved["autopilot"]["max_api_usd"] = 50
        write_json(out / "protocol.resolved.json", resolved)
        protocol = _autopilot_protocol(tmp_path / "p.json", llm.url, max_api_usd=50)
        finished = autopilot(protocol, data_root, out, gpus=[], env_file=key_file)
        assert finished["status"] == "complete", finished

        # A seed can join later: it trains only what is new, and keeps the fixed targets.
        targets = read_json(out / "protocol.resolved.json")["agent_stage"]["targets"]
        edited = read_json(protocol)
        edited["tasks"]["spambase"]["seeds"] = [1, 2]
        protocol.write_text(json.dumps(edited))
        calls_before = len(llm.bodies)
        grown = autopilot(protocol, data_root, out, gpus=[], env_file=key_file)
        assert grown["status"] == "complete", grown
        assert (run_dir(out, "spambase", 2, "baseline") / "baseline.json").exists()
        assert (run_dir(out, "spambase", 2, "D") / "summary.json").exists()
        assert read_json(out / "protocol.resolved.json")["agent_stage"]["targets"] == targets
        # Five LLM decisions and reviews for the new seed; nothing finished is repeated.
        assert len(llm.bodies) - calls_before == 2 + 2 + 5 + 5

        # Anything else changing means a different study, and it refuses to mix them.
        edited["tasks"]["spambase"]["initial_recipe"]["dropout"] = 0.3
        protocol.write_text(json.dumps(edited))
        with pytest.raises(ValueError, match="different settings"):
            autopilot(protocol, data_root, out, gpus=[], env_file=key_file)
        # And a seed cannot silently disappear.
        edited["tasks"]["spambase"]["initial_recipe"]["dropout"] = 0.1
        edited["tasks"]["spambase"]["seeds"] = [2]
        protocol.write_text(json.dumps(edited))
        with pytest.raises(ValueError, match="cannot be removed"):
            autopilot(protocol, data_root, out, gpus=[], env_file=key_file)


def test_autopilot_stops_before_any_training_if_the_api_refuses(tmp_path, data_root, monkeypatch):
    from rsi_ml.autopilot import autopilot

    monkeypatch.setenv("RSI_API_MAX_RETRIES", "0")
    key_file = tmp_path / "api.env"
    key_file.write_text("OPENROUTER_API_KEY=sk-fake\n")
    out = tmp_path / "study"
    with FakeLLM() as llm:
        protocol = _autopilot_protocol(tmp_path / "p.json", llm.url)
        llm.failures_to_inject = 100
        stopped = autopilot(protocol, data_root, out, gpus=[], env_file=key_file)
    assert stopped["status"] == "stopped_api_preflight"
    assert len(llm.bodies) == 3  # Bounded attempts, then a clear stop.
    assert not (out / "runs").exists() and not (out / "api_preflight.json").exists()


def test_autopilot_checks_for_a_key_before_doing_anything(tmp_path, data_root, monkeypatch):
    from rsi_ml.autopilot import autopilot

    monkeypatch.delenv("RSI_API_KEY", raising=False)
    protocol = _autopilot_protocol(tmp_path / "p.json", "http://127.0.0.1:9/api/v1")
    out = tmp_path / "study"
    with pytest.raises(OSError):
        autopilot(protocol, data_root, out, gpus=[], env_file=tmp_path / "missing.env")
    with pytest.raises(ValueError, match="RSI_API_KEY"):
        autopilot(protocol, data_root, out, gpus=[])
    assert not out.exists()


def _settling(monkeypatch, outcomes):
    """Run _run_until_settled against scripted pass results, recording GPU order and waits."""
    from rsi_ml.autopilot import _run_until_settled

    orders, waits, outcomes = [], [], iter(outcomes)

    def stage(study_dir, name, recover, gpus, **options):
        assert recover
        orders.append(list(gpus))
        completed, failed = next(outcomes)
        return {"completed": completed, "failed": ["job"] * failed, "held": []}

    monkeypatch.setattr("rsi_ml.autopilot.run_stage", stage)
    rules = {"max_passes": 5, "seconds_between_passes": 600}
    result = _run_until_settled(
        "study", "agents", rules, waits.append, lambda *a, **k: None, gpus=["0", "1", "2"]
    )
    return result, orders, waits


def test_passes_rotate_gpus_and_only_stalled_passes_count(monkeypatch):
    # Steady progress never exhausts the limit, however many separate faults occur.
    result, orders, waits = _settling(monkeypatch, [(1, 1)] * 7 + [(1, 0)])
    assert not result["failed"] and len(orders) == 8
    assert orders[:4] == [["0", "1", "2"], ["1", "2", "0"], ["2", "0", "1"], ["0", "1", "2"]]
    assert waits == [600] * 7
    # With no progress it gives up after five passes, waiting longer each time, capped at 1h.
    result, orders, waits = _settling(monkeypatch, [(0, 1)] * 9)
    assert result["failed"] and len(orders) == 5
    assert waits == [600, 1200, 2400, 3600]


def test_a_provider_outage_does_not_retire_a_healthy_slot(tmp_path, data_root, monkeypatch):
    from rsi_ml.agents import PROVIDER_UNAVAILABLE_EXIT_CODE
    from rsi_ml.study import run_stage

    study = tmp_path / "study"
    build_study(tiny_protocol(tmp_path / "p.json", None, targets_set=False), data_root, study)
    monkeypatch.setattr(
        "rsi_ml.study.job_command",
        lambda *a, **k: [
            sys.executable,
            "-c",
            f"raise SystemExit({PROVIDER_UNAVAILABLE_EXIT_CODE})",
        ],
    )
    result = run_stage(study, "baselines", gpus=[], cpu_slots=1)
    assert len(result["failed"]) == 4
    jobs = read_json(study / "status/baselines.json")["jobs"]
    assert all(job["exit_code"] == PROVIDER_UNAVAILABLE_EXIT_CODE for job in jobs.values())


def test_jobs_that_failed_before_go_last_so_they_cannot_starve_others(
    tmp_path, data_root, monkeypatch
):
    from rsi_ml.study import run_stage

    study = tmp_path / "study"
    build_study(tiny_protocol(tmp_path / "p.json", None, targets_set=False), data_root, study)

    def command(job, *args, **kwargs):
        code = 3 if job.task_id in {"spambase", "mrpc"} else 0
        return [sys.executable, "-c", f"raise SystemExit({code})"]

    monkeypatch.setattr("rsi_ml.study.job_command", command)
    run_stage(study, "baselines", gpus=[], cpu_slots=1)  # Two failures retire the only slot.
    run_stage(study, "baselines", gpus=[], cpu_slots=1)
    jobs = read_json(study / "status/baselines.json")["jobs"]
    assert jobs["sst2.seed-1.baseline"]["state"] == "completed"
    assert jobs["cifar100.seed-1.baseline"]["state"] == "completed"


def test_autopilot_refuses_a_protocol_nobody_approved(tmp_path, data_root):
    from rsi_ml.autopilot import autopilot

    with pytest.raises(ValueError, match="not approved"):
        autopilot(PROPOSED, data_root, tmp_path / "study", gpus=[])
    assert not (tmp_path / "study").exists()


def test_a_dead_slot_retires_instead_of_failing_the_whole_queue(tmp_path, data_root, monkeypatch):
    from rsi_ml.study import run_stage

    study = tmp_path / "study"
    build_study(tiny_protocol(tmp_path / "p.json", None, targets_set=False), data_root, study)
    monkeypatch.setattr(
        "rsi_ml.study.job_command",
        lambda *args, **kwargs: [sys.executable, "-c", "raise SystemExit(3)"],
    )
    result = run_stage(study, "baselines", gpus=[], cpu_slots=1)
    assert len(result["failed"]) == 4
    jobs = read_json(study / "status/baselines.json")["jobs"]
    assert sum("exit_code" in job for job in jobs.values()) == 2  # Then the slot retired.
    assert sum(job.get("error", "").startswith("No healthy") for job in jobs.values()) == 2


def test_unknown_api_cost_still_counts_against_the_cap(tmp_path):
    from rsi_ml.autopilot import UNKNOWN_COST_PER_CALL_USD, api_spend

    write_json(tmp_path / "runs/t/seed-1/A/state.json", {"usage": {"calls": 3, "cost_usd": None}})
    write_json(tmp_path / "runs/t/seed-1/B/state.json", {"usage": {"calls": 2, "cost_usd": 0.5}})
    write_json(tmp_path / "runs/t/seed-1/C/state.json", {"usage": {"calls": 0, "cost_usd": None}})
    assert api_spend(tmp_path) == pytest.approx(0.5 + 3 * UNKNOWN_COST_PER_CALL_USD)


def test_failed_baseline_is_retried_not_treated_as_done(tmp_path):
    from rsi_ml.study import Job

    out = tmp_path / "baseline"
    out.mkdir()
    job = Job("x", "baselines", "t", 1, "baseline", "cpu", tmp_path / "task.json", out)
    write_json(out / "baseline.json", {"result": {"status": "timeout"}})
    assert job_action(job) == "restart"
    write_json(out / "baseline.json", {"result": {"status": "completed"}})
    assert job_action(job) == "done"
