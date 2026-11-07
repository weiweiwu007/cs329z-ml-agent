"""Random search, the conventional reference arm for the agent designs.

Under a constraints task the agents get no menu, but random search still needs something to
draw from. Those ranges belong to this comparison method alone and are never shown to agents.
Draws that break the parameter ceiling or other limits are redrawn, so this arm respects the
same constraints as the agents.
"""

import math
import random

from .agents import MockBackend
from .schemas import Plan, Recipe

MAX_DRAWS = 500
# Optimizer steps relative to the starting recipe. Time limits allow 4x the baseline, so a draw
# past this would likely be cut off and could end the run after a couple of attempts. Agents
# avoid that by reading timings, so a reference arm that cannot should not be sunk by it.
MAX_STEPS_VS_BASELINE = 2.5


def draw(rng: random.Random, spec: dict):
    ((kind, value),) = spec.items()
    if kind == "choice":
        return rng.choice(value)
    if kind == "uniform":
        return rng.uniform(*value)
    if kind == "log_uniform":
        return math.exp(rng.uniform(math.log(value[0]), math.log(value[1])))
    if kind == "int":
        return rng.randint(*value)
    if kind == "int_log":
        return round(math.exp(rng.uniform(math.log(value[0]), math.log(value[1]))))
    if kind == "layers":
        depth = rng.randint(*value["depth"])
        return [draw(rng, {"int_log": value["width"]}) for _ in range(depth)]
    if kind == "stage_widths":  # The ResNet pattern: each stage doubles the first width.
        base = draw(rng, {"int_log": value["base"]})
        return [base * 2**stage for stage in range(value["stages"])]
    if kind == "int_list":
        return [rng.randint(*value["range"]) for _ in range(value["length"])]
    raise ValueError(f"Unknown sampling rule: {kind}")


class RandomSearchBackend(MockBackend):
    name = "random"

    def __init__(self):
        super().__init__()
        self._validator = None
        self.metadata = {
            "backend": "random",
            "network": False,
            "algorithm": "independent draws from the ranges or choices declared for this arm",
        }

    def bind_validator(self, validator) -> None:
        self._validator = validator

    def _sample(self, rng, initial: dict, ranges: dict) -> Recipe:
        ranges = dict(ranges)
        # A study that gives attempts more time than usual declares a matching cap.
        step_limit = ranges.pop("max_steps_vs_baseline", MAX_STEPS_VS_BASELINE)
        for _ in range(MAX_DRAWS):
            values = dict(initial)
            # Optimizer first, because conventional learning rates depend on it.
            for field, spec in sorted(ranges.items(), key=lambda i: (i[0] != "optimizer", i[0])):
                if "by_optimizer" in spec:
                    spec = {"log_uniform": spec["by_optimizer"][values["optimizer"]]}
                values[field] = draw(rng, spec)
            steps = (values["epochs"] / initial["epochs"]) * (
                initial["batch_size"] / values["batch_size"]
            )
            if steps > step_limit:
                continue
            if values.get("optimizer") != "sgd":
                values["momentum"] = 0.0
            try:
                recipe = Recipe.model_validate(values)
                if self._validator:
                    self._validator(recipe)
            except ValueError:
                continue
            return recipe
        raise ValueError("Random search found no recipe within the limits.")

    def _plan(self, context):
        task = context["task"]
        count = context.get("history_count", len(context["records"]))
        rng = random.Random(task["seed"] * 1000003 + count)
        ranges = context.get("random_search")
        if ranges:
            recipe = self._sample(rng, task["initial_recipe"], ranges)
        else:
            values = dict(task["initial_recipe"])
            for field, choices in sorted(task.get("search_space", {}).items()):
                values[field] = rng.choice(choices)
            recipe = Recipe.model_validate(values)
        return Plan(
            observation=f"{count} counted attempts.",
            hypothesis="Conventional random search; no model-based explanation.",
            predicted_effect="Unknown; evaluate the sampled recipe.",
            experiment="Random draw from this arm's declared ranges.",
            recipe=recipe,
        )
