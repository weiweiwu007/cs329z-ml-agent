# Connecting an API later

The offline demo uses `MockBackend`. It returns scripted plans and reviews while
the training tool computes actual metrics. Its candidate order is identical for
the original and revised policy, so it cannot establish agent self-improvement.

The API backend uses DSPy signatures and `Predict`. In A–D, only the proposing
agent and optional reviewer make decision calls:

| Role | Output |
|---|---|
| Proposing agent | Observation, hypothesis, predicted effect, executable recipe, evidence IDs; diagnostic requests allowed only in B/D |
| Trusted builder | Validates and implements the recipe; **no API call** |
| Reviewer (C/D) | Interpretation, next step; diagnostic requests allowed only in D |
| Workflow improver | Candidate planner instructions, rationale, development-run IDs |

A/C cannot read detailed error reports. The tool is absent from their context,
and attempted requests are rejected. B/D receive reports automatically after
completed experiments; this works with `max_diagnostic_requests: 0`. A positive
allowance permits B/D to revisit reports; it never grants A/C access.

Install the optional `api` dependency group, then supply these environment
variables in the terminal where the CLI runs:

- `RSI_MODEL`: the provider/model identifier you selected, using DSPy's naming.
- `RSI_API_KEY`: the corresponding credential.
- `RSI_API_BASE`: optional custom HTTP(S) endpoint. Credentials and query strings
  are rejected in this URL; use `RSI_API_KEY` for authentication.

No provider or model is selected automatically. The backend does not read a `.env`
file by itself. Do not put credentials in task specifications, policy files, or
experiment records. Follow the main README for the CLI invocation.

## Bounds and accounting

- Each decision consumes one call from the limit. Parser repairs do not trigger another
  call, and a malformed role response is a rejected attempt, never retried.
- Provider faults are not agent decisions. Rate limits, server errors, timeouts, and
  connection failures retry with backoff (`RSI_API_MAX_RETRIES`, default 4), counted in
  `transient_retries`, not the call limit. When retries run out, or on authentication,
  billing, or invalid-request errors, the run stops at a resumable point and consumes no
  trial. Fix the cause, then use `study-run --recover`.
- The context character limit is checked again after DSPy formats its full prompt.
  An oversized prompt is rejected before any call; constraints are never silently
  truncated by the backend.
- The requested output limit is passed as `max_tokens` to the selected model.
  Provider compatibility still requires a live check when credentials are supplied.
- `RSI_API_TIMEOUT` defaults to 60 seconds, and the study protocol sets it explicitly.
  `RSI_REASONING_EFFORT` is optional. Hidden reasoning shares `max_output_tokens`, so a
  response cut off at the limit is reported as such. The constructor also accepts an
  explicit sampling temperature. Otherwise, provider sampling defaults apply and that
  fact is recorded in backend metadata.
- API cost is the amount OpenRouter reports charging for each response, falling back to
  DSPy's estimate only if the provider sends none. Unknown tokens are flagged; unknown
  dollar cost is `null`, not zero. Call and token limits do not guarantee a dollar budget.
- Raw provider exceptions, credentials, and endpoint URLs are not written to run
  records by this backend. The caller records the typed inputs and outputs.

The same model connection is used for every role in a run. Changing the model or
sampling configuration creates a different comparison condition. The workflow
improver can propose only planning-instruction changes; it cannot modify the
grader, data splits, compute budget, underlying language model, or acceptance rule.

## Implemented optimization boundary

V1 uses a versioned reflection-and-evaluation loop around the planner. It uses
DSPy for typed inference; it does **not** claim to implement GEPA. A candidate
instruction revision is not an improvement until the independent comparison
accepts it. Tests exercise the DSPy path with an injected offline `DummyLM` and
make no paid requests.

The adapter uses DSPy's canonical `lm15.Request` interface, tested with DSPy 3.4.1.

Research run commands require both a finalized task/suite and `--execute`.
The default `compare-designs` command is read-only and does not construct a backend.

## OpenRouter

Use `RSI_MODEL=openrouter/<provider>/<model>`, for example `openrouter/openai/gpt-6-sol`.
The study runner reads only the key from `--env-file`, accepting `OPENROUTER_API_KEY` or
`RSI_API_KEY`. Training workers never inherit it. The wire format, retries, and failure
handling are tested against a local server. On 2026-11-04, five real `gpt-6-luna` calls
parsed correctly and all four recipes were valid for the proposed search spaces.

`rsi_ml api-check --env-file FILE --model MODEL` shows remaining credit and the model's price.
Adding `--live --execute` makes one real call.
