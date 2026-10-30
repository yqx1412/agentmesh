# agentmesh

Multi-agent system over A2A: an orchestrator plus specialist agents.

**Question this project answers:** When do specialized agents working together over A2A beat a single agent, and what does it cost?

Part of the local AI agent ecosystem; see `../ROADMAP.md`.

## M1: one agent behind A2A

`agentmesh serve` runs one AgentOS agent as an A2A server (`a2a-sdk`, JSON-RPC
binding). The agent is described by a TOML spec (`agents/general.toml`): its model,
tool permissions, workspace and the skills its Agent Card advertises.

```powershell
uv run agentmesh serve agents/general.toml          # http://127.0.0.1:9101
uv run agentmesh card http://127.0.0.1:9101         # print the Agent Card
uv run agentmesh ask  http://127.0.0.1:9101 "What is 17.5% of 2,348, rounded to two decimals?"
```

```text
found 'General Agent' at http://127.0.0.1:9101, skills: workspace_files, calculation
  [submitted] task 27d79368-...
  [working] Starting the agent.
  [working] step 1: calling calculator
  [working] calculator: ok
  [working] step 2: writing the answer
[completed] {"completion_tokens": 75, "prompt_tokens": 884, "tool_errors": 0, "stop_reason": "final_answer", "tool_calls": 1, "steps": 2}
Final Answer: 410.90
```

- **Lifecycle:** every task goes submitted -> working -> completed or failed. Each
  AgentOS step is streamed as a `working` status update. An agent error (for
  example a model Ollama doesn't have) ends the task as `failed` with the error
  text, and `ask` exits with code 3.
- **Workspace:** file tools are confined to the spec's `workspace` folder
  (git-ignored).
- **No authentication (until M5):** the server binds to `127.0.0.1` unless the spec
  sets `host`. Don't expose it on a network. M5 adds logins; see below.
- `--trace-dir` writes an AgentOS trace per task; `ask --no-stream` sends one
  blocking request instead of streaming.

## M2: four specialist agents

`agentmesh up` starts four specialists, each as its own A2A process, and waits
until every Agent Card is served. Ctrl+C stops them all. An agent that exits is
restarted on its own (2 s backoff, doubling, up to `--max-restarts`, default 3)
while the others keep running, so one dead agent never takes the fleet down.

| Agent | Port | Model | Tools |
|---|---|---|---|
| Research | 9111 | qwen3:8b | `web_search`, `fetch_url` |
| Analyst | 9112 | qwen3:14b | `run_python` (A7 sandbox), `calculator`, workspace files |
| Verifier | 9113 | qwen3:14b | `fetch_url` only, no search |
| Knowledge | 9114 | qwen3:8b | DomainGraph over MCP (the only agent that writes) |

```powershell
uv run agentmesh up                                  # logs in runs/fleet/<timestamp>
uv run agentmesh ask http://127.0.0.1:9114 "What did lecture 7 say about radix sort?"
```

- **Specs:** each agent is `agents/<name>.toml` (role prompt, port, model, card
  skills) plus `agents/<name>.agentos.toml` (its tools and MCP servers). An agent
  gets only the tools its spec lists.
- **Rules enforced in code, not just the prompt:** `require_before_answer` sends
  an answer back until the listed tools have been called (Research must fetch 2
  pages, the Verifier 1, the Analyst must run Python once). The Verifier's quotes
  are checked against the pages it fetched, and a quote no page contains is sent
  back. qwen3 ignored both rules when they were only in the prompt.
- **Web tools** (`agentmesh.webtools`, an MCP server): `fetch_url` only reaches
  public addresses. Every redirect hop is resolved and refused if it points at
  loopback, private or link-local space, so a page can't steer an agent at
  Neo4j, Ollama or other local services. DNS rebinding is not covered.
- **Knowledge** needs the DomainGraph Neo4j and Ollama running.

## Watching the fleet live

`agentmesh watch` serves a page on `http://127.0.0.1:9130/` that follows every
agent's tasks as they run.

```powershell
uv run agentmesh up --trace-dir runs/fleet-traces    # one trace folder per agent
uv run agentmesh watch                               # then open http://127.0.0.1:9130/
```

- **One lane per agent**, showing whether it's up and busy. Each task is a card
  that moves through submitted, working, then completed or failed, with its
  elapsed time, tokens, tool calls, tool output and final answer.
- **A2A journal:** `serve --trace-dir` now writes `a2a.jsonl` beside the traces,
  one line per task state change, with the A2A task and context IDs. Traces from
  before the journal still show, with their state worked out from the trace.
- **Any client:** it reads the trace folders rather than acting as an A2A
  client, so it shows tasks from `ask`, another agent or a raw JSON-RPC call.
- **No login:** it binds to `127.0.0.1` only. The page shows tool output and
  fetched web pages, so don't expose it.

## M3: the orchestrator

`agentmesh orchestrate` splits one task across the fleet. It finds the agents
from their Agent Cards, plans which skills to use, runs the steps over A2A and
combines the results into one answer.

```powershell
uv run agentmesh up --trace-dir runs/fleet-traces
uv run agentmesh orchestrate "Research local LLM inference and update my knowledge base" --allow-writes
uv run agentmesh orchestrate "..." --plan-only       # print the plan, send nothing
```

- **Discovery:** the orchestrator reads each agent's card (by default the URLs in
  `agents/*.toml`, or `--agent URL`, repeated). The skills on the cards are the
  only skills a plan may use.
- **Planning:** qwen3:14b writes a JSON plan of at most 6 steps, constrained to
  a schema whose skill IDs come from the cards. A plan naming an unknown skill,
  an unknown step or a cycle is rejected.
- **Running:** a step can use earlier answers with `{{s1}}`, and steps whose
  inputs are ready run in parallel. A failed step skips every step that needs
  it, and the rest still run. Each step has a timeout (`--step-timeout`,
  default 900 s).
- **Tracing:** every delegated task carries its orchestration ID and step in the
  A2A message metadata, and each agent's journal records them. The orchestrator
  writes its own trace to `runs/fleet-traces/orchestrator/`, so `watch` shows
  it as the first lane and links it to its child tasks.

## M4: reliability

The orchestrator now survives an agent that fails, hangs or dies mid-task.

```powershell
uv run agentmesh orchestrate "..." --retries 2 --step-timeout 300
uv run agentmesh orchestrate "..." --no-fallback     # never switch agents
uv run agentmesh orchestrate "..." --push            # follow steps by webhook
```

- **Retries:** a failed attempt is retried on the same agent (`--retries`,
  default 1).
- **Fallback:** when the retries run out, the step goes to the next agent whose
  card lists the same skill, in discovery order. `--no-fallback` turns this off.
- **Timeouts cancel:** `--step-timeout` now bounds each attempt, and a timed-out
  attempt's remote task is cancelled over A2A, so the agent stops working on it.
  Cancelling the whole orchestration cancels every remote task still running, including
  one whose agent hasn't yet reported its task id: the step waits up to 5 s for the id
  before giving up.
- **Push notifications:** agents advertise A2A push support. With `--push` the
  orchestrator gives each task a webhook on `127.0.0.1` instead of holding a
  stream open. Agents accept webhooks only on loopback addresses, so an agent
  can't be used to POST to other hosts.

## M5: security

Agents now need a login, each skill says who may use it, and the Knowledge agent
writes only what the Verifier signed off on.

```powershell
uv run agentmesh auth init                 # secrets/ (git-ignored): tokens + Verifier key
uv run agentmesh up --trace-dir runs/fleet-traces          # logins on by default
uv run agentmesh ask http://127.0.0.1:9114 "What is radix sort?" --skill knowledge_lookup
uv run agentmesh orchestrate "Research local LLM inference and update my knowledge base" --allow-writes
```

- **Logins:** every caller (principal) has its own random bearer token in
  `secrets/tokens/<name>.token`; `ask` logs in as `user`, `orchestrate` as
  `orchestrator` (`--as` changes it). Agents keep only each token's SHA-256 in
  `secrets/principals.toml`. A JSON-RPC request without a known token gets HTTP 401;
  the Agent Card stays public, and it advertises the bearer scheme. Tasks are scoped to
  the principal that created them, so one caller can't read or cancel another's task.
  `serve`/`up` refuse to start without the auth folder unless given `--insecure`.
- **Per-skill access:** a request names the skill it wants (`ask --skill`; the
  orchestrator sends each step's skill). A skill in a spec lists its `callers` and the
  `tools` a task under it may use. A caller not on the list is `rejected`. An agent
  with several skills rejects a request that doesn't name one.
- **Writes only from the Verifier:** the Verifier signs every completed verdict
  (input + answer) with an Ed25519 key only it holds (`attest_as = "verifier"`). The
  orchestrator passes a step's signed results on to the steps that depend on it, but
  can't produce one itself. Knowledge's `knowledge_write` skill
  (`require_attestation = "verifier"`) is `rejected` without a valid signature, and a
  forged, altered or stale one (over an hour old) is rejected too. With one, each
  `add_fact` call must cite a source URL that a signed verdict marked SUPPORTED, at most
  once per supported claim; anything else comes back to the model as an error.
  `knowledge_lookup` gets no write tool at all.
- **What was really stored:** `add_fact` returns `"new": false` for a fact that was
  already stored, and small models still reported those as "successfully stored". Now
  the gate tells the model `ALREADY STORED` in words, and Knowledge's answer ends with a
  write record built from the tool results (`id 314: already stored, not added again`).
  The orchestrator appends every step's record to the final answer, so it holds even
  when a summary gets it wrong. Live, storing the same fact again 4 times gave the right
  record each time; with the matching summary rule, 3 of 3 summaries said so too.
  Duplicates are still matched on exact (normalised) text, so a reworded fact is stored
  as new.

| Who asks Knowledge to write | Result |
|---|---|
| `user` | rejected: only `orchestrator` may use `knowledge_write` |
| `orchestrator`, no Verifier signature | rejected |
| `orchestrator`, forged or altered signature | rejected |
| `orchestrator`, valid signature | runs; `add_fact` refuses sources the verdict didn't support |

**Limits:** tokens travel over plain HTTP, so the agents still bind to `127.0.0.1`.
All secrets sit in one folder that every agent process can read. What keeps an
agent's model away from it is the AgentOS tool confinement: file tools stay in the
workspace, and the `run_python` sandbox refuses file access outside it. That
contains a confused model, not a hostile one. The next step would be separate OS
users or containers per agent. The source check also matches statements now: a write
must cite a URL the Verifier marked SUPPORTED *and* say what it checked with that URL
(the claim it was asked, with the quote it found). Every number in the statement must be
there, and at least 60% of its content words, so rewording passes but another claim
under the same URL doesn't. Over 50 saved Verifier verdicts, no unsupported claim that
shared a URL with a supported one would have passed. It's a word-level match, not
entailment: a statement built from a verified claim's own words can still twist its
meaning.

## M6: one agent vs. the whole fleet

The headline result of AgentMesh: the same 16 tasks, 3 times each, on a single AgentOS
agent with every specialist's tools (`agents/solo.toml`, port 9115) and on the fleet
through the orchestrator. Full write-up:
[`benchmarks/results/m6-single-vs-multi.md`](benchmarks/results/m6-single-vs-multi.md).

```powershell
uv run agentmesh up --trace-dir runs/fleet-traces
uv run agentmesh serve agents/solo.toml --trace-dir runs/fleet-traces/solo
uv run agentmesh eval run --out runs/m6/full --repeats 3 --trace-root runs/fleet-traces   # resumable; match the fleet's --trace-dir
uv run agentmesh eval report runs/m6/full
```

| | single | multi |
|---|---|---|
| Runs passed | 35/48 (73%) | 39/48 (81%) |
| Passed, every cited page fetched | 23/48 (48%) | 37/48 (77%) |
| Median time per run | 27 s | 45 s |
| Mean tokens per run | 4,707 | 16,032 |
| Composite (two skills chained) | 6/9 | 2/9 |

- **The fleet helps** where a specialist's rule is enforced in code: the single agent
  cited pages it never fetched in all 14 of its citing answers, and called true claims
  false without fetching them. The fleet also picked the right tool more reliably
  (web vs. lecture notes).
- **It hurts** on chained tasks: each step sees only the planner's rewrite of the
  task, so needed inputs went missing and one question got inverted. It also costs
  about 3.4x the tokens and 1.7x the time, and the planner added a knowledge-base write
  nobody asked for.
- **Scoring** is deterministic on the final answer (phrases, numbers, regexes, lecture
  timestamps), plus a citation check against the pages each system's traces show it
  fetched. Tokens are summed over every model call, including replies the answer guard
  sent back.

### After M6: closing the gaps M6 found

M6 exposed two kinds of fleet weakness, each closed by code (with tests), not prompt
tuning. The full blow-by-blow — every fix, with its live and 10-run eval numbers — is in
the [write-up](benchmarks/results/m6-single-vs-multi.md); the short version:

- **Chained tasks lost information at the hand-off** (composite 2/9). Fixes: every step
  also gets the user's task, a step may fail with `CANNOT: <reason>` so retries and
  fallback run, the planner re-plans once on a missing input, writes are hidden unless
  `--allow-writes`, and the final reply must restate the values the task asked to look up.
  Composite rose to 9/9 on these tasks (about 90-95% on 10-run evals; the 16 tasks were
  the tuning set, so 9/9 overstates new chained tasks).
- **Citations and numbers were decoration.** The single agent cited pages it never
  fetched in every citing answer. Research now checks both in code: a cited URL no
  `fetch_url` returned, or a number on no fetched page, is sent back
  (`check_citations`/`check_numbers` in `agents/research.toml`). Supporting fetch fixes:
  PDFs are read, a Hugging Face card comes with its `config.json`, model-card front matter
  (the license) and Mintlify `data-as="p"` paragraphs survive extraction, and search
  returns the model's own page first.
- **The guard survives compaction and finite nudges.** It reconstructs "what the task
  gave" from the task message and the tool results it saw, never the compaction summary
  (so a folded-in search hit can't pass as a task URL); and when the nudge budget is spent
  still citing an unfetched page, that citation is stripped rather than shipped.
- **The orchestrator and the single agent are guarded too**, and the eval scores
  citations honestly — `citations_unscoreable` when the fetch traces aren't found (match
  the eval's `--trace-root` to the fleet's `--trace-dir`), and a comma/bracket-clipped
  URL matches its fetched entry. A matched-trace-root rerun passed the fleet 48/48 with
  every cited page fetched and all citations scoreable.

Three full reruns (2026-10-19/23/24) and a matched-trace-root rerun (2026-10-27) track
these fixes; see the write-up for the tables and caveats.

**Side-effect tasks** (`benchmarks/m6/tasks-side-effects.yaml`, kept apart from the 16 M6
tasks) test what a run *does*, not only what it answers: knowledge-base writes (read from
the Knowledge agent's trace, not the answer) and cancels (every remote task must end
`canceled` with no more than one model call after the cancel). Run with
`agentmesh eval run --tasks benchmarks/m6/tasks-side-effects.yaml`.

## S1: end-to-end demo

The showcase milestone: given the 10-lecture MIT 6.006 algorithms corpus in DomainGraph
(600 concepts, 3,293 facts, 2,214 relations), find concepts covered only by lecture notes
with no web sources, research those gaps, verify the findings, and write the supported
facts back through the full M5 security chain.

```powershell
# Prereqs: fleet up (uv run agentmesh up --trace-dir runs/fleet-traces),
#          Neo4j and Ollama running
uv run python runs/s1_run.py
```

The orchestrator plans and runs a four-step pipeline — Knowledge gap-find → Research →
Verifier → Knowledge write — from a single natural-language task. No S1-specific agent
code was written. The clean demo pass stored 4 new AVL tree facts into DomainGraph
(IDs 323–326: balance factor definition, ±2 triggers rotation, four rotation variants
LL/RR/LR/RL, height formula with golden-ratio bound), all gated through the M5
Verifier-attestation write check.

**Running S1 surfaced two auth.py parse gaps** (PR #40) that the M6 benchmark's uniform
verdict format never hit: the live Verifier wraps source URLs as `[<url>]` (angle
brackets) and double-numbers its output lines (`1. 1. SUPPORTED`). Both were fixed before
the clean pass; 289 tests pass, ruff clean. Full write-up:
[`benchmarks/results/m6-single-vs-multi.md`](benchmarks/results/m6-single-vs-multi.md#s1-end-to-end-demo-2026-10-29).
Pipeline trace visualization: [yqx1412.github.io/agent-ecosystem](https://yqx1412.github.io/agent-ecosystem/).

## Development

```powershell
uv sync
uv run pre-commit install
uv run pytest
uv run ruff check .
```
