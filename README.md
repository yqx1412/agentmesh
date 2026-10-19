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
  Cancelling the whole orchestration cancels every remote task still running.
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
users or containers per agent. The source check matches URLs, not statements: a claim
the Verifier never saw can still be stored under a URL that supported some other
claim, as long as the per-claim count allows it.

## M6: one agent vs. the whole fleet

The headline result of AgentMesh: the same 16 tasks, 3 times each, on a single AgentOS
agent with every specialist's tools (`agents/solo.toml`, port 9115) and on the fleet
through the orchestrator. Full write-up:
[`benchmarks/results/m6-single-vs-multi.md`](benchmarks/results/m6-single-vs-multi.md).

```powershell
uv run agentmesh up --trace-dir runs/fleet-traces
uv run agentmesh serve agents/solo.toml --trace-dir runs/fleet-traces/solo
uv run agentmesh eval run --out runs/m6/full --repeats 3   # resumable
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

### After M6: chained-task fixes

Four changes to the orchestrator for the fleet weaknesses M6 found:

- **Every step also receives the user's task**, after the planner's input and marked as
  context only, so a step isn't limited to the planner's rewording. The planner is also
  told to ask for every input a later computation needs, in the task's direction.
- **`CANNOT: <reason>` fails the step.** Every step is told it may reply with that line
  when an input is missing or a tool failed. The orchestrator then treats the attempt
  as failed, so retries and fallback run and dependent steps are skipped instead of
  receiving an apology as data.
- **Writes only when asked for.** Skills a card tags `write` (Knowledge's
  `knowledge_write`) are hidden from the planner unless `orchestrate --allow-writes`
  is given, and the combiner is told when they were. M5's signature check stops
  unverified writes; this stops unrequested ones.
- **Re-plan on a missing input.** When a step still fails with `CANNOT` after its
  retries, the planner is called once more (`--replans`, default 1, 0 turns it off). It
  sees every step so far with its result or `CANNOT` reason, and writes only new steps:
  they may use completed results with `{{sN}}`, so finished work isn't repeated, may not
  reuse earlier ids, and must use the failed step's skill again (a re-plan that only
  fetched the missing value once left the combiner to do the arithmetic, wrongly). The
  steps they replace are marked `replaced`.

Rerun of the 3 composite tasks on the fleet, 3 times each: **6/9, up from 2/9**. The
context-window ratio now passes 3/3 (was 0/3) and the lecture speed-up 3/3 (was 2/3).
The Qwen3-8B KV-cache task failed 0/3 before re-planning was added: the plan didn't ask
Research for the context length, and the Analyst replied `CANNOT` instead of guessing
one. With re-planning, all 3 runs computed the right 4.5 GiB (Research -> Analyst
`CANNOT` -> Research for the context length -> Analyst), and 1 of 3 passed; the other
two final answers left out the looked-up layer count the task's check also requires.
The final summary is now told to state every value the task asked to look up, with its
source. Rerun 3 times: **2/3 passed**, and all 3 answers listed the layer count. The
failure was a lookup error, not the summary: Research gave the 32 attention heads where
the task asked for the KV heads (8), and the calculation used the 131,072-token YaRN
context although Research also reported the 32,768 native one, giving 72 GiB.
Research is now told to report each value under the exact name the task uses, never a
related one (KV heads are not attention heads, native context is not YaRN-extended), to
list every related value a page gives with its label, and to prefer the model's own card
over a library's default config. The planner is told to keep the task's qualifiers word
for word. Rerun 3 times: **3/3 passed**, each with 8 KV heads and 32,768 tokens (4.5 GiB),
and the five research tasks still passed 5/5 with no unfetched citations.
The planner still never asked for the context length in its first plan, so every KV run
needed a re-plan. Its JSON reply now starts with an `inputs` list (every value the answer
depends on, and which step gets it, including values the task names only as a condition,
like "at its full native context length"); constrained decoding writes it before the
steps. The first live try showed a new failure: the planner then wrote the formula into
the Analyst's step, without the x2 for keys and values, and all 3 runs got it wrong. A
step input that spells out a formula (`*`, `^`, "formula") is now sent back unless the
user's task contains it. Rerun 3 times: every plan asked for the context length up front,
no run re-planned (about 100 s per run, down from about 350 s), and **2/3 passed**. The
failure took the context length from `config.json`'s `max_position_embeddings` (40,960)
instead of the card's 32,768 native tokens.
Research is now told that a model's native context length is the one its card states
(report config.json's `max_position_embeddings` beside it when they differ), to report
every case of a value that depends on a condition such as VRAM, and to rank official docs
above blogs. The ratio task now names the VRAM tier: Ollama's docs changed, and
`/context-length` gives 4k, 32k or 256k by VRAM while `/faq` still says 4096. Rerun: KV
cache **3/3** and ratio **5/6** (all six had 4,096 and 32,768 right; the failure ended
with Research's own "RATIO: 0.25", which it wrote because the user's task, shown to every
step for context, asks for that line), and the five research tasks 5/5.
Each step is now told that the task's format instructions (a line to end with, a
template) are for the final reply only, and the final summary adds the requested format
to its full reply, filled in from the step that computed the value. In 6 ratio runs none
of the 12 Research steps wrote a `RATIO:` line. A first wording of the summary rule made
it reply with only "RATIO: 8.0" (ratio 1/3); after rewording, ratio **3/3** with 4,096
and 32,768 stated. KV cache 5/6 across both rounds (the failure: the Analyst left out the
x2 for keys and values and got 2.25 GiB), lecture speed-up 3/3 (first round only).
The Analyst is now told to give every factor of a formula its own named variable, never a
bare number, and that the KV cache's two factors of 2 at fp16 (key and value, 2 bytes per
element) are separate. Rerun: all three composite tasks **3/3** (9/9; every KV script
had both `kv_tensors` and `bytes_per_element`, 4.5 GiB each time), and the four analysis
tasks 4/4.
No plan included a write.
The planner is now told to add a verification step only when the task asks to check or
verify something, or before a knowledge-base write; asking for citations doesn't need one
(the M6 rerun's planner added one to every Qwen3-8B license lookup). In 3 planner-only
passes over all 16 tasks, no lookup got a verification step, the claim-check task kept
its own, and "research ... and update my knowledge base" with `--allow-writes` still went
Research -> Verifier -> Knowledge. The license plan now asks two Research steps in
parallel, one per question. Live: the license task 4/4 (47-56 s, was a 67 s median),
and the other research tasks and the claim check 5/5.
The planner is now told to give each skill one step that asks for everything the task
needs from it, and `parse_plan` sends back a plan where two steps share a skill and
neither needs the other's result. In 3 planner-only passes over the 16 tasks every plan
had one step per skill (the check never had to fire): the license lookup, the
context-window ratio and the lecture speed-up each lost a step. Live, 3 times each: all
12 runs passed. The license task took 44-54 s and 27-44k tokens (was 47-56 s, 40-51k),
the KV cache was unchanged, and the lecture speed-up took 87-101 s, a little slower. The
ratio task got worse: in 2 of 3 runs the merged Research step ran one search for both
values, fetched only Ollama pages and couldn't give Qwen3-8B's context length, so the
Analyst replied `CANNOT` and the re-plan fetched it (219-226 s, was a 91 s median).
Research is now told to search and fetch separately for each thing it is asked about,
and to check every requested value has a fetched page behind it before answering.
Rerun: the ratio task **3/3** with no re-plan (78-89 s, 21-22k tokens; every Research
step searched Ollama and Qwen3-8B separately), the license task 3/3 (37-58 s; one search,
since both values are about the same model), and the other four research tasks and the
KV cache once each, all passed with no unfetched citations. Research still ended all
three ratio steps with the task's `RATIO:` line, but the final answers were right.

## Development

```powershell
uv sync
uv run pre-commit install
uv run pytest
uv run ruff check .
```
