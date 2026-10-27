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

**Full M6 rerun with these fixes** (2026-10-19, same tasks and settings; details in the
[write-up](benchmarks/results/m6-single-vs-multi.md#rerun-with-the-fixes-2026-10-19),
raw data in `benchmarks/results/m6-rerun/`):

| | single | multi |
|---|---|---|
| Runs passed | 34/48 (M6: 35) | 47/48 (M6: 39) |
| Passed, every cited page fetched | 23/48 (M6: 23) | 46/48 (M6: 37) |
| Composite | 6/9 (M6: 6) | 9/9 (M6: 2) |
| Median time / mean tokens | 31 s / 5.5k | 44 s / 20.4k |

The single agent's spec is unchanged. The fleet's one failure is a new regression on the
Qwen3-8B license lookup: the fetched text of the model card doesn't contain the license,
and Research, now told to cite only fetched pages, said so. The prompt rules were tuned
on these same composite tasks, so 9/9 overstates the fleet on new chained tasks.

After the rerun, three more fixes.
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

That `RATIO:` line is now taken out in code. The planner copies the task's reply format
into a new `"format"` field (checked to be word for word from the task), and every step's
answer loses the lines that start with that format's label (`RATIO:`, `FALSE:`) before a
later step or the final reply reads it; the journal records each line dropped. Two
versions that didn't work: removing the format from the task the steps see (the ratio
task 0/3, because without "RATIO" the Analyst gave 0.125 as "the ratio" and the final
reply used it; the claim check also fell to 0/3), and putting `"format"` first in the
plan (the planner then wrote "Compute the ratio of A to B" instead of the task's "How
many A fit in B", 0/3). With `"format"` last, two rounds of the ratio task and the claim
check passed **12/12**, and the five research tasks, the KV cache and the lecture
speed-up passed once each. In the final round, Research's `RATIO: 8` and the Verifier's
`FALSE: 2` were dropped in all 6 runs, and nothing else was. (An earlier round also
dropped the Analyst's "- Ratio: 4096 / 32768 = 0.125" line from its list of numbers, so
the match is now case-sensitive; in one plan the planner copied only `RATIO: <number>`,
which is now recognised too.)

**The answering step keeps the task's question.** The full rerun after the fix above
(on `62867a9`) dropped the ratio task to 0/3, and a recheck to 1/3: the planner still
rewrote "How many of A fit in B?" as "Compute the ratio of A to B" in most runs, and the
Analyst correctly answered 0.125. The earlier 12/12 had been luck. Now `parse_plan`
sends a plan back when a computing step (a skill tagged `math`) that turns earlier
results into the answer doesn't ask the task's one question in its words, in order
(`{{sN}}` references and other words may sit between them); the planner prompt says so
too. Composite tasks, fleet only, **10 runs each**:

| Task | Passed | Median |
|---|---|---|
| Qwen3-8B KV cache | 8/10 | 122 s |
| Context-window ratio | 8/10 | 75 s |
| Lecture speed-up | 10/10 | 87 s |

All 10 ratio plans kept the question. Its 2 failures were the final summary replying
only `RATIO: 8.0`, dropping the two values the task asked to look up. The KV failures
were Research's lookups: a head dimension of 8, and `config.json`'s 40,960 used as the
native context length.

**The final reply states the looked-up values.** When the task asks to find, look up or
search for something, the orchestrator checks the final reply for every value the steps
looked up and a computing step used: a number in a `math` step's answer that also
appears in the answer of a step it depends on, and isn't in the task itself (0, 1 and 2
are ignored as counts and factors; URLs and names like "Qwen3-8B" don't count). If any
are missing, the summary is asked once more with those values named, and the second
reply is kept only if it leaves out fewer. Over 277 saved runs with a computing step, the
check never fired on a task that doesn't ask for a lookup, and it caught both summaries
above. Composite tasks again, fleet only, **10 runs each**:

| Task | Passed | Median | Reply asked again |
|---|---|---|---|
| Qwen3-8B KV cache | 9/10 | 129 s | 0 |
| Context-window ratio | 9/10 | 83 s | 1 (fixed) |
| Lecture speed-up | 10/10 | 88 s | 10 (all fixed) |

No reply left a value out after the second try. The lecture summary always gave only the
speed-up at first, without the two timings the task asked to find (its check doesn't need
them, so it passed before too); each fix cost one extra summary call. Both failures were
Research's lookups again: a head dimension of 5,120 (the hidden size) and 32,768 as
Ollama's default for a GPU under 24 GiB.

**Research's numbers come from pages it fetched.** Those two failures had one cause: the
value was in no page Research fetched. Qwen3-8B's model card gives no head dimension, so
5,120 was made up, and Research had answered the Ollama question after a search that came
back as one empty hit. Most passing KV runs had the same gap: their 128 was from memory.
The fixes are in code:
- **Number check** (`check_numbers = "fetch_url"` in `agents/research.toml`): a final
  answer with a number that's in no fetched page and not in the task is sent back, like
  the Verifier's quote check. URLs, list markers, names like "Qwen3-8B" and 0-2 don't count,
  and "4k" on a page covers 4,096. Over 296 saved Research answers it would have fired on 34,
  mostly the remembered 128.
- **Model card and config.json together** (`webtools.with_model_config`): fetching a
  Hugging Face model page also returns its `config.json`, first, with a note that
  `max_position_embeddings` isn't the native context length when the card says otherwise.
  Fetching a `config.json` (any revision) also returns the card. Research had looked for
  `config.json` on GitHub paths that don't exist, and when it fetched only `config.json`
  it took 40,960 as the native length.
- **Search:** an empty hit is dropped, and no hits at all is an error, so Research searches
  again. Ten hits are always fetched from DuckDuckGo, at least five are returned, and a
  model's own Hugging Face pages are moved to the top: Research fetched the first hit in
  174 of 229 saved runs, and its usual query's 3- or 5-hit lists often left the card out
  while a 10-hit list included it.
- **Prompt:** a model card is the model's Hugging Face page or its developer's site, not a
  spec database or context-window list.

Fleet only, **10 runs each**, adding the fixes one at a time:

| After | KV cache | Ratio |
|---|---|---|
| Summary check (above) | 9/10 | 9/10 |
| Number check, empty-search fix | 3/7 (stopped) | 6/6 |
| Card and config.json together | 8/10 | 9/10 |
| config.json also brings the card | 10/10 | 8/10 |
| Model-card prompt rule | 7/10 | 10/10 |
| At least 5 hits | 3/7 (stopped) | 7/7 |
| 10 fetched, model pages first | **10/10** | **9/10** |

With the number check alone, Research honestly said the card has no head dimension and
the Analyst assumed one; each later row fixes how Research gets to the page that has it.
In the last round every KV run read the head dimension from `config.json`. The one ratio
failure took 40,960 from contextwindows.dev: no Hugging Face page was among that search's
10 hits. The number check never fired in the last 65 live runs, since the values were now
on the fetched pages; it stays as the guard for when they aren't. Rows differ by more than
any one fix explains, so treat single rows as noisy: 19/20 in the last round against 18/20
before these fixes is not a significant difference by itself. The 5 research tasks passed
4/5; `r-mcp-definition` failed because the MCP intro page Research fetched no longer has
the USB-C comparison in its extracted text.

That last failure was the fetch tool, not the page. The MCP docs (a Mintlify site) write
their paragraphs as `<span data-as="p">`, and trafilatura drops the loose spans before the
first heading, which is where the lead paragraph and its "USB-C port" line sit. The full
M6 rerun on `ef8febc` failed that task 3/3 for the fleet (45/48 overall, composites 9/9).
`fetch_page` now turns `data-as="p"` elements into real `<p>` tags before extracting. On
the Ollama FAQ, another Mintlify site, this also stops trafilatura repeating half-sentences
around inline code ("Use theollama ps command ... command to see ..."). Live, fleet only:
`r-mcp-definition` 3/3 with the intro page cited, context-window ratio 3/3, and the 5
research tasks 5/5 once each.

The Hugging Face card hid its license the same way: the page shows "License: apache-2.0"
only in its header, which trafilatura drops. A fetched model card now starts with a
`Card metadata:` line built from the front matter of the repository's `README.md`
(license, base model, pipeline tag and other short fields; the gated-access form and
`model-index` are left out). It costs one more request per model card.

**Second full M6 rerun** (2026-10-23, same tasks and settings, with every fix above;
details in the [write-up](benchmarks/results/m6-single-vs-multi.md#second-rerun-2026-10-23),
raw data in `benchmarks/results/m6-rerun-2/`):

| | single | multi |
|---|---|---|
| Runs passed | 36/48 (first rerun: 34) | 48/48 (first rerun: 47) |
| Passed, every cited page fetched | 26/48 (23) | 48/48 (46) |
| Composite | 5/9 (6) | 9/9 (9) |
| Median time / mean tokens | 34 s / 6.1k | 45 s / 15.9k |

The single agent's spec is unchanged, though its web tools got the fetch and search fixes.
With 3 repeats, 48/48 is partly luck; the 10-run composite evals above (about 90-95%) are
the better estimate for chained tasks.

After that rerun: when a search names a model (`Qwen3-8B`, `qwen3:8b`, `org/model`) and
none of its hits is a Hugging Face model page, `web_search` asks the hub's own search
(`huggingface.co/api/models`) for a model with exactly that name and puts its page first.
The last failing ratio run had no such hit and took 40,960 from a context-window list.
Live, the fleet passed the ratio, KV and five research tasks 21/21 (3 each); the lookup
fired in all 3 ratio runs, whose searches again returned no Hugging Face page.

`fetch_url` now reads PDFs (`pypdf`, text layer only, first 40 pages). It used to refuse
`application/pdf`, and in the next full rerun two license answers cited NVIDIA's Qwen3-8B
model card PDF after its fetch had failed. A scanned or unreadable PDF is still an error.
The eval also counts a fetched page under its URL after redirects: that PDF moves to
`developer.download.nvidia.com`, and an answer citing the final URL was scored as citing
an unfetched page. Re-scored from their traces, M6 and both published reruns are
unchanged. Live, fleet only: license 3/3, all three reading the PDF, and KV cache 3/3,
every cited page fetched.

Research's citations are now checked in code (`check_citations = "fetch_url"` in
`research.toml`): an answer citing a URL no successful `fetch_url` call returned goes back
to Research, like the number check. A redirect's final URL and the card/`config.json`
pages a Hugging Face fetch returns together count as fetched; the task's own URLs and
local addresses (`http://localhost:11434` as a value) are not citations. The guard keeps
the fetches it has seen, so a history compaction can't make a cited page look unfetched.
Before, only a prompt rule asked for this, and the next full rerun had 3 fleet answers
citing pages never fetched (a failed PDF fetch twice, ssdnodes.com once). Over 555 saved
Research answers it would have fired on 16, all real: those 3, and earlier citations of
pages whose fetch failed or never happened. On the 21 Research answers of the rerun before
that (every cited page fetched) it fired on none. Live, fleet only: license 3/3, ratio 3/3
and the 5 research tasks 5/5, no kept answer citing an unfetched page. In 4 of these runs
the NVIDIA PDF's fetch failed (before PDF reading), and no kept answer cited it; these
traces predate send-back logging, so whether the guard sent an answer back first isn't
known.

The number check now reads the pages each claim cites. A number on a line that cites URLs
(one of Research's claims) must be on a fetched page that line cites; the summary's
numbers may come from any fetched page the answer cites, or from any fetched page when it
cites none. Before, any fetched page counted: the 4,096 claimed for Ollama citing
ssdnodes.com passed because 4096 is also Qwen3-8B's `hidden_size`, in the `config.json`
fetched with the card another claim cited. Replayed on saved traces it flags exactly the
3 bad answers of that rerun (two of them cited the failed PDF fetch) and none of 32 good
ones (the rerun before it and the citation check's live runs); over 566 saved Research
answers it
fires on 10 more than before, each a value cited to a page that doesn't state it (often a
correct head dimension of 128 cited to the card instead of `config.json`). Live, fleet
only: ratio 3/3 and KV cache 3/3, every cited page fetched.

**Third full M6 rerun** (2026-10-24, same tasks and settings, on `84a7c26` with every fix
above; details in the [write-up](benchmarks/results/m6-single-vs-multi.md#third-rerun-2026-10-24),
raw data in `benchmarks/results/m6-rerun-3/`):

| | single | multi |
|---|---|---|
| Runs passed | 33/48 (second rerun: 36) | 48/48 (second rerun: 48) |
| Passed, every cited page fetched | 26/48 (26) | 48/48 (48) |
| Composite | 5/9 (5) | 9/9 (9) |
| Median time / mean tokens | 31 s / 5.5k | 40 s / 14.6k |

The unpublished rerun before the last three fixes had 45/48 with every cited page
fetched; this one has 48/48. Research's kept answers all pass the new checks, but this
run's traces predate send-back logging (below), so it isn't known whether the checks
fired live.

Every answer a check sends back is now in the agent's trace, as an `answer_sent_back`
event: which check (`empty`, `required_calls`, `quotes`, `numbers` or `citations`), what it
flagged, the refused reply and its tokens. Before, the trace had only the reply the agent
kept, so whether a check had fired couldn't be told afterwards. `agentmesh eval` counts
them per run (`sent_back` in `results.jsonl`, an "Answers sent back by a check" row in the
report) and counts each refused reply as a model call. Live, research, verification and
composite tasks once each on both systems: 5 send-backs in 4 of the fleet's 9 runs, all
`required_calls` (Research answering before its two fetches 3 times, the Analyst
answering without `run_python` twice), none from the citation or number checks; the
single agent's guard sent nothing back.

Those required-call send-backs cost a whole extra model call each, so the guard now also
tells the model what is still missing before it answers: while a required call is
missing, each model call gets a one-line note after the history, "(Status: before your
final answer you need at least 1 more successful fetch_url call(s).)". The note isn't kept
in the history and sits at the end, so the prompt prefix stays the same between steps.
Fleet only, research, verification and composite tasks 3 times each (27 runs): before,
13 runs had a send-back (14, all `required_calls`); with the note, 1 run did (an empty
reply), and none was `required_calls`. Model calls fell from 188 to 174; tokens were about
the same (525k against 534k), and 26 of 27 runs passed (27 before; the failure was
`r-mcp-definition`, where that run's search and Wikipedia fetch returned no "USB-C"). Two
other wordings were worse: "Until then, reply with a tool call" made the Verifier fetch its
three pages one per step, and "several can go in one reply" made Research batch vague
searches, fetch an unofficial MCP page and fail `r-mcp-definition` 3/3. The eval also now
counts the config.json that comes back with a Hugging Face card fetch as fetched, as
Research's citation guard already did; one run citing it was scored as unfetched. Re-scored
from their traces, the published M6 results are unchanged.

In the 2026-10-26 full rerun, the Analyst's first reply on `a-weights` came back empty in all
3 runs, and the guard sent it back each time. Replaying that call against Ollama showed
why: qwen3:14b, thinking on, spent the whole 2,048-token reply limit (AgentOS's default
`num_predict`) deciding whether the script's `7.7` meant the two-decimal `7.70`, and was
cut off before writing anything. The Analyst's instructions now ask it to print rounded
results with a format string at the precision asked for (`f"{x:.2f}"`); replayed with
`7.70` printed, the same call answered in 539 tokens. AgentOS now keeps Ollama's
`done_reason` on each reply and writes it to the trace; the guard's empty check says when
a reply was cut off at the limit, and `agentmesh eval` counts such replies (`cut_off` in
`results.jsonl`, a "Replies cut off at the reply limit" row; older traces read "none").
Live, the 4 analysis and 3 composite tasks 3 times each on the fleet (21 runs): all passed,
as in the rerun; `a-weights` had no send-backs (3 before), its median time fell from 122 s
to 57 s and its tokens by 31%; over the 21 runs, model calls fell from 128 to 119 and
tokens from 342k to 318k. No reply hit the limit. The single agent's spec is unchanged, to
keep the M6 comparison.

**Side-effect tasks** (`benchmarks/m6/tasks-side-effects.yaml`, outside the 16 M6 tasks so
those numbers stay comparable) test what a run does, not only what it answers. Run them
with `agentmesh eval run --tasks benchmarks/m6/tasks-side-effects.yaml`.
- **Writes** (fleet only, with `allow_writes`): a `stored` check reads the Knowledge
  agent's add_fact results in its trace, not the answer's account of them.
  `w-verified-write` asks for a supported claim and a contradicted one under the same URL
  to be saved. `w-unverified-detail` asks for a sentence that adds an unchecked 32,768 to
  a claim the Verifier supported, so only #21's statement match can refuse it. Both must
  store the 11434 claim and nothing with 32,768.
- **Cancels:** the eval cancels the run itself, waits, then asks each agent for the
  state of every task the run started. A `remote_stopped` check needs each one
  `canceled`, and no agent may make more than one model call after the cancel.
  `x-cancel-dispatch` cancels the orchestration the moment it sends its first step, before
  any task id has come back (#19's case). `x-cancel-midrun` cancels both systems 30 s in.
  A test that puts #19's bug back (no wait for the id) fails the check.

Live, 3 times each: all 15 runs passed. In every `w-unverified-detail` run, Knowledge's
first write was refused by the statement match, and it then stored the verified claim. In
`w-verified-write`, the contradicted claim was refused by the source's spent budget, not
by #21. Every cancelled task read `canceled` afterwards. The fleet's agents made no model
call after the cancel, and the single agent made at most the one already in flight. The
write tasks add to the knowledge base: the first runs stored facts 321 and 322, and later
runs find them already stored. A cancelled fleet run reports only its agents' tokens:
it returns nothing, so the planner's tokens aren't joined to it.

**What the guard reads, across compaction.** The checks decide a URL or number is the
task's own when it appears in a user message, and count a page as fetched from the
`fetch_url` results in the history. An agent compacts a long history into one *user*
message holding the clipped older text, which breaks both: in a 2026-10-26 `c-ollama-ratio`
run the summary carried Research's search hits, so the citation check took one
(`neuralgist.com`) for a task URL and the number check took its "4,096" for a task number,
while the fetched pages the answer should have been checked against were gone. The guard
now reads a history of its own: the system prompt, the task message, and every tool call
and result it has seen during the task (each kept once, by `tool_call_id` or object
identity so two real results never merge and required-call counting is unchanged),
including ones compaction has since dropped. The compaction summary and the guard's own
send-back message count as neither task-given nor fetched, so a reworded summary can't
launder a bad citation and a retry can't pass on the values the send-back named.

**The citation guard, finished end to end** (2026-10-27, PRs #31-#34; details and the
matched-trace-root rerun in the
[write-up](benchmarks/results/m6-single-vs-multi.md#citation-integrity-chain-2026-10-27)):

- **A citation the nudge budget can't fix is stripped** (#31). `max_nudges` bounds
  retries, so a model that keeps citing an unfetched page eventually gets through; when
  the budget is spent and the final answer still cites one, the guard removes that
  citation rather than ship it (`strip_citations`, a `citations_stripped` trace event).
- **The orchestrator's combined answer is guarded too** (#32). `check_citations` is only
  on Research, so the orchestrator could relay or invent a citation no step grounded; it
  now drops any citation the final answer makes that no completed step's answer carries.
  Same PR: `agentmesh eval` flags a run `citations_unscoreable` when it cited pages but no
  located trace fetched anything — the citation metric was silently blaming every URL when
  the eval's `--trace-root` didn't match the fleet's `--trace-dir`.
- **The single agent got the same guard** (#33): `check_citations = "fetch_url"` in
  `agents/solo.toml`. It only bites when the answer cites a URL, so solo's compute and
  lecture-note tasks are unaffected. A focused live check saw it fire on every run.
- **A comma- or bracket-bearing citation now matches its fetched entry** (#34). The URL
  regex treats those as terminators, so such a URL was extracted clipped and mis-scored
  as unfetched; `urls.citation_fetched` accepts a cited URL that is a prefix of a fetched
  one at a clip-char boundary, used by both the guard and the scorer.

A full rerun on `243dd32` with `--trace-root` matched to the fleet passed 48/48 with every
cited page fetched and, for the first time, all fleet citations scoreable.

## Development

```powershell
uv sync
uv run pre-commit install
uv run pytest
uv run ruff check .
```
