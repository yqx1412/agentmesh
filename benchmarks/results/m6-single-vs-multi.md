# M6: one agent vs. the whole fleet

**Question:** on the same tasks, does the multi-agent system (orchestrator + Research,
Analyst, Verifier, Knowledge over A2A) beat a single AgentOS agent that has all of their
tools, and what does it cost?

**Short answer:** the fleet gives more trustworthy answers on single-skill tasks, mainly
because each specialist's rules are enforced in code, and it is worse on tasks that chain
two skills, because splitting the task loses information between steps. It costs about
3.4x the tokens and 1.7x the time. With 16 tasks and 3 repeats, the overall gap (81% vs
73% of runs passed) is small enough that it should be read per category, not as one score.

**Update, 2026-10-19:** after the orchestrator, Research and Analyst fixes that came out
of this evaluation (PRs #8-#12), a full rerun with the same tasks and settings passed
47/48 on the fleet and 34/48 on the unchanged single agent, and the fleet won all 9
composite runs. Those fixes were tuned on these same composite tasks, so that 9/9
overstates how the fleet would do on new chained tasks. See
[Rerun with the fixes](#rerun-with-the-fixes-2026-10-19). The sections before it
describe the original 2026-10-18 run.

Raw data: [`m6/results.jsonl`](m6/results.jsonl) (one line per run, with the answer,
check results, cost and fetched pages), [`m6/report.md`](m6/report.md) (all tables),
[`m6/meta.json`](m6/meta.json). Tasks: [`../m6/tasks.yaml`](../m6/tasks.yaml).

## Setup

| | single | multi |
|---|---|---|
| What | `agents/solo.toml`: one AgentOS agent on 9115 | `agentmesh orchestrate` over the four M2-M5 agents |
| Models | qwen3:14b, reasoning on | planner qwen3:14b; Research and Knowledge qwen3:8b; Analyst and Verifier qwen3:14b |
| Tools | web_search, fetch_url, run_python, calculator, DomainGraph search/get_concept/related_concepts/recall_facts/list_facts | each agent its own subset (Knowledge also has add_fact) |
| Code-enforced rules | empty answers refused, invented quotes refused | the same, plus Research must fetch 2 pages and the Verifier 1 before answering; Knowledge writes need a Verifier signature |
| Context | 16k, 30 steps | 16k per agent; 12-16 steps each |
| Login | `user` | `orchestrator` (M5) |

Both systems get the identical prompt over A2A. 16 tasks in 5 categories: **research**
(5 single web facts), **analysis** (4 computations from given numbers), **knowledge** (3
questions about the user's MIT 6.006 lecture notes in DomainGraph), **verification** (3
claims with sources, one false), **composite** (3 tasks needing two skills chained).
Each task ran 3 times per system (96 runs, 2026-10-18, one RTX card with 16 GB, Ollama).
The two systems ran task by task, alternating which went first.

**Scoring** is deterministic and looks only at the final answer: required phrases,
numbers (with a tolerance), regexes, and for lecture questions a timestamp inside the
right passage. Every task's checks were tested against a hand-written correct answer and
a plausible wrong one (`tests/test_m6.py`). For the 7 tasks that ask for web citations,
every URL the answer cites is compared with the pages the system actually fetched, read
from the agents' traces. Tokens are summed over every model call in every trace,
including replies the answer guard sent back (that accounting was added for M6).

## Results

| | single | multi |
|---|---|---|
| Runs passed | 35/48 (73%) | 39/48 (81%) |
| Passed, every cited page fetched | 23/48 (48%) | 37/48 (77%) |
| Checks passed | 58/75 (77%) | 64/75 (85%) |
| Median / mean time per run | 27 s / 34 s | 45 s / 59 s |
| Mean tokens per run | 4,707 | 16,032 |
| Total for 48 runs | 27 min, 226k tokens | 47 min, 770k tokens |
| Mean model calls / tool calls | 2.0 / 1.3 | 6.0 / 3.3 |
| Runs that cited an unfetched page | 14 of 14 | 2 of 19 |

| Category | single passed | multi passed | single median s | multi median s | single tokens | multi tokens |
|---|---|---|---|---|---|---|
| research | 10/15 | 13/15 | 25 | 43 | 3.6k | 18.9k |
| analysis | 12/12 | 12/12 | 36 | 53 | 5.1k | 5.5k |
| knowledge | 6/9 | 9/9 | 23 | 19 | 4.9k | 5.7k |
| verification | 1/3 | 3/3 | 12 | 43 | 3.9k | 30.6k |
| composite | 6/9 | 2/9 | 38 | 122 | 6.2k | 30.8k |

Per task, the fleet passed more often on 4, less often on 3 and equally on 9.

## Where the fleet helps

**Answers are grounded in what was actually read.** None of the single agent's 14
answers that cited a web page had fetched one: in 11 it searched and answered from the
snippets, and in 3 it made no tool call at all. Its research answers were often right
anyway (10 of 15), but the citations were decoration, and one answer was simply invented:
it gave `--speculative-model` as llama.cpp's draft-model option, citing the repository,
without opening anything (the real option is `-md` / `--model-draft`). The fleet's
Research agent cannot answer before two successful fetches (the M2 rule), so 17 of the
fleet's 19 citing answers cited only fetched pages. In the other two, Research cited one
search result it hadn't opened next to the pages it had: the rule forces fetching, not
citing only what was fetched.

**Checking really checks.** On the three-claims task the single agent answered
"FALSE: 1,2,3" in 12 s with no tool call in 2 of 3 runs, calling true claims false.
The Verifier must fetch before answering, and got it right 3 of 3 times.

**A narrow agent picks the right tool.** Asked what MCP stands for, the single agent
searched the *lecture notes* in all 3 runs, found nothing and gave up: with web and
knowledge-base tools side by side, "MCP" did not look like a web question to it. The
planner sent it to `web_research` every time. That passed 1 of 3: twice Research fetched
the official page but took the comparison from a Microsoft page ("a file explorer")
instead of the official "USB-C port". On the lecture questions, the Knowledge agent
cited the right passage of Lecture 9 (19:28) 3 of 3 times; the single agent cited an
earlier passage of the same lecture every time (12:55 twice, 14:48 once).

Knowledge questions were also the one place the fleet was *faster*: the planner sends
them straight to the 8b Knowledge agent, which doesn't reason at length.

## Where it doesn't

**Chained tasks lose information at the hand-off.** On the 3 composite tasks the fleet
passed 2 of 9; the single agent passed 6 of 9, at a third of the time.

- *Qwen3-8B's KV cache* (0/3 vs 1/3): no plan asked Research for the context length
  the computation needs, only for layers, KV heads and head dimension. Twice the Analyst
  then replied, correctly, that it couldn't compute the size. Once it assumed 32,768
  itself but used a head dimension of 5,120 that Research had taken from a third-party
  page, and answered 90 GiB. The single agent failed differently: once its fetch of the
  config hit a PDF the fetch tool can't read and it guessed a head dimension of 256, and
  once it searched, found nothing usable and gave up.
- *Ollama windows per Qwen3 context* (0/3 vs 2/3): all three plans had four steps, the
  last asking for "the ratio of Ollama's default window size to Qwen3-8B's native
  context length", the inverse of the question. The Analyst answered both steps
  correctly (8 and 0.125), and the combiner ended with `RATIO: 0.125` every time, once
  right after writing that the answer is 8. These runs took 136-262 s and 48-61k tokens.
- *Lecture 1 speed-up* (2/3 vs 3/3): Ollama's runner crashed during an embedding call
  (`0xc0000409`; three tool calls failed this way in the whole run). The Knowledge agent
  reported the error as its answer, the step counted as *completed*, and the Analyst
  received an error message instead of numbers. The single agent hit the same crash in
  the same repeat, searched again and passed.

The common cause is that the planner writes every step's input, and a step sees only
that text, never the user's task. Whatever the planner leaves out or rewords is gone.
And an agent that politely says it couldn't do its part still "completes", so the M4
retries and fallback never fire.

**Single-skill tasks gain nothing from the orchestrator.** On the four computations both
systems passed 12 of 12, and the fleet took longer (median 53 s vs 36 s) for the plan
and combine calls around the same Analyst-style work.

**It does things it wasn't asked to do.** On "who contributed A2A?" the planner added
Verifier and `knowledge_write` steps in all 3 runs, though the task asked only a
question. In the first run the M5 gate refused the write (the Verifier's reply had no
SUPPORTED line with a source it could match). In the other two the Knowledge agent
stored two facts in the user's knowledge base (ids 306 and 307; the second run got the
same ids back, so nothing was duplicated). The gate stops *unverified* writes; it does
not stop *unrequested* ones. The first of those runs cost 40k tokens against 3.6k for
the single agent.

## How much to trust this

- **Small sample.** 16 tasks, chosen and written by me with the system in mind. A
  difference of one or two tasks per category is within what a different task set
  could flip.
- **Repeats are not independent.** The fleet gave the identical answer in all 3 repeats
  on 8 of 16 tasks (the single agent on 1 of 16). Its planner runs at temperature 0, so
  the same plan tends to produce the same chain. The 48 runs per system are closer to
  16-30 distinct observations.
- **The guards are a confound.** Much of the fleet's research and verification
  advantage comes from rules like "fetch before answering", which the single agent
  can't have unconditionally (it would also block computation tasks). This measures
  *specialists with their rules* against *a generalist without them*, not "one agent vs
  several" in the abstract.
- **Tokens aren't comparable across models.** The fleet's tokens are split between 8b
  and 14b models; wall time on the one GPU is the fairer cost.
- **The web moves.** Research answers depend on DuckDuckGo results and on which pages
  allow fetching (403s, 429s and PDFs all occurred).
- **Read-only task set.** No task asked for a write, so the Verifier-signed write path
  was only exercised by the planner's own unrequested writes.

## What would close the gaps

1. **Give every step the user's task**, not only the planner's rewrite, and let the
   Analyst ask for missing inputs by failing instead of completing. That targets both
   composite failures.
2. **Treat "I couldn't" as a failed step** (e.g. a step whose answer reports a tool or
   environment error), so the M4 retry and fallback apply.
3. **Give the single agent a conditional guard:** "every URL the answer cites must have
   been fetched". It is checkable for any agent, as this evaluation does after the fact,
   and would likely remove most of the fleet's research advantage at a fraction of its
   cost.
4. **Don't let the planner add writes the task didn't ask for.**

Items 1, 2 and 4 are done (see the rerun below); item 3 is not, so the single agent in
the rerun is the same one as here.

## Rerun with the fixes (2026-10-19)

The same 16 tasks, 3 repeats per system, the same timeouts (1,200 s per run, 600 s per
step) and the same hardware, on `main` at `1b7824a`. Raw data:
[`m6-rerun/results.jsonl`](m6-rerun/results.jsonl),
[`m6-rerun/report.md`](m6-rerun/report.md), [`m6-rerun/meta.json`](m6-rerun/meta.json).

What changed in the fleet since the first run (all in the orchestrator or the agents'
prompts; models, tools and code-enforced rules are the same):

- every step also receives the user's task; a step can fail with `CANNOT: <reason>`,
  and the planner re-plans once when one does (PR #8)
- the final summary states each value the task asked to look up, with its source (#9)
- Research reports values under the exact name the task uses, prefers the model's own
  card, and cites only pages it fetched (#11, #12)
- the planner lists every input before its steps, and may not write a formula into a
  step (#12)
- delegated steps ignore the task's formatting instructions; the Analyst names every
  factor of a formula (#12)
- skills tagged `write` are hidden from the planner unless `--allow-writes` is given,
  which the evaluation doesn't pass (#8)

The single agent's spec (`agents/solo.toml`) did not change.

| | single, first run | multi, first run | single, rerun | multi, rerun |
|---|---|---|---|---|
| Runs passed | 35/48 | 39/48 | 34/48 | 47/48 |
| Passed, every cited page fetched | 23/48 | 37/48 | 23/48 | 46/48 |
| Checks passed | 58/75 | 64/75 | 60/78 | 77/78 |
| Median / mean time per run | 27 / 34 s | 45 / 59 s | 31 / 39 s | 44 / 51 s |
| Mean tokens per run | 4,707 | 16,032 | 5,501 | 20,438 |
| Runs that cited an unfetched page | 14 of 14 | 2 of 19 | 13 of 13 | 1 of 18 |
| Composite passed | 6/9 | 2/9 | 6/9 | 9/9 |

The check count went from 75 to 78 because the A2A task's host check became a separate
check that accepts either the Linux Foundation or the Agentic AI Foundation (#10).

**Composite tasks, fleet:** KV cache 0/3 -> 3/3, context-window ratio 0/3 -> 3/3,
lecture speed-up 2/3 -> 3/3. The composite median time fell from 122 s to 82 s; the
ratio task alone went from 246 s to 91 s, since the plan no longer has a fourth step
asking for the inverse ratio. **The MCP question** went from 1/3 to 3/3. Fleet tokens
rose on research (18.9k -> 28.7k per run) and verification (30.6k -> 40.0k), so the
fleet now costs about 3.7x the single agent's tokens (3.4x before) at an unchanged
median time.

**The fleet's one failure is a regression,** on `r-qwen-license` (3/3 -> 2/3). In all 3
runs the planner added a `claim_verification` step to this plain lookup (median 67 s,
was 45 s). In the failed run, Research's two other fetches failed (an NVIDIA PDF, which the
fetch tool can't read, and a 403), so it fetched the Hugging Face card twice. The card's page shows
"License: apache-2.0" in its header, but the text the fetch tool extracted from it
contains neither "license" nor "apache"; the license was only in the search snippets.
Under the new "cite only pages you fetched" rule, Research said the license was "not
directly provided in the fetched content". That is honest, but it fails the task. In
another of the 3 runs, the final answer cited that NVIDIA PDF, which no agent had
fetched: the fleet's only unfetched citation in the rerun.

**Single agent:** its 1-run differences from the first run (35 -> 34 passed:
`r-qwen-license` 3 -> 2, `r-llama-draft` 1 -> 2, `v-three-claims` 1 -> 0) are what an
unchanged agent varies by. Its failures are the same kinds as before: every citing
answer cited a page it never fetched, it searched the lecture notes for the MCP
question in all 3 runs, and on the claim check it now answered "FALSE: 2,3" all 3 times.

Per task, the fleet passed more often on 6 tasks, less often on none and equally on 10.

**How to read it:**

- **Not a fresh measurement of the fixes.** Every prompt rule above was written after
  reading failures on these composite tasks, and checked live on them, so 9/9 shows the
  fixes work on the cases they were written for, not on chained tasks in general.
- **The ratio task's wording changed:** it now names a GPU with less than 24 GiB of
  VRAM, because Ollama's docs now give a different default per VRAM tier. On that task
  the two runs aren't measuring the same thing.
- **No writes were possible.** With write skills hidden by default, the planner couldn't
  add the unrequested writes it added in the first run, so this rerun doesn't test
  whether it still would with `--allow-writes`.
- The earlier caveats (16 self-written tasks, repeats that aren't independent, guards
  as a confound, tokens across two model sizes) still apply.

## Reproducing

```powershell
uv run agentmesh up --trace-dir runs/fleet-traces
uv run agentmesh serve agents/solo.toml --trace-dir runs/fleet-traces/solo
uv run agentmesh eval run --out runs/m6/full --repeats 3 --timeout 1200 --step-timeout 600
uv run agentmesh eval report runs/m6/full
```

Re-running `eval run` into the same folder resumes where it stopped.
