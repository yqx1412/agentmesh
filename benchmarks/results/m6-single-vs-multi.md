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

**Update, 2026-10-23:** after PRs #14-#18, a second full rerun passed 48/48 on the fleet,
every cited page fetched, and 36/48 on the single agent, whose prompt is unchanged but
whose web tools got the fetch and search fixes too. 3 repeats is a small sample: 10-run
evals of the composite tasks put the fleet at about 90-95% on them. See
[Second rerun](#second-rerun-2026-10-23).

**Update, 2026-10-24:** after PRs #19-#24 (PDF reading, and Research's citations and
numbers checked in code), a third full rerun again passed 48/48 on the fleet with every
cited page fetched, and 33/48 on the single agent. See
[Third rerun](#third-rerun-2026-10-24).

**Update, 2026-10-27:** the citation guard was finished off end to end (PRs #28-#34):
compaction no longer launders a search hit into a citation, a citation the nudge budget
can't fix is stripped, the orchestrator's combined answer is guarded too, the single
agent got the same guard, and the eval stopped mis-scoring citations (both a trace-root
footgun and a URL-comma clip). A matched-trace-root full rerun passed 48/48 on the fleet
with every cited page fetched and all citations scoreable, and a focused solo check saw
the new single-agent guard fire on every run. See
[Citation-integrity chain](#citation-integrity-chain-2026-10-27).

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

## Second rerun (2026-10-23)

The same 16 tasks, 3 repeats per system, timeouts and hardware, on `main` at `ef8febc`
plus the two fetch fixes of PR #18 (`meta.json` records this as `ef8febc-dirty`; the
fixes were committed after the run, unchanged). Raw data:
[`m6-rerun-2/results.jsonl`](m6-rerun-2/results.jsonl),
[`m6-rerun-2/report.md`](m6-rerun-2/report.md),
[`m6-rerun-2/meta.json`](m6-rerun-2/meta.json).

What changed since the first rerun:

- the planner adds a verification step only when the task asks for a check or before a
  knowledge-base write; it gives each skill one step; Research searches and fetches
  separately for each thing it is asked about (#14)
- the task's format lines (`RATIO:`, `FALSE:`) are removed from each step's answer;
  duplicate knowledge-base writes are reported as already stored (#15)
- `parse_plan` rejects a computing step that doesn't ask the task's question in its words
  (#16)
- the final reply is asked again if it leaves out a value the steps looked up; Research's
  answers are sent back if they contain a number from no fetched page (#17)
- web tools: a Hugging Face model card comes with its `config.json`; search always takes
  10 hits, drops empty ones and puts the model's own pages first (#17); paragraphs written
  as `<span data-as="p">` are kept, and a model card starts with its license and other
  front-matter fields (#18)

The single agent's spec (`agents/solo.toml`) did not change, but it uses the same web
tools, so the last bullet applies to it too.

| | single, first run | multi, first run | single, rerun | multi, rerun | single, now | multi, now |
|---|---|---|---|---|---|---|
| Runs passed | 35/48 | 39/48 | 34/48 | 47/48 | 36/48 | 48/48 |
| Passed, every cited page fetched | 23/48 | 37/48 | 23/48 | 46/48 | 26/48 | 48/48 |
| Checks passed | 58/75 | 64/75 | 60/78 | 77/78 | 63/78 | 78/78 |
| Median / mean time per run | 27 / 34 s | 45 / 59 s | 31 / 39 s | 44 / 51 s | 34 / 42 s | 45 / 50 s |
| Mean tokens per run | 4,707 | 16,032 | 5,501 | 20,438 | 6,066 | 15,895 |
| Runs that cited an unfetched page | 14 of 14 | 2 of 19 | 13 of 13 | 1 of 18 | 12 of 12 | 0 of 19 |
| Composite passed | 6/9 | 2/9 | 6/9 | 9/9 | 5/9 | 9/9 |

**Fleet:** every task passed 3/3. Against the first rerun, the only per-task change is
the license lookup (2/3 -> 3/3): the plan is a single Research step again (49 s median,
was 67 s), and the card's text now contains its license. Mean tokens fell from 20.4k to
15.9k, mostly on verification (40.0k -> 9.5k; the claim check's median went from 49 s to
21 s) and research (28.7k -> 22.9k), so the fleet now costs about 2.6x the single agent's
tokens (3.7x in the first rerun). The composite tasks got slower (median 82 s -> 96 s):
the KV task from 84 s to 103 s and the lecture speed-up from 71 s to 96 s, whose plan
lost a step but whose summary, in the 10-run evals described next, was asked again every
time.

**Two full reruns in between** aren't published here. On `62867a9` (after #15) the fleet
passed 44/48 with composites 5/9: PR #15's plan `format` field made the planner reword
"how many A fit in B" as "the ratio of A to B" (ratio task 0/3), which #16 fixed. On
`ef8febc` (after #17) it passed 45/48 with composites 9/9; all three failures were the MCP
question, whose "USB-C" line the fetch tool dropped, which #18 fixed. Between them, the
composite tasks were run **10 times each** on the fleet: after #16, lecture 10/10, KV
8/10, ratio 8/10; with #17's summary check, 10/10, 9/10 and 9/10; with all of #17, KV
10/10 and ratio 9/10 (lecture not rerun).

**Single agent:** 36 vs 34 passed is within what an unchanged agent varies by
(`v-three-claims` 0 -> 3 and `r-qwen-license` 2 -> 3, but `c-ollama-ratio` 2 -> 0 and
`r-llama-draft` 2 -> 1). Its failures are the same kinds: all 12 citing answers cited a
page it never fetched, it searched the lecture notes for the MCP question in 2 of 3 runs
(and found no definition in the third),
no timestamp was in the right window on `k-table-doubling` (0/3), and on the ratio task it
replied only `RATIO: 8.0`, without the two values the task asked to look up (0/3).

**How to read it:**

- **48/48 was partly luck.** With 3 repeats, a task that passes 90% of the time still
  scores 3/3 about 73% of the time. The 10-run composite evals above are the better
  estimate for chained tasks.
- **Still not a fresh measurement.** Every fix since the first run was written after
  reading failures on these 16 tasks and checked on them.
- **Research reads live pages,** and two of the regressions between reruns came from the
  web side (a docs page's markup, a model card's header), not from the agents.
- **No writes were possible**, as in the first rerun, and the earlier caveats still apply.

## Third rerun (2026-10-24)

The same 16 tasks, 3 repeats per system, timeouts and hardware, on `main` at `84a7c26`
with nothing uncommitted. Both systems were restarted from that commit first. Raw data:
[`m6-rerun-3/results.jsonl`](m6-rerun-3/results.jsonl),
[`m6-rerun-3/report.md`](m6-rerun-3/report.md),
[`m6-rerun-3/meta.json`](m6-rerun-3/meta.json).

What changed since the second rerun:

- a cancel that arrives before the remote task's id is known still cancels the remote
  run (#19); DomainGraph retries Ollama embedding errors that may pass, such as the
  runner crash of the first run (DomainGraph #10)
- when no search hit is a Hugging Face model page, `web_search` asks the hub's own search
  for the named model and puts its page first (#20)
- a Verifier-signed knowledge write must state a claim the Verifier supported with that
  source, not only cite the URL (#21)
- `fetch_url` reads the text layer of PDFs (#22)
- Research's answer goes back to it if it cites a URL no successful fetch returned (#23),
  or if a claim's numbers are not on the fetched pages that claim cites (#24)
- scoring: a fetched page also counts under its URL after redirects (#22). Re-scored from
  their traces, the first run and both earlier reruns are unchanged.

The single agent's spec did not change. It shares the web tools and DomainGraph, so #10,
#20 and #22 apply to it too, but it has neither of Research's new checks.

| | single, second rerun | multi, second rerun | single, now | multi, now |
|---|---|---|---|---|
| Runs passed | 36/48 | 48/48 | 33/48 | 48/48 |
| Passed, every cited page fetched | 26/48 | 48/48 | 26/48 | 48/48 |
| Checks passed | 63/78 | 78/78 | 61/78 | 78/78 |
| Median / mean time per run | 34 / 42 s | 45 / 50 s | 31 / 38 s | 40 / 49 s |
| Mean tokens per run | 6,066 | 15,895 | 5,472 | 14,635 |
| Runs that cited an unfetched page | 12 of 12 | 0 of 19 | 13 of 13 | 0 of 20 |
| Composite passed | 5/9 | 9/9 | 5/9 | 9/9 |

**Fleet:** every task passed 3/3 again, and none of its 20 citing answers cited a page it
hadn't fetched. Research got cheaper and faster (research tasks: 22.9k -> 17.4k tokens,
median 40 s -> 36 s), so the fleet costs about 2.7x the single agent's tokens (2.6x in the
second rerun). The composite tasks cost a little more (26.3k -> 28.7k tokens, median 95 s).
In the traces, Research fetched NVIDIA's Qwen3-8B model card PDF twice and read it both
times, and the hub lookup of #20 supplied the model card in all 3 ratio runs, whose
searches again returned no Hugging Face page. Replaying the citation and number checks
on the 21 answers Research kept, none would be sent back. This run's traces predate
send-back logging (#25), so whether the checks sent any answer back during the run isn't
known; that path is covered by the tests and by replays on saved runs. In an 18-run live
check after #25, the fleet's 5 send-backs all came from the required-calls check.

**One full rerun in between** isn't published here. On `43d932c` (after #19-#21 and
DomainGraph #10) the fleet passed 48/48 with composites 9/9, but only 45/48 with every
cited page fetched: two license answers cited the NVIDIA PDF after its fetch had failed,
and one ratio answer cited ssdnodes.com, which Research never fetched, for Ollama's 4,096.
All three answers were correct. The number check had let the 4,096 through because
4096 is also Qwen3-8B's `hidden_size`, on a page another claim cited. #22-#24 came out of
those three answers. The single agent passed 34/48 there.

**Single agent:** 33 vs 36 passed is within what an unchanged agent varies by:
`r-qwen-license` 3 -> 1 (two answers gave the license but not the context length) and
`r-llama-draft` 1 -> 0. Its failures are the same kinds: all 13 citing answers cited a
page it never fetched, it searched the lecture notes for the MCP question in 2 of 3 runs,
no timestamp was in the right window on `k-table-doubling` (0/3), and on the ratio task
it replied only `RATIO: 8.0` (0/3).

**How to read it:**

- **The fleet has passed 144 of 144 runs** across the last three full reruns, but the code
  changed between them, the repeats aren't independent (the planner runs at temperature
  0), and the last 10-run composite evals, before #19-#24, put it at about 90-95% on
  chained tasks.
- **The new checks weren't seen working live.** No kept answer needed them, so this rerun
  shows they don't get in the way, not that they catch anything.
- **Still not a fresh measurement,** and **no writes were possible**, so #21 wasn't
  exercised; nothing cancels a run in the eval either (#19). The earlier caveats still
  apply.

## Citation-integrity chain (2026-10-27)

The three reruns above each passed the fleet 48/48 with "every cited page fetched", but
that line hid two separate weaknesses that later work exposed and closed. The first was a
real hole in the guard; the rest were the guard not covering every path, and the eval
mis-scoring citations so the metric couldn't be trusted. The seven PRs below finish the
citation guard end to end. Unlike the earlier prompt tuning, these are code changes with
unit tests, so they are not tuned to these 16 tasks.

**What each PR fixed:**

- **#28** — the required-calls reminder (a one-line status of what's still missing, shown
  before the model answers) cut first-try required-call send-backs to zero on a 27-run
  fleet baseline. Not a citation fix, but it shipped in the same batch.
- **#29** (+ AgentOS #11) — the Analyst's output format and a `done_reason` the guard
  needs to tell an empty-because-cut-off reply from an empty-because-nothing one.
- **#30** — compaction was laundering a search hit into a citation. When AgentOS compacts
  a long history it folds old tool results into one summary *user* message; a search hit
  for `neuralgist.com` in that summary was then read as a URL the task itself gave, so the
  citation check let it through. `AnswerGuardLLM._history` now reconstructs "what the task
  gave" from the first user message plus the tool results the guard actually saw, never
  the compaction summary.
- **#31** — a citation the nudge budget can't fix is now stripped. `max_nudges` bounds
  retries, so a model that keeps citing a page it never fetched (swapping one unfetched
  URL for another each retry) eventually gets its answer through. When the budget is spent
  and the final answer still cites an unfetched page, the guard removes that citation
  (`strip_citations`) rather than ship it, and records a `citations_stripped` trace event.
  Citations only — a bad number or quote is load-bearing in the sentence.
- **#32** — the orchestrator's combined answer was unguarded. `check_citations` is only on
  `research.toml`, so the orchestrator could relay or invent a citation no step grounded.
  It now drops any citation the final answer makes that no completed step's answer carries
  (and the task didn't give): a Research sub-answer has already passed Research's own
  guard, so "cited by some step" is a sound ground truth without an A2A protocol change.
  Same PR: the eval now flags a run `citations_unscoreable` when it cited pages but no
  located trace did any `fetch_url` — the "cited an unfetched page" metric was silently
  blaming every URL whenever the eval's `--trace-root` didn't match the fleet's
  `--trace-dir` and the agents' fetch traces weren't found.
- **#33** — the single agent got the same `check_citations = "fetch_url"` guard as
  Research. It only bites when the answer cites a URL, so solo's compute and lecture-note
  tasks are unaffected.
- **#34** — a citation whose URL contains a comma or bracket (the `_URL` regex treats those
  as terminators) was extracted clipped and never string-matched its fetched entry, so the
  eval reported a phantom unfetched citation. `citation_fetched` now accepts a cited URL
  that is a prefix of a fetched one at a clip-char boundary, used by both the guard and the
  scorer so they agree.

**The earlier 48/48s were partly an artifact.** A full rerun on `16c16b5` with the eval's
`--trace-root` pointed at a fresh directory, while the fleet was already writing to
`runs/fleet-traces`, scored the entire fleet `citations_unscoreable` (8 of 8 on a focused
`c-ollama-ratio` batch): the orchestrator's own fetch set is empty — fetching happens in
the Research A2A sub-process, in a different trace dir — so `cited - fetched` flagged
everything the moment the agents' traces weren't found. #32's flag makes that visible
instead of silent; the fix is to run the eval with `--trace-root` equal to the fleet's
`--trace-dir`.

**Matched-trace-root full rerun**, 16 tasks, 3 repeats per system, same timeouts and
hardware, on `main` at `243dd32` (before #33/#34), with
`--trace-root runs/fleet-traces` matching the fleet. Both systems restarted from that
commit first.

| | single, third rerun | multi, third rerun | single, now | multi, now |
|---|---|---|---|---|
| Runs passed | 33/48 | 48/48 | 36/48 | 48/48 |
| Passed, every cited page fetched | 26/48 | 48/48 | 27/36* | 48/48 |
| Citations unscoreable | — | — | 12 (single) | none |
| Runs citing an unfetched page | 13 of 13 | 0 of 20 | 0 of 2* | 0 of 19 |
| Composite passed | 5/9 | 9/9 | 6/9 | 9/9 |

\* The single agent ran **unguarded** here (this rerun predates #33). 12 of its research
runs cited a page they only searched (`fetched = 0`, only `web_search` in the trace), so
the eval now marks those `citations_unscoreable` and scores the citation rate over the
remaining scoreable runs rather than mislabelling them "cited an unfetched page". With the
trace roots matched, **the fleet's citations were fully scoreable for the first time** and
it still cited no unfetched page. The compaction case `c-ollama-ratio` rep 1 had its
Research citation guard fire live (a `neuralgist.com` search hit caught post-compaction),
which the earlier contaminated run had missed only because that worker was running stale
pre-#30 code after a restart/`uv sync` race, not because #30 was wrong.

**Solo guard-check**, after #33, on `main` at `83fc745`: the three research tasks that had
been unguarded (`r-qwen-license`, `r-ollama-port`, `r-a2a-origin`), 2 repeats each.

| run | before (#33 absent) | after (#33) |
|---|---|---|
| all 6 | `fetched = 0`, unscoreable, cited a searched page | guard fired (`answer_sent_back` citations, 6 of 6), `fetched = 1-5`, all scoreable |

The single agent now behaves like Research: it tries to cite a searched page, the guard
sends it back, and the model fetches the page before citing it. Four of the six ended with
no unfetched citation; the other two kept one through nudge exhaustion — one of those was
`http://localhost:11434` (an address the answer names, not a web page), and the other,
`r-qwen-license` rep 0, was the `#34` comma-clip artifact: the cited llm-explorer URL *was*
fetched, its `,<token>` suffix just made the clipped citation miss its fetched entry. #34
makes that score correctly.

**How to read it:**

- **The guard is now verified live, not only by replay.** The third rerun's new checks
  "weren't seen working" because no kept answer needed them; here the compaction guard
  fired on `c-ollama-ratio` and the solo guard fired on all 6 runs. The `citations_stripped`
  path is still only exercised by unit tests — no live run spent its whole nudge budget
  still citing an unfetched page.
- **These are code fixes with tests**, so unlike the prompt tuning in the earlier reruns
  they aren't tuned to these 16 tasks. The guard, strip and match logic each have unit
  tests (`tests/test_m2.py`, `tests/test_chained.py`, `tests/test_m6.py`).
- **The matched-trace-root run is the one to trust for citations.** The earlier
  "every cited page fetched 48/48" lines were only meaningful when the roots happened to
  match; with #32 a mismatch now shows as `citations_unscoreable` rather than a false
  pass or a false fail.
- The earlier caveats (16 self-written tasks, repeats that aren't independent at
  temperature 0, guards as a confound, tokens across two model sizes, read-only task set)
  still apply.

## Reproducing

```powershell
uv run agentmesh up --trace-dir runs/fleet-traces
uv run agentmesh serve agents/solo.toml --trace-dir runs/fleet-traces/solo
uv run agentmesh eval run --out runs/m6/full --repeats 3 --timeout 1200 --step-timeout 600 --trace-root runs/fleet-traces
uv run agentmesh eval report runs/m6/full
```

Pass `--trace-root` equal to the fleet's `--trace-dir` (both `runs/fleet-traces` above),
or the eval won't find the agents' fetch traces and will score every citation
`citations_unscoreable` (see the 2026-10-27 section). Re-running `eval run` into the same
folder resumes where it stopped.
