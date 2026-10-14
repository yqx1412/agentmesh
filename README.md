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
- **No authentication:** the server binds to `127.0.0.1` unless the spec sets
  `host`. Don't expose it on a network.
- `--trace-dir` writes an AgentOS trace per task; `ask --no-stream` sends one
  blocking request instead of streaming.

## M2: four specialist agents

`agentmesh up` starts four specialists, each as its own A2A process, and waits
until every Agent Card is served. Ctrl+C stops them all.

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
uv run agentmesh orchestrate "Research local LLM inference and update my knowledge base"
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

## Development

```powershell
uv sync
uv run pre-commit install
uv run pytest
uv run ruff check .
```
