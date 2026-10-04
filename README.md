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

## Development

```powershell
uv sync
uv run pre-commit install
uv run pytest
uv run ruff check .
```
