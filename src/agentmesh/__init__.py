"""agentmesh: Multi-agent system over A2A: an orchestrator plus specialist agents."""

__version__ = "0.1.0"


def main(argv: list[str] | None = None) -> int:
    from agentmesh.cli import main as cli_main

    return cli_main(argv)
