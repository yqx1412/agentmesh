import agentmesh


def test_version() -> None:
    assert agentmesh.__version__ == "0.1.0"


def test_main_runs(capsys) -> None:
    agentmesh.main()
    assert "agentmesh" in capsys.readouterr().out
