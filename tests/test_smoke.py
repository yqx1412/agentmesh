import agentmesh


def test_version() -> None:
    assert agentmesh.__version__ == "0.1.0"


def test_main_prints_help(capsys) -> None:
    assert agentmesh.main([]) == 0
    assert "agentmesh" in capsys.readouterr().out
