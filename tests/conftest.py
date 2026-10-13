"""
Configuração compartilhada da suíte de testes.

O ponto principal aqui é impedir que os testes escrevam no log de experimentos.
`multi_agent_output_writer.write` chama `_append_experiments_log`, que anexa em
`research/experiments_log.md` — um arquivo **versionado**. Qualquer teste que passe
`full_config` ao trainer acaba registrando um "experimento" apontando para um diretório
temporário do pytest, que já não existe quando alguém for ler.

Foram 24 entradas de lixo numa única rodada de `make test`.
"""

import pytest

from src.general_utils import multi_agent_output_writer


@pytest.fixture(autouse=True, scope="session")
def _no_experiments_log():
    """Desliga o append em `research/experiments_log.md` durante os testes.

    `autouse` + `scope="session"`: vale para toda a suíte sem que cada teste precise
    lembrar de pedir. O log é registro de experimentos de verdade, não de smoke tests.
    """
    original = multi_agent_output_writer._append_experiments_log
    multi_agent_output_writer._append_experiments_log = lambda *a, **k: None
    try:
        yield
    finally:
        multi_agent_output_writer._append_experiments_log = original
