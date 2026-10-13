"""Localização de diretórios do repositório, independente da profundidade do script.

Antes desta função cada script de análise fazia `Path(__file__).resolve().parents[3]`,
o que amarra o script à sua profundidade na árvore: mover o arquivo um nível quebra
silenciosamente a resolução (o caminho continua existindo, só aponta para o lugar
errado). `repo_root()` sobe procurando um marcador, então funciona de qualquer lugar.
"""

from __future__ import annotations

from pathlib import Path

_MARKERS = ("README.md", ".git")


def repo_root(start: str | Path | None = None) -> Path:
    """Sobe a partir de `start` (default: este arquivo) até achar a raiz do repo."""
    here = Path(start).resolve() if start else Path(__file__).resolve()
    if here.is_file():
        here = here.parent
    for candidate in (here, *here.parents):
        if any((candidate / m).exists() for m in _MARKERS):
            return candidate
    raise RuntimeError(
        f"raiz do repositório não encontrada a partir de {here} "
        f"(procurando por {' ou '.join(_MARKERS)})"
    )


def project_dir(project: str) -> Path:
    """`projects/<project>/` — a casa de um paper e da análise que o sustenta."""
    return repo_root() / "projects" / project


def results_dir(project: str | None = None) -> Path:
    """Raiz das saídas de run.

    Com `project`, devolve `results/<project>/` — o segmento que evita que dois
    experimentos colidam no mesmo namespace plano. Sem, devolve `results/`, que é
    onde vivem os 107 runs anteriores à convenção.
    """
    root = repo_root() / "results"
    return root / project if project else root


def campaign_dir(project: str, campaign: str) -> Path:
    """`results/<project>/<campaign>/` — uma leva de runs que responde UMA pergunta.

    O nível de campanha existe porque o timestamp não basta: ele separa invocações, mas
    não diz o que a leva respondia nem garante que duas levas sejam comparáveis. Uma
    campanha agrupa os estágios de um mesmo experimento (controle, tratamento, ablações),
    ainda que rodados em dias diferentes, e carrega um `campaign.json` com a impressão
    digital do config — o que impede repetir o acidente do `grid_extend` no p1, em que
    duas populações acabaram em grades de preço diferentes sem registro.
    """
    return results_dir(project) / campaign


def run_dir(experiment_name: str, seed: int | str, timestamp: str,
            project: str | None = None) -> Path:
    """Caminho canônico de um run: `<results>/<exp>__<seed>__<timestamp>/`.

    Fonte única do nome. Antes, cada script de sweep montava o seu próprio prefixo,
    então não havia como saber, a partir do nome, a que experimento um diretório
    pertencia.
    """
    return results_dir(project) / f"{experiment_name}__{seed}__{timestamp}"
