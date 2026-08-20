from __future__ import annotations

import chz
from typing_extensions import override

from paperbench.nano.task import PBTask
from paperbench.solvers.piagent.solver import PiAgentSolver


@chz.chz
class PiVanillaAgentSolver(PiAgentSolver):
    """Pi CLI solver that uses the raw official PaperBench prompt and no skills."""

    @override
    def shortname(self) -> str:
        return "piagent-vanilla"

    def _pi_use_skills(self) -> bool:
        return False

    @override
    def _prompt(self, task: PBTask) -> str:
        if self.loop_enabled or self.continuation_enabled:
            return super()._prompt(task)
        return self._prompt_instructions(task.judge.code_only)
