"""Collectors: campaign data, local first, then on the shared volume.

``python3 -m arbor.collect check|project|realtime|knowledge`` (``__main__``).
The skills that drive them: ``.claude/skills/realtime-collector/`` and
``.claude/skills/knowledge-collector/``; ``arbor/dashboard/`` reads what they
write (``.claude/skills/dashboard/``). Standard library only, so they run
wherever a campaign runs.
"""
