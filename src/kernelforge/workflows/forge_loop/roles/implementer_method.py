# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How a gated Implementer session is told to work: its method, its pacing and the time it has."""

from __future__ import annotations

# What a session has read, built and measured is paid for again by the next one, so a session whose gate requires
# performance keeps working until its clock or its own measurements tell it to finish. Its first direction is the
# round's plan when the round planned one, and otherwise the one it expects most from, structural changes included.
#
# How such a session paces itself. The wrap-up minute is this fraction of its wall-clock budget; the rest covers the
# session's final check and the gate's own correctness and benchmark runs.
SESSION_WRAP_UP_FRACTION = 0.8
# Measured changes in a row without a new session best before a direction is left, and before the session wraps up
# whatever the clock says.
DIRECTION_STALL_CHANGES = 8
SESSION_STALL_CHANGES = 12
# With more than this many minutes before the wrap-up minute, the time left is no reason to pass over a direction.
AMPLE_TIME_MINUTES = 20
# The longest one measurement command may run. A longer one outlasts the provider's prompt cache, so the request that
# follows it pays for the whole conversation again.
MAX_COMMAND_MINUTES = 4

PERFORMANCE_WORKING_METHOD = """\
2. Fix this session's direction before the first edit. When this invocation
   names a required optimization plan, that plan is the direction: read it and
   work it. When it leaves the direction to you, choose it from the evidence in
   this prompt — where each scored case spends its time, what earlier sessions
   tried and what they measured — and pick the ONE direction you expect to move
   the score most: a mechanism (restructure a loop, fuse two passes, change the
   data layout, specialize per case), not a single constant. Take the larger
   expected gain even when it means restructuring the kernel. Either way, a
   direction may need twenty or thirty minutes of work before its first
   measurement: the session is long enough for that, and tuning constants
   follows a structural change, it does not replace one. State the direction
   in one or two sentences: what it is, which cases it should move, and the
   per-case speedup and resulting score you expect. That number is this
   session's goal.
3. Measure before you change anything. Run
     `python3 {driver_run} --warmup 3 --iters 20 --bench-mode; date`
   on the unchanged kernel and note the per-case times. They are this session's
   reference: only two measurements taken in the same session compare cleanly.
   End every later driver command with `; date` as well, so each measurement
   carries the time it was taken.
4. Work the direction as a loop, one change at a time:
     Edit → `python3 {driver_run}` (full-suite correctness verdict)
          → `python3 {driver_run} --warmup 3 --iters 20 --bench-mode`
          → compare the per-case times with the best version this session measured.
   - A change that fails correctness or raises a compile error is not finished:
     READ the error and FIX it.
   - A change that measurably helps becomes the session's best. Copy every file
     it touches to a scratch directory outside the workspace, with its per-case
     numbers beside it.
   - A change that does not help is undone at once: restore the best version's
     files from that scratch directory before you do anything else. Outside the
     minute or two in which one change is being measured, the workspace holds
     the best version this session has measured, because that is what the loop
     validates if the session is cut off. Do not save or restore work with git
     (`stash`, `add`, `commit`, `reset`): the loop owns the repository state and
     rejects a session that changes it.
   - Judge a change by the score defined above, not by raw `mean_ms` and not
     by a single case. A difference inside the spread of your own repeated
     measurements is not a gain.
   - Run every measurement in the foreground and keep each command under
     {max_command_minutes} minutes; split a longer sweep into several commands.
     A measurement left in the background keeps the session waiting on it, and
     that wait is paid for twice: on the clock, and when the whole conversation
     is read again after the pause.
5. A first improvement is where the session starts, not where it ends. It shows
   the direction is real, so push it: tune the parameters it introduced, carry
   it to the cases it has not reached, remove the overhead it exposed. Leave a
   direction when measurement has disproved its mechanism, or when
   {direction_stall} measured changes in a row have not improved on the
   session's best; then take the next most valuable direction. While more than
   {ample_minutes} minutes remain before the wrap-up minute, "not enough time"
   and "too risky" are not reasons to pass over a direction: the best version
   is safe in the scratch directory, so take the direction and drop it at the
   wrap-up minute if it has not landed.
6. Read the clock, never estimate it: the `date` after each driver command is
   the only clock, and elapsed time judged from memory is unreliable. The
   "Session budget" section of this invocation names the wrap-up minute. Go to
   step 7 when the clock reaches it, when {session_stall} measured changes in a
   row — across directions — have not improved on the session's best, or when
   every direction worth measuring has been measured. In the last case name, in
   your final message, each direction you did not take and the measurement that
   rules it out. A first improvement is none of these three.
7. Submit the best version, not the last one. Make sure the workspace holds
   exactly the best version you measured, and run correctness and the benchmark
   on it once more; the gate that judges your stop then repeats correctness and
   takes three benchmark measurements. If nothing you measured beats the
   current best, restore every file to how the session found it and say so in
   the `PLAN:` line.
8. There is NO hard edit budget — make as many changes to non-protected tracked
   implementation files as a direction needs. What IS bounded is fruitless
   stopping: each time you STOP with a candidate that is correct but not faster
   than best, the gate sends you back to keep improving, and after {max_blocks}
   such rejected stops it ends the session and hands off. So do not circle
   micro-tweaks inside the noise band — if a direction is not beating best,
   switch to a materially different one rather than resubmitting near-ties.
"""

# A correctness-only gate (the PORT phase) has no performance to search for: the first correct kernel is the result.
CORRECTNESS_ONLY_WORKING_METHOD = """\
2. Use Edit to make a change with a clear hypothesis.
3. VERIFY IT YOURSELF with Bash before finishing, using the driver documented
   above (run from the current directory):
     `python3 {driver_run}` → full-suite correctness verdict
   If correctness fails or a compile error is raised, READ the error and FIX it —
   do not stop on a broken kernel.
4. This phase is judged on correctness alone. As soon as the kernel passes the
   driver's correctness verdict, stop and submit it to the outer loop, which is
   the sole authority for canonical correctness. Do not spend the rest of the
   Session tuning performance.
5. There is NO hard edit budget — make as many changes to non-protected tracked
   implementation files as the kernel needs, verifying as you go. What IS
   bounded is fruitless stopping: each time you STOP with a kernel that fails
   correctness, the gate sends you back to fix it, and after {max_blocks} such
   rejected stops it ends the session and hands off.
"""


def render_method(template: str, *, driver_run: str, max_blocks: int) -> str:
    """One of the two working methods above, with the driver command and every threshold it names filled in."""
    return template.format(
        driver_run=driver_run,
        max_blocks=max_blocks,
        direction_stall=DIRECTION_STALL_CHANGES,
        session_stall=SESSION_STALL_CHANGES,
        ample_minutes=AMPLE_TIME_MINUTES,
        max_command_minutes=MAX_COMMAND_MINUTES,
    )


def deadline_section(deadline_sec: int) -> str:
    """Tell the session its own wall-clock bound."""
    deadline_min = max(1, round(deadline_sec / 60))
    return (
        "## Session deadline\n"
        f"You have about {deadline_min} minutes of wall-clock for "
        "THIS session before it is cut off; a session cut off mid-turn hands "
        "off nothing. Well before then, stop improving and submit the best "
        "candidate you actually have through the clean handoff "
        "(candidate_submitted) — do not reach for a larger gain you cannot "
        "land in time.\n"
    )


def session_budget_section(deadline_sec: int) -> str:
    """Tell a session that searches for performance how long it has, and the minute at which it starts to finish."""
    deadline_min = max(1, round(deadline_sec / 60))
    wrap_up_min = max(1, int(deadline_min * SESSION_WRAP_UP_FRACTION))
    return (
        "## Session budget\n"
        "Run `date` before anything else: that reading is minute 0 of this session. "
        f"You have about {deadline_min} minutes of wall-clock from it, and the session is meant to use "
        "them: reading the kernel, building a way to measure and reaching a first result are paid for once, so each "
        "further direction costs less here than it would in a new session. "
        f"The wrap-up minute is {wrap_up_min}: work your directions until the clock reaches it, then finish as "
        'step 7 of "How to work" describes. '
        f"At minute {deadline_min} the session is cut off, and the loop validates whatever tracked changes the "
        "workspace holds at that moment.\n"
    )
