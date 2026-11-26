"""Training gate utilities.

Prevents training pipelines from proceeding when certification has failed.
"""


def require_certified(report):
    passed = getattr(report, "passed", False)
    if not passed:
        raise RuntimeError("Training blocked: simulator/data certification failed")
    return True
