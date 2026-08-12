"""Modules more than one stage script needs.

A stage script is invoked as a plain file, so ``sys.path[0]`` is its own stage directory and a
sibling stage's modules are not importable. Every launcher instead puts the ``stages`` directory on
``PYTHONPATH``, which makes this package reachable from any stage as ``common.<module>``.
"""
