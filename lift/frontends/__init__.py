"""Lift front-ends: parse + elaborate a framework's native config into TrainIR.

Front-ends never import the framework (specs/015-lift/DESIGN.md §6): they
parse the config text and apply a semantic table sourced from reading the
framework's code, with the oracle + gates as the safety net.
"""
