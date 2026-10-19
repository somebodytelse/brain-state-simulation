"""Compatibility test entry point; use unittest discover -s tests."""
import importlib
import unittest

def load_tests(loader, suite, pattern):
    if pattern is not None:
        return loader.suiteClass()
    return loader.loadTestsFromModule(importlib.import_module("tests.test_connectivity"))

if __name__ == "__main__":
    unittest.main()
