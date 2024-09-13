import pytest

import os

def pytest_addoption(parser):
    parser.addoption(
            "--runslow"
            , action="store_true"
            , default=False
            , help="run slow tests"
            )
    parser.addoption(
            "--onlyselected"
            , action="store_true"
            , default=False
            , help="run only selected tests"
            )


def pytest_collection_modifyitems(config, items):
    if(config.getoption("--runslow")
            and config.getoption("--onlyselected")):
        return

    if(not config.getoption("--onlyselected")):
        skip_slow = pytest.mark.skip(reason="slow test")
        run_slow = config.getoption("--runslow")
        for item in items:
            if not run_slow:
                if "slow" in item.keywords:
                    item.add_marker(skip_slow)
    else:
        skip_not_selected = pytest.mark.skip(reason="not selected")
        for item in items:
            if not "selected" in item.keywords:
                item.add_marker(skip_not_selected)

