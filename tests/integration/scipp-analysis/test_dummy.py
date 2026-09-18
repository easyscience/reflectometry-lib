# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import pytest


@pytest.mark.fast
def test_dummy_fast():
    calculated = 2 + 2
    expected = 4
    assert calculated == expected


@pytest.mark.slow
def test_dummy_slow():
    calculated = sum(i * j for i in range(10000) for j in range(10000))
    expected = 2499500025000000
    assert calculated == expected
