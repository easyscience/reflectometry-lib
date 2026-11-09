# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""The names the changelog and tutorials tell users to import."""

import easyreflectometry


def test_minimizers_is_exported():
    from easyscience import Minimizers

    assert easyreflectometry.Minimizers is Minimizers
    assert 'Minimizers' in easyreflectometry.__all__
