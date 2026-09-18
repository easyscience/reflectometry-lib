# SPDX-FileCopyrightText: 2026 EasyReflectometry contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the Bayesian analysis module."""

import numpy as np
import pytest


@pytest.fixture
def sample_draws():
    """Generate synthetic posterior draws for testing."""
    rng = np.random.default_rng(42)
    n_samples = 100
    # Two parameters: 'thickness' and 'sld'
    thickness = rng.normal(loc=250, scale=10, size=n_samples)
    sld = rng.normal(loc=2.0, scale=0.2, size=n_samples)
    draws = np.column_stack([thickness, sld])
    param_names = ['Film_thickness', 'Film_sld']
    return draws, param_names


class TestPosteriorSummary:
    def test_returns_string(self, sample_draws):
        from easyreflectometry.analysis.bayesian import posterior_summary

        draws, param_names = sample_draws
        result = posterior_summary(draws, param_names)
        assert isinstance(result, str)
        assert 'parameter' in result
        assert 'mean' in result
        assert 'sd' in result

    def test_header_uses_quantile_labels(self, sample_draws):
        """The header should label the columns as equal-tailed quantiles, not HDI."""
        from easyreflectometry.analysis.bayesian import posterior_summary

        draws, param_names = sample_draws
        result = posterior_summary(draws, param_names)
        header = result.splitlines()[0]
        assert 'q2.5%' in header
        assert 'q97.5%' in header
        assert 'hdi' not in header.lower()

    def test_contains_param_names(self, sample_draws):
        from easyreflectometry.analysis.bayesian import posterior_summary

        draws, param_names = sample_draws
        result = posterior_summary(draws, param_names)
        for name in param_names:
            assert name in result


class TestCredibleIntervals:
    def test_returns_dict(self, sample_draws):
        from easyreflectometry.analysis.bayesian import credible_intervals

        draws, param_names = sample_draws
        result = credible_intervals(draws, param_names)
        assert isinstance(result, dict)
        for name in param_names:
            assert name in result
            lo, hi = result[name]
            assert lo < hi

    def test_alpha_95_coverage(self, sample_draws):
        from easyreflectometry.analysis.bayesian import credible_intervals

        draws, param_names = sample_draws
        result = credible_intervals(draws, param_names, alpha=0.95)
        for i, name in enumerate(param_names):
            lo, hi = result[name]
            # 95% interval should contain at least 90% of samples
            col = draws[:, i]
            inside = np.sum((col >= lo) & (col <= hi))
            assert inside / len(col) >= 0.90

    def test_alpha_50_narrower(self, sample_draws):
        from easyreflectometry.analysis.bayesian import credible_intervals

        draws, param_names = sample_draws
        ci_95 = credible_intervals(draws, param_names, alpha=0.95)
        ci_50 = credible_intervals(draws, param_names, alpha=0.50)
        for name in param_names:
            assert (ci_95[name][1] - ci_95[name][0]) > (ci_50[name][1] - ci_50[name][0])


class TestPosteriorResults:
    def test_repr(self, sample_draws):
        from easyreflectometry.analysis.bayesian import PosteriorResults

        draws, param_names = sample_draws
        pr = PosteriorResults(draws, param_names)
        rep = repr(pr)
        assert 'PosteriorResults' in rep
        assert str(draws.shape[0]) in rep

    def test_summary_delegates(self, sample_draws):
        from easyreflectometry.analysis.bayesian import PosteriorResults

        draws, param_names = sample_draws
        pr = PosteriorResults(draws, param_names)
        summary_str = pr.summary()
        assert isinstance(summary_str, str)
        assert 'parameter' in summary_str

    def test_credible_interval_delegates(self, sample_draws):
        from easyreflectometry.analysis.bayesian import PosteriorResults

        draws, param_names = sample_draws
        pr = PosteriorResults(draws, param_names)
        ci = pr.credible_interval(alpha=0.95)
        assert isinstance(ci, dict)
        for name in param_names:
            assert name in ci


class TestPosteriorPredictiveReflectivity:
    def test_returns_tuples(self, sample_draws):
        """Test with a mock model that returns a constant array."""
        from unittest.mock import MagicMock

        from easyreflectometry.analysis.bayesian import posterior_predictive_reflectivity

        draws, param_names = sample_draws
        mock_model = MagicMock()
        mock_model.unique_name = 'test_model'
        mock_model.interface = MagicMock()
        mock_model.interface.fit_func = MagicMock(return_value=np.ones(50))
        mock_model.get_parameters = MagicMock(return_value=[])

        q_values = np.linspace(0.01, 0.3, 50)
        median, lower, upper = posterior_predictive_reflectivity(
            draws,
            param_names,
            mock_model,
            q_values,
            n_samples=20,
        )
        assert median.shape == (50,)
        assert lower.shape == (50,)
        assert upper.shape == (50,)


class TestPosteriorPredictiveSLDProfile:
    def test_returns_tuples(self, sample_draws):
        """Test with a mock model that returns constant z and sld."""
        from unittest.mock import MagicMock

        from easyreflectometry.analysis.bayesian import posterior_predictive_sld_profile

        draws, param_names = sample_draws
        mock_model = MagicMock()
        mock_model.unique_name = 'test_model'
        mock_model.interface = MagicMock()
        mock_model.interface.sld_profile = MagicMock(return_value=(np.linspace(0, 500, 100), np.ones(100) * 2.0))
        mock_model.get_parameters = MagicMock(return_value=[])

        z, median, lower, upper = posterior_predictive_sld_profile(
            draws,
            param_names,
            mock_model,
            n_samples=20,
        )
        assert z.shape == (100,)
        assert median.shape == (100,)
        assert lower.shape == (100,)
        assert upper.shape == (100,)


class TestCornerPlot:
    def test_plot_corner_returns_plotly_figure(self, sample_draws):
        """plot_corner returns a Plotly Figure built from posterior draws."""
        try:
            from plotly.graph_objects import Figure

            from easyreflectometry.analysis.bayesian import plot_corner
        except ImportError:
            pytest.skip('plotly not installed')

        draws, param_names = sample_draws
        fig = plot_corner(draws, param_names)
        assert isinstance(fig, Figure)
        assert len(fig.data) > 0


class TestSaveRestoreParameterState:
    def test_save_and_restore(self):
        """Test that parameter state save/restore works correctly."""
        from easyreflectometry.analysis.bayesian import _restore_parameter_state
        from easyreflectometry.analysis.bayesian import _save_parameter_state

        # Use simple objects that support attribute assignment
        class MockParam:
            def __init__(self, unique_name, raw_value, error, independent=True):
                self.unique_name = unique_name
                self.value = raw_value
                self.error = error
                self.independent = independent

        param1 = MockParam('param_a', 1.5, 0.1)
        param2 = MockParam('param_b', 3.0, 0.2)

        class MockModel:
            def get_parameters(self):
                return [param1, param2]

        model = MockModel()

        state = _save_parameter_state(model)
        assert state['param_a'] == (1.5, 0.1)
        assert state['param_b'] == (3.0, 0.2)

        # Modify values
        param1.raw_value = 99.0
        param1.value = 99.0
        param2.raw_value = 99.0
        param2.value = 99.0

        _restore_parameter_state(model, state)
        assert param1.value == 1.5
        assert param1.error == 0.1
        assert param2.value == 3.0
        assert param2.error == 0.2

    def test_dependent_parameters_are_left_alone(self):
        """Derived parameters cannot be written back, and do not need to be."""
        from easyreflectometry.analysis.bayesian import _restore_parameter_state
        from easyreflectometry.analysis.bayesian import _save_parameter_state

        class MockParam:
            def __init__(self, unique_name, independent):
                self.unique_name = unique_name
                self.value = 1.0
                self.error = 0.1
                self.independent = independent

            def __setattr__(self, name, value):
                if name in ('value', 'error') and not self.__dict__.get('independent', True):
                    raise AttributeError(f'This is a dependent parameter, its {name} cannot be set directly.')
                super().__setattr__(name, value)

        free = MockParam('free', True)
        derived = MockParam('derived', False)

        class MockModel:
            def get_parameters(self):
                return [free, derived]

        model = MockModel()
        state = _save_parameter_state(model)

        assert 'derived' not in state
        _restore_parameter_state(model, state)  # would raise if it wrote to `derived`
        assert free.value == 1.0


class TestApplyDraw:
    def test_apply_draw_updates_parameters(self):
        """Test that _apply_draw sets parameter values correctly."""
        from easyreflectometry.analysis.bayesian import _apply_draw

        class MockParam:
            def __init__(self, unique_name):
                self.unique_name = unique_name
                self.value = None

        param_a = MockParam('thickness')
        param_b = MockParam('sld')

        class MockModel:
            def get_parameters(self):
                return [param_a, param_b]

        model = MockModel()
        draws = np.array([[250.0, 2.0], [260.0, 2.1]])
        param_names = ['thickness', 'sld']

        _apply_draw(model, draws, param_names, row=0)
        assert param_a.value == 250.0
        assert param_b.value == 2.0

        _apply_draw(model, draws, param_names, row=1)
        assert param_a.value == 260.0
        assert param_b.value == 2.1


class TestGelmanRubinRequiresMultipleChains:
    def test_raises_on_2d_draws(self, sample_draws):
        """R-hat is undefined for a single chain; ``gelman_rubin`` must reject 2-D input."""
        pytest.importorskip('arviz')
        from easyreflectometry.analysis.bayesian import PosteriorResults

        draws, param_names = sample_draws  # shape (n_samples, n_params)
        pr = PosteriorResults(draws, param_names)
        with pytest.raises(ValueError, match='at least 2 chains'):
            pr.gelman_rubin()

    def test_raises_on_single_chain_3d(self, sample_draws):
        """Even with an explicit chain axis, n_chains == 1 must raise."""
        pytest.importorskip('arviz')
        from easyreflectometry.analysis.bayesian import PosteriorResults

        draws, param_names = sample_draws
        single_chain = draws[np.newaxis, ...]  # (1, n_draws, n_params)
        pr = PosteriorResults(single_chain, param_names)
        with pytest.raises(ValueError, match='at least 2 chains'):
            pr.gelman_rubin()

    def test_accepts_multi_chain(self, sample_draws, monkeypatch):
        """With n_chains >= 2 the diagnostic should forward to arviz and return its values."""
        pytest.importorskip('arviz')
        from easyreflectometry.analysis import bayesian as bayesian_mod
        from easyreflectometry.analysis.bayesian import PosteriorResults

        draws, param_names = sample_draws
        rng = np.random.default_rng(7)
        second = np.column_stack([
            rng.normal(loc=250, scale=10, size=draws.shape[0]),
            rng.normal(loc=2.0, scale=0.2, size=draws.shape[0]),
        ])
        multi = np.stack([draws, second], axis=0)  # (2, n_draws, n_params)

        # Stub arviz.rhat so the test verifies the wrapper's contract (≥2 chains
        # accepted, results unpacked per parameter) without depending on arviz's
        # small-sample numerics, which can raise platform-specific TypeErrors.
        class _FakeRhatVar:
            def __init__(self, value: float) -> None:
                self.values = np.array(value)

        fake_rhat = {name: _FakeRhatVar(1.01 + 0.001 * i) for i, name in enumerate(param_names)}
        monkeypatch.setattr(bayesian_mod._arviz, 'rhat', lambda _data: fake_rhat)

        pr = PosteriorResults(multi, param_names)
        result = pr.gelman_rubin()
        assert isinstance(result, dict)
        for i, name in enumerate(param_names):
            assert name in result
            assert result[name] == pytest.approx(1.01 + 0.001 * i)


class TestPlotFigureFallbackWarnings:
    """When ``return_figure=True`` and plotly is missing, the helpers must warn."""

    def test_plot_trace_warns_without_plotly(self, sample_draws, monkeypatch):
        import builtins

        from easyreflectometry.analysis.bayesian import plot_trace

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name.startswith('plotly'):
                raise ImportError('plotly disabled for test')
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', _fake_import)

        draws, param_names = sample_draws
        with pytest.warns(UserWarning, match='plotly'):
            result = plot_trace(draws, param_names, return_figure=True)
        assert result is None

    def test_plot_distribution_warns_without_plotly(self, sample_draws, monkeypatch):
        import builtins

        from easyreflectometry.analysis.bayesian import plot_distribution

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name.startswith('plotly'):
                raise ImportError('plotly disabled for test')
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', _fake_import)

        draws, param_names = sample_draws
        with pytest.warns(UserWarning, match='plotly'):
            result = plot_distribution(draws, param_names, return_figure=True)
        assert result is None


# ===================================================================
# Persistence helpers — save_posterior / load_posterior
# ===================================================================


class TestSaveLoadPosterior:
    """Tests for ``save_posterior`` and ``load_posterior``."""

    @pytest.fixture
    def mock_posterior_results(self, sample_draws):
        """Build a PosteriorResults with a real-looking mocked sampler_state."""
        from unittest.mock import MagicMock

        from easyreflectometry.analysis.bayesian import PosteriorResults

        draws, param_names = sample_draws

        # Build a mock that passes isinstance(obj, MCMCDraw) for the
        # type guard in save_posterior.  We use a non-spec MagicMock and
        # reassign its __class__ so isinstance succeeds.
        import bumps.dream.state as _bds

        mock_state = MagicMock()
        mock_state.__class__ = _bds.MCMCDraw

        mock_state.Nvar = draws.shape[1]
        mock_state.Npop = 5
        mock_state.labels = [f'p{name}' for name in param_names]
        mock_draw = MagicMock()
        mock_draw.points = draws
        mock_draw.logp = np.zeros(draws.shape[0])
        mock_state.draw.return_value = mock_draw

        pr = PosteriorResults(
            draws=draws,
            param_names=param_names,
            logp=np.zeros(draws.shape[0]),
            sampler_state=mock_state,
        )
        return pr

    def test_save_posterior_no_state_raises(self, sample_draws):
        """PosteriorResults without sampler_state raises ValueError."""
        from easyreflectometry.analysis.bayesian import PosteriorResults
        from easyreflectometry.analysis.bayesian import save_posterior

        draws, param_names = sample_draws
        pr = PosteriorResults(draws, param_names)
        with pytest.raises(ValueError, match='no sampler_state'):
            save_posterior(pr, 'dummy')

    def test_save_posterior_wrong_state_type_raises(self, sample_draws):
        """Non-MCMCDraw sampler_state raises TypeError."""
        from unittest.mock import MagicMock

        from easyreflectometry.analysis.bayesian import PosteriorResults
        from easyreflectometry.analysis.bayesian import save_posterior

        draws, param_names = sample_draws
        pr = PosteriorResults(draws, param_names, sampler_state=MagicMock())
        with pytest.raises(TypeError, match='MCMCDraw'):
            save_posterior(pr, 'dummy')

    def test_save_and_load_roundtrip(self, mock_posterior_results, monkeypatch, tmp_path):
        """Save then load, verify draws, param_names, logp, and state."""
        from unittest.mock import MagicMock

        # Mock save_state and load_state
        import bumps.dream.state as _bds

        from easyreflectometry.analysis.bayesian import load_posterior
        from easyreflectometry.analysis.bayesian import save_posterior

        saved_state_ref = mock_posterior_results.sampler_state
        monkeypatch.setattr(_bds, 'save_state', MagicMock())
        monkeypatch.setattr(_bds, 'load_state', MagicMock(return_value=saved_state_ref))

        prefix = str(tmp_path / 'test_run')
        save_posterior(mock_posterior_results, prefix)

        # Verify save_state was called
        _bds.save_state.assert_called_once_with(saved_state_ref, prefix)

        loaded = load_posterior(prefix)

        assert np.allclose(loaded.draws, mock_posterior_results.draws)
        assert loaded.param_names == mock_posterior_results.param_names
        assert loaded.sampler_state is saved_state_ref

    def test_save_convenience_method(self, mock_posterior_results, monkeypatch, tmp_path):
        """PosteriorResults.save() delegates to save_posterior."""
        from unittest.mock import MagicMock

        import bumps.dream.state as _bds

        monkeypatch.setattr(_bds, 'save_state', MagicMock())

        prefix = str(tmp_path / 'test_convenience')
        mock_posterior_results.save(prefix)

        _bds.save_state.assert_called_once_with(mock_posterior_results.sampler_state, prefix)

    def test_load_posterior_skip(self, mock_posterior_results, monkeypatch, tmp_path):
        """load_posterior with skip>0 forwards skip to load_state."""
        from unittest.mock import MagicMock

        import bumps.dream.state as _bds

        from easyreflectometry.analysis.bayesian import load_posterior

        monkeypatch.setattr(_bds, 'load_state', MagicMock(return_value=mock_posterior_results.sampler_state))
        monkeypatch.setattr(_bds, 'save_state', MagicMock())

        prefix = str(tmp_path / 'test_skip')
        load_posterior(prefix, skip=5)

        _bds.load_state.assert_called_once_with(prefix, skip=5)


class TestPlotDistributionExported:
    def test_in_analysis_namespace(self):
        """``plot_distribution`` should be importable from the analysis package."""
        from easyreflectometry import analysis

        assert hasattr(analysis, 'plot_distribution')
        assert 'plot_distribution' in analysis.__all__


# ===================================================================
# Label wrapping helper
# ===================================================================


class TestWrapPairLabel:
    def test_empty_string_unchanged(self):
        from easyreflectometry.analysis.bayesian import _wrap_pair_label

        assert _wrap_pair_label('') == ''

    def test_short_name_unchanged(self):
        from easyreflectometry.analysis.bayesian import _wrap_pair_label

        assert _wrap_pair_label('thickness') == 'thickness'

    def test_dotted_name_breaks_on_dots(self):
        from easyreflectometry.analysis.bayesian import _wrap_pair_label

        assert _wrap_pair_label('layer1.thickness') == 'layer1.<br>thickness'

    def test_long_multiword_name_wraps(self):
        from easyreflectometry.analysis.bayesian import _wrap_pair_label

        wrapped = _wrap_pair_label('a very long parameter name indeed', max_len=16)
        assert '<br>' in wrapped
        assert wrapped.replace('<br>', ' ') == 'a very long parameter name indeed'

    def test_long_single_word_unchanged(self):
        from easyreflectometry.analysis.bayesian import _wrap_pair_label

        name = 'averyveryverylongsingleword'
        assert _wrap_pair_label(name, max_len=16) == name


# ===================================================================
# Optional-dependency guards
# ===================================================================


class TestRequireHelpers:
    def test_require_arviz_raises_when_unavailable(self, monkeypatch):
        from easyreflectometry.analysis import bayesian as bayesian_mod

        monkeypatch.setattr(bayesian_mod, '_HAS_ARVIZ', False)
        with pytest.raises(ImportError, match='arviz'):
            bayesian_mod._require_arviz()

    def test_require_plotly_raises_when_unavailable(self, monkeypatch):
        import builtins

        from easyreflectometry.analysis.bayesian import _require_plotly

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name.startswith('plotly'):
                raise ImportError('plotly disabled for test')
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', _fake_import)
        with pytest.raises(ImportError, match='plotly'):
            _require_plotly()

    def test_gelman_rubin_warns_and_returns_none_without_arviz(self, sample_draws, monkeypatch):
        from easyreflectometry.analysis import bayesian as bayesian_mod
        from easyreflectometry.analysis.bayesian import PosteriorResults

        draws, param_names = sample_draws
        pr = PosteriorResults(draws, param_names)
        monkeypatch.setattr(bayesian_mod, '_HAS_ARVIZ', False)
        with pytest.warns(UserWarning, match='arviz'):
            result = pr.gelman_rubin()
        assert result is None


# ===================================================================
# arviz data conversion
# ===================================================================


class TestToArvizData:
    def test_2d_draws_become_single_chain(self, sample_draws):
        pytest.importorskip('arviz')
        from easyreflectometry.analysis.bayesian import _to_arviz_data

        draws, param_names = sample_draws
        idata = _to_arviz_data(draws, param_names)
        posterior = idata.posterior
        # Dimension naming varies across arviz versions, so assert on the
        # contract itself: every sample survives conversion with its values
        # intact, in a single chain.
        assert posterior.sizes['chain'] == 1
        for i, name in enumerate(param_names):
            values = np.asarray(posterior[name].values).reshape(-1)
            assert values.size == draws.shape[0]
            assert np.allclose(values, draws[:, i])


# ===================================================================
# Plot construction (plotly available)
# ===================================================================


class TestPlotTraceFigure:
    def test_returns_figure_for_2d_draws(self, sample_draws):
        Figure = pytest.importorskip('plotly.graph_objects').Figure
        from easyreflectometry.analysis.bayesian import plot_trace

        draws, param_names = sample_draws
        fig = plot_trace(draws, param_names, return_figure=True)
        assert isinstance(fig, Figure)
        # One line trace and one histogram per parameter for the single chain.
        assert len(fig.data) == 2 * len(param_names)

    def test_returns_figure_for_multi_chain_draws(self, sample_draws):
        Figure = pytest.importorskip('plotly.graph_objects').Figure
        from easyreflectometry.analysis.bayesian import plot_trace

        draws, param_names = sample_draws
        multi = np.stack([draws, draws + 1.0], axis=0)  # (2, n_draws, n_params)
        fig = plot_trace(multi, param_names, return_figure=True)
        assert isinstance(fig, Figure)
        assert len(fig.data) == 2 * 2 * len(param_names)

    def test_inline_path_delegates_to_arviz(self, sample_draws, monkeypatch):
        pytest.importorskip('arviz')
        from unittest.mock import MagicMock

        from easyreflectometry.analysis import bayesian as bayesian_mod

        draws, param_names = sample_draws
        mock_plot = MagicMock()
        monkeypatch.setattr(bayesian_mod._arviz, 'plot_trace', mock_plot)
        result = bayesian_mod.plot_trace(draws, param_names)
        assert result is None
        mock_plot.assert_called_once()


class TestPlotDistributionFigure:
    def test_returns_none_without_return_figure(self, sample_draws):
        from easyreflectometry.analysis.bayesian import plot_distribution

        draws, param_names = sample_draws
        assert plot_distribution(draws, param_names) is None

    def test_returns_figure_with_expected_overlays(self, sample_draws):
        Figure = pytest.importorskip('plotly.graph_objects').Figure
        from easyreflectometry.analysis.bayesian import plot_distribution

        draws, param_names = sample_draws
        logp = np.arange(draws.shape[0], dtype=float)
        fig = plot_distribution(draws, param_names, logp=logp, return_figure=True)
        assert isinstance(fig, Figure)
        trace_names = {trace.name for trace in fig.data}
        assert 'Posterior histogram' in trace_names
        assert '95% credible interval' in trace_names
        assert 'Median' in trace_names
        # logp was supplied, so the best posterior sample line must be drawn.
        assert 'Best posterior sample' in trace_names

    def test_accepts_3d_draws(self, sample_draws):
        Figure = pytest.importorskip('plotly.graph_objects').Figure
        from easyreflectometry.analysis.bayesian import plot_distribution

        draws, param_names = sample_draws
        multi = np.stack([draws, draws], axis=0)  # (2, n_draws, n_params)
        fig = plot_distribution(multi, param_names, return_figure=True)
        assert isinstance(fig, Figure)


class TestPosteriorResultsPlotDelegates:
    def test_corner_returns_figure(self, sample_draws):
        Figure = pytest.importorskip('plotly.graph_objects').Figure
        from easyreflectometry.analysis.bayesian import PosteriorResults

        draws, param_names = sample_draws
        fig = PosteriorResults(draws, param_names).corner()
        assert isinstance(fig, Figure)

    def test_distribution_returns_figure(self, sample_draws):
        Figure = pytest.importorskip('plotly.graph_objects').Figure
        from easyreflectometry.analysis.bayesian import PosteriorResults

        draws, param_names = sample_draws
        fig = PosteriorResults(draws, param_names).distribution()
        assert isinstance(fig, Figure)

    def test_trace_delegates_to_plot_trace(self, sample_draws, monkeypatch):
        from unittest.mock import MagicMock

        from easyreflectometry.analysis import bayesian as bayesian_mod
        from easyreflectometry.analysis.bayesian import PosteriorResults

        draws, param_names = sample_draws
        mock_plot = MagicMock()
        monkeypatch.setattr(bayesian_mod, 'plot_trace', mock_plot)
        PosteriorResults(draws, param_names).trace()
        mock_plot.assert_called_once()


class TestPlotCornerEdgeCases:
    def test_accepts_3d_draws(self, sample_draws):
        Figure = pytest.importorskip('plotly.graph_objects').Figure
        from easyreflectometry.analysis.bayesian import plot_corner

        draws, param_names = sample_draws
        multi = np.stack([draws, draws], axis=0)
        fig = plot_corner(multi, param_names)
        assert isinstance(fig, Figure)

    def test_thins_scatter_for_large_posteriors(self):
        go = pytest.importorskip('plotly.graph_objects')
        from easyreflectometry.analysis.bayesian import _POSTERIOR_PAIR_SCATTER_MAX_POINTS
        from easyreflectometry.analysis.bayesian import plot_corner

        rng = np.random.default_rng(3)
        n_samples = _POSTERIOR_PAIR_SCATTER_MAX_POINTS * 2
        draws = rng.normal(size=(n_samples, 2))
        fig = plot_corner(draws, ['a', 'b'])
        scatters = [t for t in fig.data if isinstance(t, go.Scatter) and t.name == 'Posterior samples']
        assert scatters
        assert all(len(t.x) <= _POSTERIOR_PAIR_SCATTER_MAX_POINTS for t in scatters)

    def test_single_sample_falls_back_to_histogram(self):
        go = pytest.importorskip('plotly.graph_objects')
        from easyreflectometry.analysis.bayesian import plot_corner

        # A single draw defeats the KDE, so the diagonal must fall back to a
        # histogram and the pair panels must omit contours.
        draws = np.array([[250.0, 2.0]])
        fig = plot_corner(draws, ['thickness', 'sld'])
        assert any(isinstance(t, go.Histogram) for t in fig.data)
        assert not any(isinstance(t, go.Contour) for t in fig.data)


# ===================================================================
# Density-estimation helpers
# ===================================================================


class TestPosteriorAxisBounds:
    def test_empty_returns_none(self):
        from easyreflectometry.analysis.bayesian import _posterior_axis_bounds

        assert _posterior_axis_bounds(np.array([])) is None
        assert _posterior_axis_bounds(np.array([np.nan, np.inf])) is None

    def test_constant_values_get_padding(self):
        from easyreflectometry.analysis.bayesian import _posterior_axis_bounds

        lo, hi = _posterior_axis_bounds(np.array([5.0, 5.0, 5.0]))
        assert lo < 5.0 < hi

    def test_constant_zero_gets_padding(self):
        from easyreflectometry.analysis.bayesian import _posterior_axis_bounds

        lo, hi = _posterior_axis_bounds(np.zeros(3))
        assert lo < 0.0 < hi


class TestPosteriorDensityCurve:
    def test_too_few_samples_returns_none(self):
        from easyreflectometry.analysis.bayesian import _posterior_density_curve

        assert _posterior_density_curve(np.array([1.0])) is None

    def test_constant_samples_yield_gaussian_bump(self):
        pytest.importorskip('scipy')
        from easyreflectometry.analysis.bayesian import _posterior_density_curve

        result = _posterior_density_curve(np.full(50, 3.0))
        assert result is not None
        grid, density = result
        # Density peaks at the constant value and integrates to ~1.
        assert grid[np.argmax(density)] == pytest.approx(3.0, abs=(grid[1] - grid[0]))
        assert np.trapezoid(density, grid) == pytest.approx(1.0, rel=1e-6)

    def test_returns_none_without_scipy(self, sample_draws, monkeypatch):
        import builtins

        from easyreflectometry.analysis.bayesian import _posterior_density_curve

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name.startswith('scipy'):
                raise ImportError('scipy disabled for test')
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', _fake_import)
        draws, _ = sample_draws
        assert _posterior_density_curve(draws[:, 0]) is None


class TestPosteriorDensitySurface:
    def test_degenerate_samples_return_none(self):
        pytest.importorskip('scipy')
        from easyreflectometry.analysis.bayesian import _posterior_density_surface

        constant = np.full(50, 1.0)
        # Both axes constant.
        assert _posterior_density_surface(constant, constant) is None
        # One axis constant: rank-deficient covariance.
        rng = np.random.default_rng(11)
        assert _posterior_density_surface(constant, rng.normal(size=50)) is None

    def test_too_few_finite_samples_return_none(self):
        pytest.importorskip('scipy')
        from easyreflectometry.analysis.bayesian import _posterior_density_surface

        x = np.array([1.0, np.nan, np.nan])
        y = np.array([2.0, np.nan, np.nan])
        assert _posterior_density_surface(x, y) is None

    def test_valid_samples_return_grids(self, sample_draws):
        pytest.importorskip('scipy')
        from easyreflectometry.analysis.bayesian import _posterior_density_surface

        draws, _ = sample_draws
        result = _posterior_density_surface(draws[:, 0], draws[:, 1])
        assert result is not None
        x_grid, y_grid, density = result
        assert density.shape == (len(y_grid), len(x_grid))

    def test_returns_none_without_scipy(self, sample_draws, monkeypatch):
        import builtins

        from easyreflectometry.analysis.bayesian import _posterior_density_surface

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name.startswith('scipy'):
                raise ImportError('scipy disabled for test')
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', _fake_import)
        draws, _ = sample_draws
        assert _posterior_density_surface(draws[:, 0], draws[:, 1]) is None


class TestPosteriorContourColorscales:
    def test_negative_correlation_selects_red_palette(self):
        from easyreflectometry.analysis.bayesian import _POSTERIOR_NEGATIVE_CONTOUR_FILL_COLORSCALE
        from easyreflectometry.analysis.bayesian import _posterior_contour_colorscales

        x = np.linspace(0, 1, 50)
        fill, _ = _posterior_contour_colorscales(x, -x)
        assert fill is _POSTERIOR_NEGATIVE_CONTOUR_FILL_COLORSCALE

    def test_positive_correlation_selects_blue_palette(self):
        from easyreflectometry.analysis.bayesian import _POSTERIOR_CONTOUR_FILL_COLORSCALE
        from easyreflectometry.analysis.bayesian import _posterior_contour_colorscales

        x = np.linspace(0, 1, 50)
        fill, _ = _posterior_contour_colorscales(x, x)
        assert fill is _POSTERIOR_CONTOUR_FILL_COLORSCALE


class TestPosteriorMarginalYRange:
    def test_covers_histogram_and_kde_peaks(self, sample_draws):
        from easyreflectometry.analysis.bayesian import _posterior_density_curve
        from easyreflectometry.analysis.bayesian import _posterior_marginal_y_range

        draws, _ = sample_draws
        values = draws[:, 0]
        curve = _posterior_density_curve(values)
        y_range = _posterior_marginal_y_range(values, curve)
        assert y_range is not None
        lo, hi = y_range
        assert lo == 0.0
        hist, _ = np.histogram(values, bins=40, density=True)
        assert hi >= np.max(hist)

    def test_no_data_returns_none(self):
        from easyreflectometry.analysis.bayesian import _posterior_marginal_y_range

        assert _posterior_marginal_y_range(np.array([]), None) is None


# ===================================================================
# Metadata helpers
# ===================================================================


class TestMetadataHelpers:
    def test_version_returns_string(self):
        from easyreflectometry.analysis.bayesian import _easyreflectometry_version

        assert isinstance(_easyreflectometry_version(), str)

    def test_data_fingerprint_is_deterministic(self):
        from easyreflectometry.analysis.bayesian import _data_fingerprint

        x = [np.array([1.0, 2.0])]
        y = [np.array([3.0, 4.0])]
        w = [np.array([0.1, 0.2])]
        first = _data_fingerprint(x, y, w)
        assert isinstance(first, str)
        assert len(first) == 64
        assert _data_fingerprint(x, y, w) == first
        assert _data_fingerprint(x, y, [np.array([0.1, 0.3])]) != first

    def test_data_fingerprint_returns_none_on_bad_input(self):
        from easyreflectometry.analysis.bayesian import _data_fingerprint

        assert _data_fingerprint([object()], [], []) is None
