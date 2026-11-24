# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

"""Medium-range structure of hyve-forecast-crps: score one start date, then
aggregate the per-start-date scores over a window.
"""

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hyve.tools import forecast_crps
from hyve.tools.forecast_crps import cli, medium_range
from hyve.tools.forecast_crps.config import ForecastCrpsConfig
from hyve.tools.forecast_crps.errors import (
    DimensionMismatchError,
    GridMismatchError,
    InputContractError,
    MissingDataError,
)

POINTS = np.array([10, 11, 12])
MEMBERS = np.array([1, 2, 3])  # labels, deliberately not 0-based
STEP_HOURS = 6
N_STEPS = 4  # plus step 0, which is dropped


def _config(**kwargs):
    kwargs.setdefault("structure", "medium_range")
    kwargs.setdefault("spatial_dims", ["point"])
    return ForecastCrpsConfig(**kwargs)


def _crps(members, obs):
    """Empirical CRPS of one ensemble: E|X - y| - E|X - X'| / 2."""
    x = np.asarray(members, dtype=np.float64)
    return np.mean(np.abs(x - obs)) - 0.5 * np.mean(np.abs(x[:, None] - x[None, :]))


def _reanalysis(start="2025-09-20", end="2025-11-15"):
    time = pd.date_range(start, end, freq=f"{STEP_HOURS}h")
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {"dis": (("time", "point"), rng.uniform(1, 10, (time.size, POINTS.size)))},
        coords={"time": time, "point": POINTS},
    )


def _forecast(start, seed=1, n_steps=N_STEPS):
    start = pd.Timestamp(start)
    time = start + pd.to_timedelta(np.arange(n_steps + 1) * STEP_HOURS, unit="h")
    rng = np.random.default_rng(seed)
    values = rng.uniform(1, 10, (MEMBERS.size, time.size, POINTS.size))
    return xr.Dataset(
        {"dis": (("ensemble", "time", "point"), values)},
        coords={
            "ensemble": MEMBERS,
            "time": time,
            "point": POINTS,
            "forecast_reference_time": start,
        },
    )


def _climatology(issue_hours=(0,)):
    """Members encode doy and issue hour, so a wrong selection shows."""
    doy = np.arange(1, 367)
    issue_hours = np.asarray(issue_hours)
    values = (
        doy[None, :, None, None] * 0.01
        + issue_hours[None, None, :, None] * 0.1
        + MEMBERS[:, None, None, None]
        + POINTS[None, None, None, :] * 0.001
    )
    return xr.DataArray(
        values,
        dims=("ensemble", "doy", "issue_hour", "point"),
        coords={
            "ensemble": MEMBERS,
            "doy": doy,
            "issue_hour": issue_hours,
            "point": POINTS,
        },
        name="dis",
    )


# ---------------------------------------------------------------- score


def test_score_matches_hand_computed_crps_and_persistence():
    fc = _forecast("2025-10-01")
    rea = _reanalysis()
    scores = medium_range.score_start_date(fc, rea, _config())

    start = pd.Timestamp("2025-10-01")
    for i, step in enumerate(scores.step.values):
        valid = start + pd.Timedelta(step)
        for p in POINTS:
            obs = float(rea.dis.sel(time=valid, point=p))
            members = fc.dis.sel(time=valid, point=p).values
            assert scores.crps_forecast.sel(point=p).values[i] == pytest.approx(
                _crps(members, obs), rel=1e-6
            )
            persistence = float(rea.dis.sel(time=start, point=p))
            assert scores.crps_persistence.sel(point=p).values[i] == pytest.approx(
                abs(obs - persistence), rel=1e-6
            )
    assert "crps_climatology" not in scores


def test_score_has_lead_time_and_start_time_coordinates():
    scores = medium_range.score_start_date(
        _forecast("2025-10-01 06:00"), _reanalysis(), _config()
    )
    assert scores.crps_forecast.dims == ("step", "point")
    expected_steps = pd.to_timedelta(
        np.arange(1, N_STEPS + 1) * STEP_HOURS, unit="h"
    ).values
    np.testing.assert_array_equal(scores.step.values, expected_steps)
    np.testing.assert_array_equal(
        scores.valid_time.values,
        (pd.Timestamp("2025-10-01 06:00") + pd.to_timedelta(expected_steps)).values,
    )
    assert pd.Timestamp(scores.forecast_reference_time.values) == pd.Timestamp(
        "2025-10-01 06:00"
    )
    np.testing.assert_array_equal(scores.point.values, POINTS)


def test_score_file_is_float32_and_named_by_start_time_with_hour(tmp_path):
    scores = medium_range.score_start_date(
        _forecast("2025-10-01 06:00"), _reanalysis(), _config()
    )
    path = medium_range.write_scores(scores, tmp_path)
    assert path.name == "scores_2025100106.nc"
    assert not list(tmp_path.glob("*.tmp"))
    with xr.open_dataset(path) as ds:
        assert ds.crps_forecast.dtype == np.float32
        assert ds.crps_persistence.dtype == np.float32


def test_failed_write_leaves_no_file(tmp_path):
    class Failing:
        def to_netcdf(self, path):
            path.write_bytes(b"partial")
            raise OSError("disk full")

    target = tmp_path / "scores_2025100100.nc"
    target.write_bytes(b"previous")
    with pytest.raises(OSError, match="disk full"):
        medium_range._write_atomic(Failing(), target)
    assert sorted(p.name for p in tmp_path.iterdir()) == [target.name]
    assert target.read_bytes() == b"previous"


def test_score_without_forecast_reference_time_fails():
    fc = _forecast("2025-10-01").drop_vars("forecast_reference_time")
    with pytest.raises(InputContractError, match="forecast_reference_time"):
        medium_range.score_start_date(fc, _reanalysis(), _config())


def test_score_reads_forecast_reference_time_only_as_a_scalar_coordinate():
    fc = _forecast("2025-10-01")
    fc = fc.drop_vars("forecast_reference_time")
    fc.attrs["forecast_reference_time"] = "2025-10-01T00:00:00"
    with pytest.raises(InputContractError, match="scalar coordinate"):
        medium_range.score_start_date(fc, _reanalysis(), _config())


def test_score_rejects_valid_times_before_the_start():
    fc = _forecast("2025-10-01").assign_coords(
        forecast_reference_time=pd.Timestamp("2025-10-01 12:00")
    )
    with pytest.raises(InputContractError, match="before"):
        medium_range.score_start_date(fc, _reanalysis(), _config())


@pytest.mark.parametrize("start", ["2025-10-01", "2025-10-01 18:00"])
def test_daily_climatology_uses_valid_date_doy_and_issue_hour_zero(start):
    fc = _forecast(start)
    rea = _reanalysis()
    clim = _climatology(issue_hours=(0,))
    scores = medium_range.score_start_date(fc, rea, _config(), climatology=clim)

    for i, step in enumerate(scores.step.values):
        valid = pd.Timestamp(start) + pd.Timedelta(step)
        for p in POINTS:
            obs = float(rea.dis.sel(time=valid, point=p))
            members = clim.sel(doy=valid.dayofyear, issue_hour=0, point=p).values
            assert scores.crps_climatology.sel(point=p).values[i] == pytest.approx(
                _crps(members, obs), rel=1e-6
            )


def test_sub_daily_climatology_uses_the_issue_hour_of_the_valid_time():
    fc = _forecast("2025-10-01")
    rea = _reanalysis()
    clim = _climatology(issue_hours=(0, 6, 12, 18))
    scores = medium_range.score_start_date(fc, rea, _config(), climatology=clim)

    for i, step in enumerate(scores.step.values):
        valid = pd.Timestamp("2025-10-01") + pd.Timedelta(step)
        obs = float(rea.dis.sel(time=valid, point=10))
        members = clim.sel(doy=valid.dayofyear, issue_hour=valid.hour, point=10).values
        assert scores.crps_climatology.sel(point=10).values[i] == pytest.approx(
            _crps(members, obs), rel=1e-6
        )


def test_climatology_without_an_issue_hour_slot_for_a_valid_time_fails():
    clim = _climatology(issue_hours=(6,))
    with pytest.raises(InputContractError, match="issue_hour"):
        medium_range.score_start_date(
            _forecast("2025-10-01"), _reanalysis(), _config(), climatology=clim
        )


def test_reanalysis_gap_gives_nan_in_that_cell_only():
    rea = _reanalysis()
    gap = pd.Timestamp("2025-10-01 12:00")
    rea["dis"].loc[{"time": gap, "point": 11}] = np.nan
    scores = medium_range.score_start_date(
        _forecast("2025-10-01"), rea, _config(), climatology=_climatology()
    )
    step = gap - pd.Timestamp("2025-10-01")
    for name in ("crps_forecast", "crps_persistence", "crps_climatology"):
        cell = scores[name].sel(step=step, point=11)
        assert np.isnan(cell)
        others = scores[name].drop_sel(point=11)
        assert np.isfinite(others).all()
        assert np.isfinite(scores[name].sel(point=11).drop_sel(step=step)).all()


def test_reanalysis_missing_for_a_whole_step_fails_and_names_it():
    rea = _reanalysis()
    rea = rea.drop_sel(time=pd.Timestamp("2025-10-01 18:00"))
    with pytest.raises(MissingDataError, match="2025-10-01T18:00"):
        medium_range.score_start_date(_forecast("2025-10-01"), rea, _config())


def test_lead_times_between_reanalysis_times_fail_with_a_hint():
    fc = _forecast("2025-10-01", n_steps=2 * N_STEPS)
    fc = fc.assign_coords(time=fc.time - (fc.time - fc.time[0]) / 2)  # 3-hourly
    with pytest.raises(MissingDataError, match=r"03:00.*must be a reanalysis time"):
        medium_range.score_start_date(fc, _reanalysis(), _config())


def test_reanalysis_missing_at_the_start_time_fails():
    rea = _reanalysis()
    rea = rea.drop_sel(time=pd.Timestamp("2025-10-01"))
    with pytest.raises(MissingDataError, match="persistence"):
        medium_range.score_start_date(_forecast("2025-10-01"), rea, _config())


def test_point_without_forecast_data_is_nan_in_every_score():
    fc = _forecast("2025-10-01")
    fc["dis"].loc[{"point": 12}] = np.nan
    scores = medium_range.score_start_date(
        fc, _reanalysis(), _config(), climatology=_climatology()
    )
    for name in ("crps_forecast", "crps_persistence", "crps_climatology"):
        assert scores[name].sel(point=12).isnull().all()
        assert np.isfinite(scores[name].sel(point=[10, 11])).all()


def test_forecast_without_any_data_fails():
    fc = _forecast("2025-10-01")
    fc["dis"][:] = np.nan
    with pytest.raises(MissingDataError, match="no data at any point"):
        medium_range.score_start_date(fc, _reanalysis(), _config())


def test_forecast_reference_time_must_be_scalar():
    fc = _forecast("2025-10-01")
    fc = fc.assign_coords(forecast_reference_time=("time", fc.time.values))
    with pytest.raises(InputContractError, match="must be a scalar"):
        medium_range.score_start_date(fc, _reanalysis(), _config())


def test_forecast_without_a_lead_time_after_the_start_fails():
    fc = _forecast("2025-10-01").isel(time=[0])
    with pytest.raises(InputContractError, match="no lead time after its start"):
        medium_range.score_start_date(fc, _reanalysis(), _config())


def test_climatology_must_cover_every_forecast_point():
    clim = _climatology().isel(point=[0, 1])
    with pytest.raises(GridMismatchError, match="lacks 1 of the 3 'point'"):
        medium_range.score_start_date(
            _forecast("2025-10-01"), _reanalysis(), _config(), climatology=clim
        )


def test_float_coordinates_match_within_the_tolerance():
    def floats(ds, offset):
        return ds.assign_coords(point=ds.point.astype(float) + offset)

    expected = medium_range.score_start_date(
        floats(_forecast("2025-10-01"), 0), floats(_reanalysis(), 0), _config()
    )
    scores = medium_range.score_start_date(
        floats(_forecast("2025-10-01"), 0),
        floats(_reanalysis(), 1e-9),
        _config(coord_tolerance=1e-6),
    )
    xr.testing.assert_identical(scores, expected)


def test_score_matches_reanalysis_points_by_label():
    rea = _reanalysis().isel(point=[2, 0, 1])  # other order, same points
    a = medium_range.score_start_date(_forecast("2025-10-01"), _reanalysis(), _config())
    b = medium_range.score_start_date(_forecast("2025-10-01"), rea, _config())
    xr.testing.assert_allclose(a, b)


def test_score_does_not_depend_on_chunk_size():
    fc = _forecast("2025-10-01")
    rea = _reanalysis()
    clim = _climatology()
    whole = medium_range.score_start_date(fc, rea, _config(), climatology=clim)
    chunked = medium_range.score_start_date(
        fc, rea, _config(chunk_size=1), climatology=clim
    )
    xr.testing.assert_allclose(whole, chunked)


def _step_layout(fc, valid_time_coord=True):
    """`fc` with lead times on ``step`` instead of valid times on ``time``."""
    frt = fc["forecast_reference_time"].values
    fc = fc.assign_coords(step=("time", (fc["time"].values - frt))).swap_dims(
        time="step"
    )
    if valid_time_coord:  # as cfgrib and earthkit write it
        return fc.rename(time="valid_time")
    return fc.drop_vars("time")


@pytest.mark.parametrize("valid_time_coord", [True, False])
def test_step_layout_gives_the_same_scores(valid_time_coord):
    fc = _forecast("2025-10-01 12:00")
    rea = _reanalysis()
    clim = _climatology()
    expected = medium_range.score_start_date(fc, rea, _config(), climatology=clim)
    scores = medium_range.score_start_date(
        _step_layout(fc, valid_time_coord),
        rea,
        _config(lead_time_dim="step"),
        climatology=clim,
    )
    xr.testing.assert_identical(scores, expected)


def test_step_layout_ignores_a_wrong_valid_time_coordinate():
    fc = _forecast("2025-10-01")
    expected = medium_range.score_start_date(fc, _reanalysis(), _config())
    stepped = _step_layout(fc)
    stepped = stepped.assign_coords(
        valid_time=stepped["valid_time"] + np.timedelta64(1, "D")
    )
    scores = medium_range.score_start_date(
        stepped, _reanalysis(), _config(lead_time_dim="step")
    )
    xr.testing.assert_identical(scores, expected)


def test_step_layout_without_step_fails():
    with pytest.raises(InputContractError, match="lead_time_dim"):
        medium_range.score_start_date(
            _forecast("2025-10-01"), _reanalysis(), _config(lead_time_dim="step")
        )


def test_step_layout_needs_forecast_reference_time():
    fc = _step_layout(_forecast("2025-10-01")).drop_vars("forecast_reference_time")
    with pytest.raises(InputContractError, match="forecast_reference_time"):
        medium_range.score_start_date(fc, _reanalysis(), _config(lead_time_dim="step"))


def test_step_must_hold_time_differences():
    fc = _step_layout(_forecast("2025-10-01"))
    fc = fc.assign_coords(step=np.arange(fc.sizes["step"]) * STEP_HOURS)
    with pytest.raises(InputContractError, match="time differences"):
        medium_range.score_start_date(fc, _reanalysis(), _config(lead_time_dim="step"))


def test_step_layout_rejects_a_time_dimension_as_well():
    fc = _forecast("2025-10-01").expand_dims(step=[np.timedelta64(0, "h")])
    with pytest.raises(InputContractError, match="both"):
        medium_range.score_start_date(fc, _reanalysis(), _config(lead_time_dim="step"))


def test_configured_ensemble_dimension():
    fc = _forecast("2025-10-01")
    clim = _climatology()
    expected = medium_range.score_start_date(
        fc, _reanalysis(), _config(), climatology=clim
    )
    scores = medium_range.score_start_date(
        fc.rename(ensemble="number"),
        _reanalysis(),
        _config(ensemble_dim="number"),
        climatology=clim.rename(ensemble="number"),
    )
    xr.testing.assert_identical(scores, expected)


def test_unconfigured_ensemble_dimension_fails():
    with pytest.raises(DimensionMismatchError, match="number"):
        medium_range.score_start_date(
            _forecast("2025-10-01").rename(ensemble="number"), _reanalysis(), _config()
        )


def test_unsorted_lead_times_give_sorted_steps():
    fc = _forecast("2025-10-01")
    expected = medium_range.score_start_date(fc, _reanalysis(), _config())
    scores = medium_range.score_start_date(
        fc.isel(time=[0, 3, 1, 4, 2]), _reanalysis(), _config()
    )
    xr.testing.assert_identical(scores, expected)


def test_score_uses_the_configured_variable():
    clim = _climatology()
    expected = medium_range.score_start_date(
        _forecast("2025-10-01"), _reanalysis(), _config(), climatology=clim
    )
    fc = _forecast("2025-10-01").rename(dis="q")
    rea = _reanalysis().rename(dis="q")
    # as read from a file: a dataset, here with another variable as a decoy
    clim_ds = xr.Dataset({"q": clim, "dis": clim + 100})
    scores = medium_range.score_start_date(
        fc, rea, _config(variable="q"), climatology=clim_ds
    )
    expected.attrs["variable"] = "q"
    xr.testing.assert_identical(scores, expected)


# ------------------------------------------------------------ aggregate


def _scores(start, steps_h=(24, 48, 72), value=None, with_clim=True, points=POINTS):
    """A per-start-date score dataset with known values.

    By default every score is the day of month of the start date, so means
    show which start dates were used. Persistence is twice the forecast.
    """
    start = pd.Timestamp(start)
    steps = pd.to_timedelta(np.asarray(steps_h), unit="h")
    v = float(start.day if value is None else value)
    shape = (steps.size, len(points))
    data = {
        "crps_forecast": (("step", "point"), np.full(shape, v, np.float32)),
        "crps_persistence": (("step", "point"), np.full(shape, 2 * v, np.float32)),
    }
    if with_clim:
        data["crps_climatology"] = (
            ("step", "point"),
            np.full(shape, 4 * v, np.float32),
        )
    return xr.Dataset(
        data,
        coords={
            "step": steps.values,
            "point": points,
            "forecast_reference_time": start,
            "valid_time": ("step", (start + steps).values),
        },
    )


def _write_all(folder, datasets):
    folder.mkdir(exist_ok=True)
    return [medium_range.write_scores(ds, folder) for ds in datasets]


def test_aggregate_by_valid_time_uses_start_dates_per_step(tmp_path):
    starts = pd.date_range("2025-09-27", "2025-10-03", freq="D")
    paths = _write_all(tmp_path / "scores", [_scores(s) for s in starts])
    out = medium_range.aggregate(paths, "2025-10-01", "2025-10-02", _config())

    # step 1 day: starts 30 Sep and 1 Oct; 2 days: 29 and 30 Sep; 3 days: 28, 29 Sep
    expected = {24: (30 + 1) / 2, 48: (29 + 30) / 2, 72: (28 + 29) / 2}
    for hours, mean in expected.items():
        step = pd.Timedelta(hours, "h")
        np.testing.assert_allclose(out.crps_forecast.sel(step=step), mean)
        np.testing.assert_allclose(out.crps_persistence.sel(step=step), 2 * mean)
        np.testing.assert_array_equal(out["count"].sel(step=step), 2)
    assert out.crps_forecast.dims == ("step", "point")
    assert out.attrs["window_start"] == "2025-10-01"
    assert out.attrs["window_end"] == "2025-10-02"
    assert out.attrs["aggregation_by"] == "valid_time"


def test_aggregate_by_start_time(tmp_path):
    starts = pd.date_range("2025-09-27", "2025-10-03", freq="D")
    paths = _write_all(tmp_path / "scores", [_scores(s) for s in starts])
    config = _config(aggregation={"by": "start_time"})
    out = medium_range.aggregate(paths, "2025-10-01", "2025-10-02", config)
    np.testing.assert_allclose(out.crps_forecast, (1 + 2) / 2)
    np.testing.assert_array_equal(out["count"], 2)
    assert out.attrs["aggregation_by"] == "start_time"


def test_window_edges_follow_the_calendar_date_of_the_valid_time(tmp_path):
    # valid 1 Oct 00 UTC is in October, valid 1 Nov 00 UTC is not
    paths = _write_all(
        tmp_path / "scores",
        [
            _scores("2025-09-30 18:00", steps_h=(6,), value=1),
            _scores("2025-10-31 12:00", steps_h=(6,), value=3),
            _scores("2025-10-31 18:00", steps_h=(6,), value=100),
        ],
    )
    config = _config(aggregation={"start_frequency": "6h", "allow_missing": True})
    out = medium_range.aggregate(paths, "2025-10-01", "2025-10-31", config)
    np.testing.assert_allclose(out.crps_forecast.sel(step=pd.Timedelta(6, "h")), 2)
    np.testing.assert_array_equal(out["count"], 2)


def test_same_samples_for_forecast_and_benchmarks(tmp_path):
    a = _scores("2025-10-01", value=1)
    b = _scores("2025-10-02", value=3)
    b["crps_climatology"].loc[{"step": pd.Timedelta(24, "h"), "point": 11}] = np.nan
    paths = _write_all(tmp_path / "scores", [a, b])
    config = _config(aggregation={"by": "start_time"})
    out = medium_range.aggregate(paths, "2025-10-01", "2025-10-02", config)

    cell = {"step": pd.Timedelta(24, "h"), "point": 11}
    assert int(out["count"].sel(cell)) == 1
    assert float(out.crps_forecast.sel(cell)) == pytest.approx(1)
    assert float(out.crps_persistence.sel(cell)) == pytest.approx(2)
    assert float(out.crps_climatology.sel(cell)) == pytest.approx(4)
    other = {"step": pd.Timedelta(24, "h"), "point": 10}
    assert int(out["count"].sel(other)) == 2
    assert float(out.crps_forecast.sel(other)) == pytest.approx(2)


def test_skill_scores_and_zero_benchmark(tmp_path):
    a = _scores("2025-10-01", value=1)
    a["crps_persistence"].loc[{"point": 12}] = 0
    paths = _write_all(tmp_path / "scores", [a])
    config = _config(aggregation={"by": "start_time"})
    out = medium_range.aggregate(paths, "2025-10-01", "2025-10-01", config)
    np.testing.assert_allclose(out.crpss_persistence.sel(point=[10, 11]), 1 - 1 / 2)
    np.testing.assert_allclose(out.crpss_climatology, 1 - 1 / 4)
    assert out.crpss_persistence.sel(point=12).isnull().all()
    assert not np.isinf(out.crpss_persistence).any()


def test_min_count_masks_cells_with_too_few_samples(tmp_path):
    a = _scores("2025-10-01", value=1)
    b = _scores("2025-10-02", value=3)
    b["crps_forecast"].loc[{"point": 10}] = np.nan
    paths = _write_all(tmp_path / "scores", [a, b])
    config = _config(aggregation={"by": "start_time", "min_count": 2})
    out = medium_range.aggregate(paths, "2025-10-01", "2025-10-02", config)
    for name in ("crps_forecast", "crps_persistence", "crpss_persistence"):
        assert out[name].sel(point=10).isnull().all()
        assert np.isfinite(out[name].sel(point=[11, 12])).all()
    np.testing.assert_array_equal(out["count"].sel(point=10), 1)


def test_aggregate_without_climatology(tmp_path):
    paths = _write_all(tmp_path / "scores", [_scores("2025-10-01", with_clim=False)])
    config = _config(aggregation={"by": "start_time"})
    out = medium_range.aggregate(paths, "2025-10-01", "2025-10-01", config)
    assert "crps_climatology" not in out
    assert "crpss_climatology" not in out
    np.testing.assert_allclose(out.crpss_persistence, 0.5)


def test_aggregate_with_different_step_sets(tmp_path):
    paths = _write_all(
        tmp_path / "scores",
        [
            _scores("2025-10-01", steps_h=(24,), value=1),
            _scores("2025-10-02", steps_h=(24, 48), value=3),
        ],
    )
    config = _config(aggregation={"by": "start_time"})
    out = medium_range.aggregate(paths, "2025-10-01", "2025-10-02", config)
    np.testing.assert_array_equal(out["count"].sel(step=pd.Timedelta(24, "h")), 2)
    np.testing.assert_array_equal(out["count"].sel(step=pd.Timedelta(48, "h")), 1)
    np.testing.assert_allclose(out.crps_forecast.sel(step=pd.Timedelta(48, "h")), 3)


def test_aggregate_ignores_files_outside_the_window(tmp_path):
    paths = _write_all(
        tmp_path / "scores",
        [_scores("2025-10-01", value=1), _scores("2025-12-01", points=[1, 2])],
    )
    config = _config(aggregation={"by": "start_time"})
    out = medium_range.aggregate(paths, "2025-10-01", "2025-10-01", config)
    np.testing.assert_allclose(out.crps_forecast, 1)


def test_aggregate_requires_the_same_grid_in_every_file(tmp_path):
    paths = _write_all(
        tmp_path / "scores",
        [_scores("2025-10-01"), _scores("2025-10-02", points=[10, 11, 13])],
    )
    config = _config(aggregation={"by": "start_time"})
    with pytest.raises(GridMismatchError, match="point"):
        medium_range.aggregate(paths, "2025-10-01", "2025-10-02", config)


def test_aggregate_rejects_two_files_with_the_same_start_time(tmp_path):
    paths = _write_all(tmp_path / "scores", [_scores("2025-10-01", value=1)])
    copy = tmp_path / "scores" / "scores_2025100100_rerun.nc"
    _scores("2025-10-01", value=5).to_netcdf(copy)
    config = _config(aggregation={"by": "start_time"})
    with pytest.raises(InputContractError, match="same start time 2025-10-01T00:00"):
        medium_range.aggregate([*paths, copy], "2025-10-01", "2025-10-01", config)


def test_aggregate_with_no_file_in_the_window_fails(tmp_path):
    paths = _write_all(tmp_path / "scores", [_scores("2025-10-01")])
    with pytest.raises(MissingDataError, match="none of the 1 score files.*2025-12-01"):
        medium_range.aggregate(paths, "2025-12-01", "2025-12-31", _config())


@pytest.mark.parametrize(
    "by, outside",
    [
        # valid on 1 Oct only, the day before the window
        ("valid_time", "2025-09-30"),
        # started at the end of the window, at 00 UTC on the next day
        ("start_time", "2025-10-02"),
    ],
)
def test_files_just_outside_the_window_are_not_read(tmp_path, by, outside):
    paths = _write_all(
        tmp_path / "scores",
        [
            _scores("2025-10-01", steps_h=(24,), value=1),
            # on another grid, so reading it would fail
            _scores(outside, steps_h=(24,), points=[1, 2]),
        ],
    )
    config = _config(aggregation={"by": by, "allow_missing": True})
    window = ("2025-10-01", "2025-10-01") if by == "start_time" else ("2025-10-02",) * 2
    out = medium_range.aggregate(paths, *window, config)
    np.testing.assert_allclose(out.crps_forecast, 1)
    assert out.attrs["score_files"] == 1


def test_expected_start_times_with_steps_off_the_start_grid(tmp_path):
    # a 30 h step: the run of 30 Sep is valid on 1 Oct, the run of 29 Sep is not
    paths = _write_all(tmp_path / "scores", [_scores("2025-09-30", steps_h=(30,))])
    out = medium_range.aggregate(paths, "2025-10-01", "2025-10-01", _config())
    assert out.attrs["expected_start_times"] == 1
    assert out.attrs["missing_start_times"] == ""


def test_aggregate_rejects_climatology_in_some_files_only(tmp_path):
    paths = _write_all(
        tmp_path / "scores",
        [_scores("2025-10-01"), _scores("2025-10-02", with_clim=False)],
    )
    config = _config(aggregation={"by": "start_time"})
    with pytest.raises(InputContractError, match="scores_2025100200.nc"):
        medium_range.aggregate(paths, "2025-10-01", "2025-10-02", config)


def test_aggregate_requires_forecast_and_persistence_scores(tmp_path):
    paths = _write_all(
        tmp_path / "scores", [_scores("2025-10-01").drop_vars("crps_persistence")]
    )
    config = _config(aggregation={"by": "start_time"})
    with pytest.raises(InputContractError, match="crps_persistence"):
        medium_range.aggregate(paths, "2025-10-01", "2025-10-01", config)


def test_aggregate_rejects_an_end_before_the_start(tmp_path):
    paths = _write_all(tmp_path / "scores", [_scores("2025-10-01")])
    with pytest.raises(ValueError, match="before"):
        medium_range.aggregate(paths, "2025-10-31", "2025-10-01", _config())


# ------------------------------------------------- window completeness


@pytest.mark.parametrize(
    "by, start, end, frequency, expected",
    [
        # start times in the window
        ("start_time", "2025-10-01", "2025-10-02", "1D", ["2025-10-01", "2025-10-02"]),
        (
            "start_time",
            "2025-10-01",
            "2025-10-01",
            "12h",
            ["2025-10-01 00:00", "2025-10-01 12:00"],
        ),
        # steps 24-72 h valid on 1-2 Oct: started 28 Sep to 1 Oct
        (
            "valid_time",
            "2025-10-01",
            "2025-10-02",
            "1D",
            ["2025-09-28", "2025-09-29", "2025-09-30", "2025-10-01"],
        ),
    ],
)
def test_expected_start_times(by, start, end, frequency, expected):
    datasets = [_scores("2025-10-01", steps_h=(24, 48, 72))]
    lo, hi = medium_range._window(start, end)
    got = medium_range.expected_start_times(
        datasets, by, lo, hi, pd.Timedelta(frequency)
    )
    assert list(got) == list(pd.to_datetime(expected))


def test_missing_start_time_fails_and_names_it(tmp_path):
    starts = ["2025-10-01", "2025-10-03"]
    paths = _write_all(tmp_path / "scores", [_scores(s) for s in starts])
    config = _config(aggregation={"by": "start_time"})
    with pytest.raises(MissingDataError, match=r"1 of the 3 .*2025-10-02T00:00"):
        medium_range.aggregate(paths, "2025-10-01", "2025-10-03", config)


def test_missing_start_time_before_the_window_by_valid_time(tmp_path):
    # the 72 h step of the forecast started 28 Sep is valid on 1 Oct
    starts = pd.date_range("2025-09-29", "2025-10-02", freq="D")
    paths = _write_all(tmp_path / "scores", [_scores(s) for s in starts])
    with pytest.raises(MissingDataError, match="2025-09-28T00:00"):
        medium_range.aggregate(paths, "2025-10-01", "2025-10-02", _config())


def test_allow_missing_aggregates_and_records_the_gaps(tmp_path, caplog):
    starts = ["2025-10-01", "2025-10-03"]
    paths = _write_all(tmp_path / "scores", [_scores(s) for s in starts])
    config = _config(aggregation={"by": "start_time", "allow_missing": True})
    with caplog.at_level("WARNING"):
        out = medium_range.aggregate(paths, "2025-10-01", "2025-10-03", config)
    assert "2025-10-02T00:00" in caplog.text
    np.testing.assert_allclose(out.crps_forecast, (1 + 3) / 2)
    np.testing.assert_array_equal(out["count"], 2)
    assert out.attrs["expected_start_times"] == 3
    assert out.attrs["missing_start_times"] == "2025-10-02T00:00"
    assert out.attrs["start_frequency"] == "1D"


def test_start_frequency_sets_the_expected_start_times(tmp_path):
    starts = ["2025-10-01 00:00", "2025-10-01 12:00"]
    paths = _write_all(tmp_path / "scores", [_scores(s) for s in starts])
    daily = _config(aggregation={"by": "start_time"})
    out = medium_range.aggregate(paths, "2025-10-01", "2025-10-01", daily)
    assert out.attrs["expected_start_times"] == 1  # the 12 UTC run is extra
    twice = _config(aggregation={"by": "start_time", "start_frequency": "12h"})
    out = medium_range.aggregate(paths, "2025-10-01", "2025-10-01", twice)
    assert out.attrs["expected_start_times"] == 2
    six_hourly = _config(aggregation={"by": "start_time", "start_frequency": "6h"})
    with pytest.raises(MissingDataError, match="06:00"):
        medium_range.aggregate(paths, "2025-10-01", "2025-10-01", six_hourly)


def test_cli_aggregate_missing_start_time(monkeypatch, tmp_path, capsys):
    folder = tmp_path / "scores"
    _write_all(folder, [_scores("2025-10-01"), _scores("2025-10-03")])
    args = [
        "aggregate",
        "--spatial_dims", "point",
        "--scores", str(folder),
        "--start", "2025-10-01",
        "--end", "2025-10-03",
        "--by", "start_time",
    ]  # fmt: skip
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, *args, "--output", str(tmp_path / "a.nc"))
    assert exc.value.code == 1
    assert "allow_missing" in capsys.readouterr().err
    assert not (tmp_path / "a.nc").exists()

    _main(monkeypatch, *args, "--allow_missing", "--output", str(tmp_path / "b.nc"))
    with xr.open_dataset(tmp_path / "b.nc") as ds:
        assert ds.attrs["missing_start_times"] == "2025-10-02T00:00"

    # the flag overrides allow_missing of the configuration
    config = tmp_path / "config.yaml"
    config.write_text("aggregation:\n  allow_missing: true\n")
    with pytest.raises(SystemExit) as exc:
        _main(
            monkeypatch, *args, "--config", str(config), "--no-allow_missing",
            "--output", str(tmp_path / "c.nc"),
        )  # fmt: skip
    assert exc.value.code == 1
    assert not (tmp_path / "c.nc").exists()


def test_aggregate_does_not_depend_on_chunk_size(tmp_path):
    starts = pd.date_range("2025-09-27", "2025-10-03", freq="D")
    rng = np.random.default_rng(5)
    datasets = []
    for s in starts:
        ds = _scores(s)
        for name in ds.data_vars:
            ds[name].values[:] = rng.uniform(0, 5, ds[name].shape)
        datasets.append(ds)
    paths = _write_all(tmp_path / "scores", datasets)
    whole = medium_range.aggregate(paths, "2025-10-01", "2025-10-02", _config())
    chunked = medium_range.aggregate(
        paths, "2025-10-01", "2025-10-02", _config(chunk_size=1)
    )
    xr.testing.assert_allclose(whole, chunked)


def test_score_then_aggregate_end_to_end(tmp_path):
    rea = _reanalysis()
    clim = _climatology()
    folder = tmp_path / "scores"
    folder.mkdir()
    paths = []
    for i, start in enumerate(pd.date_range("2025-09-29", "2025-10-02", freq="D")):
        scores = medium_range.score_start_date(
            _forecast(start, seed=i), rea, _config(), climatology=clim
        )
        paths.append(medium_range.write_scores(scores, folder))
    out = medium_range.aggregate(paths, "2025-10-01", "2025-10-01", _config())

    # each step counts the start dates whose valid time falls on 1 October
    for step in out.step.values:
        valid_day = pd.Timestamp("2025-10-01")
        starts = [
            s
            for s in pd.date_range("2025-09-29", "2025-10-02", freq="D")
            if (s + pd.Timedelta(step)).normalize() == valid_day
        ]
        np.testing.assert_array_equal(out["count"].sel(step=step), len(starts))


# ------------------------------------------------------------------ CLI


def _write_inputs(tmp_path):
    _reanalysis().to_netcdf(tmp_path / "reanalysis.nc")
    _climatology().to_netcdf(tmp_path / "climatology.nc")
    folder = tmp_path / "forecasts"
    folder.mkdir()
    for i, start in enumerate(pd.date_range("2025-09-29", "2025-10-02", freq="D")):
        _forecast(start, seed=i).to_netcdf(folder / f"{start:%Y%m%d%H}.nc")
    return folder


def _main(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["hyve-forecast-crps", *argv])
    forecast_crps.main()


def test_cli_score_and_aggregate(monkeypatch, tmp_path):
    folder = _write_inputs(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("structure: medium_range\nspatial_dims: [point]\n")
    _main(
        monkeypatch,
        "score",
        "--config", str(config),
        "--forecast", str(folder),
        "--reanalysis", str(tmp_path / "reanalysis.nc"),
        "--climatology", str(tmp_path / "climatology.nc"),
        "--output", str(tmp_path / "scores"),
        "--chunk_size", "2",
    )  # fmt: skip
    written = sorted(p.name for p in (tmp_path / "scores").iterdir())
    assert written == [
        "scores_2025092900.nc",
        "scores_2025093000.nc",
        "scores_2025100100.nc",
        "scores_2025100200.nc",
    ]

    out = tmp_path / "october.nc"
    _main(
        monkeypatch,
        "aggregate",
        "--config", str(config),
        "--scores", str(tmp_path / "scores"),
        "--start", "2025-10-01",
        "--end", "2025-10-01",
        "--output", str(out),
    )  # fmt: skip
    with xr.open_dataset(out) as ds:
        # valid on 1 October with steps of 6 to 24 h: starts 30 Sep and 1 Oct
        assert ds.attrs["expected_start_times"] == 2
        assert ds.attrs["missing_start_times"] == ""
        assert set(ds.data_vars) == {
            "crps_forecast",
            "crps_persistence",
            "crps_climatology",
            "crpss_persistence",
            "crpss_climatology",
            "count",
        }
        assert ds.attrs["aggregation_by"] == "valid_time"


def test_cli_score_reads_only_the_nc_files_of_a_folder(monkeypatch, tmp_path):
    folder = _write_inputs(tmp_path)
    (folder / "README.txt").write_text("not a forecast")
    _main(
        monkeypatch,
        "score",
        "--forecast", str(folder),
        "--reanalysis", str(tmp_path / "reanalysis.nc"),
        "--spatial_dims", "point",
        "--output", str(tmp_path / "scores"),
    )  # fmt: skip
    assert len(list((tmp_path / "scores").iterdir())) == 4


def test_cli_score_an_empty_folder_exits_1(monkeypatch, tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    _reanalysis().to_netcdf(tmp_path / "reanalysis.nc")
    with pytest.raises(SystemExit) as exc:
        _main(
            monkeypatch,
            "score",
            "--forecast", str(tmp_path / "empty"),
            "--reanalysis", str(tmp_path / "reanalysis.nc"),
            "--spatial_dims", "point",
            "--output", str(tmp_path / "scores"),
        )  # fmt: skip
    assert exc.value.code == 1
    assert "no forecast file" in capsys.readouterr().err


def test_cli_aggregate_min_count_and_start_frequency_flags(monkeypatch, tmp_path):
    folder = tmp_path / "scores"
    a = _scores("2025-10-01 00:00", value=1)
    b = _scores("2025-10-01 12:00", value=3)
    b["crps_forecast"].loc[{"point": 10}] = np.nan
    _write_all(folder, [a, b])
    config = tmp_path / "config.yaml"
    config.write_text("spatial_dims: [point]\naggregation:\n  by: start_time\n")
    out = tmp_path / "out.nc"
    _main(
        monkeypatch,
        "aggregate",
        "--config", str(config),
        "--scores", str(folder),
        "--start", "2025-10-01",
        "--end", "2025-10-01",
        "--min_count", "2",
        "--start_frequency", "12h",
        "--output", str(out),
    )  # fmt: skip
    with xr.open_dataset(out) as ds:
        assert ds.attrs["start_frequency"] == "12h"
        assert ds.attrs["expected_start_times"] == 2
        assert ds.attrs["min_count"] == 2
        assert ds.crps_forecast.sel(point=10).isnull().all()
        np.testing.assert_allclose(ds.crps_forecast.sel(point=[11, 12]), 2)


def test_cli_score_a_single_file_without_config(monkeypatch, tmp_path):
    folder = _write_inputs(tmp_path)
    _main(
        monkeypatch,
        "score",
        "--forecast", str(folder / "2025100100.nc"),
        "--reanalysis", str(tmp_path / "reanalysis.nc"),
        "--spatial_dims", "point",
        "--output", str(tmp_path / "scores"),
    )  # fmt: skip
    assert [p.name for p in (tmp_path / "scores").iterdir()] == ["scores_2025100100.nc"]


def test_cli_aggregate_by_flag_overrides_config(monkeypatch, tmp_path):
    folder = tmp_path / "scores"
    _write_all(folder, [_scores("2025-09-30"), _scores("2025-10-01")])
    config = tmp_path / "config.yaml"
    config.write_text("spatial_dims: [point]\naggregation:\n  by: valid_time\n")
    out = tmp_path / "out.nc"
    _main(
        monkeypatch,
        "aggregate",
        "--config", str(config),
        "--scores", str(folder),
        "--start", "2025-10-01",
        "--end", "2025-10-01",
        "--by", "start_time",
        "--output", str(out),
    )  # fmt: skip
    with xr.open_dataset(out) as ds:
        assert ds.attrs["aggregation_by"] == "start_time"
        np.testing.assert_array_equal(ds["count"], 1)


@pytest.mark.parametrize("from_flags", [False, True])
def test_cli_score_step_layout_from_config(monkeypatch, tmp_path, from_flags):
    _reanalysis().to_netcdf(tmp_path / "reanalysis.nc")
    fc = _step_layout(_forecast("2025-10-01").rename(ensemble="number"))
    fc.to_netcdf(tmp_path / "forecast.nc")
    config = tmp_path / "config.yaml"
    layout = ["--lead_time_dim", "step", "--ensemble_dim", "number"]
    if from_flags:
        config.write_text("spatial_dims: [point]\n")
    else:
        config.write_text(
            "spatial_dims: [point]\nlead_time_dim: step\nensemble_dim: number\n"
        )
        layout = []
    _main(
        monkeypatch,
        "score",
        "--config", str(config),
        *layout,
        "--forecast", str(tmp_path / "forecast.nc"),
        "--reanalysis", str(tmp_path / "reanalysis.nc"),
        "--output", str(tmp_path / "scores"),
    )  # fmt: skip
    expected = medium_range.score_start_date(
        _forecast("2025-10-01"), _reanalysis(), _config()
    )
    with xr.open_dataset(tmp_path / "scores" / "scores_2025100100.nc") as ds:
        xr.testing.assert_allclose(ds[list(expected.data_vars)], expected)


@pytest.mark.parametrize("command", ["score", "aggregate"])
def test_cli_dask_settings_come_from_the_config(monkeypatch, tmp_path, command):
    seen = {}

    def fake_run(args, config):
        seen.update(
            scheduler=dask.config.get("scheduler"),
            num_workers=dask.config.get("num_workers", None),
        )

    monkeypatch.setattr(cli, f"run_{command}", fake_run)
    config = tmp_path / "config.yaml"
    config.write_text("threads: 3\nscheduler: synchronous\n")
    inputs = {
        "score": ["--forecast", "f.nc", "--reanalysis", "r.nc"],
        "aggregate": ["--scores", "s", "--start", "2025-10-01", "--end", "2025-10-01"],
    }[command]
    argv = [command, "--config", str(config), *inputs, "--output", "out"]
    _main(monkeypatch, *argv)
    assert seen == {"scheduler": "synchronous", "num_workers": 3}
    _main(monkeypatch, *argv, "--threads", "2")
    assert seen == {"scheduler": "synchronous", "num_workers": 2}


def test_cli_subcommand_rejects_a_reforecast_config(monkeypatch, tmp_path, capsys):
    folder = _write_inputs(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text("structure: reforecast\n")
    with pytest.raises(SystemExit) as exc:
        _main(
            monkeypatch,
            "score",
            "--config", str(config),
            "--forecast", str(folder),
            "--reanalysis", str(tmp_path / "reanalysis.nc"),
            "--output", str(tmp_path / "scores"),
        )  # fmt: skip
    assert exc.value.code == 1
    assert "medium_range" in capsys.readouterr().err


def test_cli_missing_data_exits_1(monkeypatch, tmp_path, capsys):
    folder = _write_inputs(tmp_path)
    _reanalysis(end="2025-10-01").to_netcdf(tmp_path / "short.nc")
    with pytest.raises(SystemExit) as exc:
        _main(
            monkeypatch,
            "score",
            "--forecast", str(folder / "2025100100.nc"),
            "--reanalysis", str(tmp_path / "short.nc"),
            "--spatial_dims", "point",
            "--output", str(tmp_path / "scores"),
        )  # fmt: skip
    assert exc.value.code == 1
    # log lines may come first
    errors = [x for x in capsys.readouterr().err.splitlines() if x.startswith("Error:")]
    assert len(errors) == 1
    assert "must cover the longest lead time" in errors[0]
    assert not (tmp_path / "scores" / "scores_2025100100.nc").exists()
