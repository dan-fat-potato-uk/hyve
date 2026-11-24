# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

"""Medium-range structure: score one forecast per start date, then average
the per-start-date scores over a window.

A forecast file holds an ensemble dimension (``ensemble_dim``), a lead-time
dimension and the spatial dimensions, with its start time in the scalar
coordinate ``forecast_reference_time``. The lead-time dimension
(``lead_time_dim``) is ``time``, holding valid times, or ``step``, holding
lead times from the start time. Scores are written per lead time (``step``), so
that `aggregate` can select samples by valid time or by start time.
"""

import logging as log
import os
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr

from hyve.tools.forecast_crps.chunks import chunk_spatially
from hyve.tools.forecast_crps.crps import check_dims, forecast_crps, persistence_crps
from hyve.tools.forecast_crps.errors import (
    InputContractError,
    MissingDataError,
    mismatch_error_class,
)
from hyve.tools.forecast_crps.matching import match_coords, narrow, relabel

FORECAST = "crps_forecast"
PERSISTENCE = "crps_persistence"
CLIMATOLOGY = "crps_climatology"
# benchmark score -> skill score against it
SKILL = {PERSISTENCE: "crpss_persistence", CLIMATOLOGY: "crpss_climatology"}


def start_time(ds, name):
    """Start time of a forecast or score dataset, from ``forecast_reference_time``."""
    if "forecast_reference_time" not in ds.coords:
        hint = ""
        if "forecast_reference_time" in ds.attrs:
            hint = " (it is an attribute; write it as a coordinate)"
        raise InputContractError(
            f"{name} has no 'forecast_reference_time' scalar coordinate{hint}"
        )
    frt = ds["forecast_reference_time"]
    if frt.ndim != 0:
        raise InputContractError(
            f"{name}: 'forecast_reference_time' must be a scalar coordinate,"
            f" it has dimensions {list(frt.dims)}"
        )
    return pd.Timestamp(frt.values)


def _variable(ds, variable, name):
    if isinstance(ds, xr.DataArray):
        return ds
    if variable not in ds.data_vars:
        raise InputContractError(
            f"{name} has no variable {variable!r} (its variables are"
            f" {list(ds.data_vars)}). Set 'variable' in the configuration"
        )
    return ds[variable]


def _rename_ensemble(da, ensemble_dim):
    if ensemble_dim != "ensemble" and ensemble_dim in da.dims:
        da = da.rename({ensemble_dim: "ensemble"})
    return da


def to_valid_time(fc, frt, config, name):
    """Forecast `fc` on ``ensemble`` and ``time`` (valid times, sorted).

    With ``lead_time_dim: step`` the ``step`` dimension holds lead times from
    the start time `frt`; it is replaced by their valid times. A
    ``valid_time(step)`` coordinate, if any, is ignored.
    """
    fc = _rename_ensemble(fc, config.ensemble_dim)
    if config.lead_time_dim == "step":
        if "step" not in fc.dims:
            raise InputContractError(
                f"{name} has no 'step' dimension (its dimensions are"
                f" {list(fc.dims)}); set 'lead_time_dim' in the configuration"
            )
        if "time" in fc.dims:
            raise InputContractError(
                f"{name} has both 'step' and 'time' dimensions; with"
                " lead_time_dim: step, 'time' must not be a dimension"
            )
        if not np.issubdtype(fc["step"].dtype, np.timedelta64):
            raise InputContractError(
                f"{name}: 'step' must hold time differences, it has dtype"
                f" {fc['step'].dtype}"
            )
        valid = frt + pd.TimedeltaIndex(fc["step"].values)
        fc = (
            fc.drop_vars(["time", "valid_time"], errors="ignore")
            .assign_coords(time=("step", valid.values))
            .swap_dims(step="time")
            .drop_vars("step")
        )
    if "time" in fc.dims:
        fc = fc.sortby("time")
    return fc


def _times(values):
    return [str(np.datetime_as_string(np.datetime64(v), unit="m")) for v in values]


def select_climatology(clim, valid_times):
    """Climatology members for each valid time, on a ``time`` dimension.

    The day of year is that of the valid date; the issue hour is the slot
    that holds the valid hour, the largest ``issue_hour`` not after it, as
    `hyve-clim-benchmark` builds its slots.
    """
    valid = pd.DatetimeIndex(valid_times)
    hours = np.sort(np.asarray(clim["issue_hour"].values))
    pos = np.searchsorted(hours, valid.hour, side="right") - 1
    if (pos < 0).any():
        missing = sorted(set(valid.hour[pos < 0].tolist()))
        raise InputContractError(
            f"climatology has no issue_hour slot for valid hour(s) {missing}"
            f" (its issue_hour values are {hours.tolist()})"
        )
    selected = clim.sel(
        doy=xr.DataArray(valid.dayofyear.values, dims="time"),
        issue_hour=xr.DataArray(hours[pos], dims="time"),
    )
    selected = selected.reset_coords(drop=True)
    return selected.assign_coords(time=valid.values)


def _match_climatology(clim, forecast, spatial_dims, name, coord_tolerance):
    """`clim` on the grid of `forecast`, which it must cover entirely."""
    check_dims(clim, "climatology", ["ensemble", "doy", "issue_hour", *spatial_dims])
    matches = match_coords(
        forecast, clim, spatial_dims, name, "climatology", coord_tolerance
    )
    for dim, (ia, _) in matches.items():
        if len(ia) < forecast.sizes[dim]:
            raise mismatch_error_class(dim)(
                f"climatology lacks {forecast.sizes[dim] - len(ia)} of the"
                f" {forecast.sizes[dim]} {dim!r} values of {name}"
            )
    return relabel(clim, matches, forecast)


def score_start_date(forecast, reanalysis, config, climatology=None, name="forecast"):
    """CRPS of one medium-range forecast per lead time, with its benchmarks.

    `forecast` and `reanalysis` are datasets (or data arrays) following the
    input contract; `climatology` is optional. Returns a dataset with
    ``crps_forecast``, ``crps_persistence`` and, with a climatology,
    ``crps_climatology`` on (``step``, *spatial dims*) in float32, with the
    coordinates ``valid_time(step)`` and ``forecast_reference_time``.

    Step 0 (the initial condition) is not scored. The forecast grid is the
    reference: the reanalysis and climatology are matched to it. Missing data
    is NaN per cell; MissingDataError is raised only when the reanalysis has
    no value at any scored point for a lead time or for the start time.
    """
    spatial_dims = list(config.spatial_dims)
    tolerance = config.coord_tolerance
    chunk_size = config.chunk_size

    frt = start_time(forecast, name)
    fc = to_valid_time(_variable(forecast, config.variable, name), frt, config, name)
    rea = _variable(reanalysis, config.variable, "reanalysis")
    check_dims(fc, name, ["ensemble", "time", *spatial_dims])
    check_dims(rea, "reanalysis", ["time", *spatial_dims])

    valid = pd.DatetimeIndex(fc["time"].values)
    if (valid < frt).any():
        raise InputContractError(
            f"{name} has valid times {_times(valid[valid < frt])} before its"
            f" forecast_reference_time {_times([frt])[0]}"
        )
    fc = fc.sel(time=valid[valid > frt])
    if fc.sizes["time"] == 0:
        raise InputContractError(f"{name} has no lead time after its start")

    matches = match_coords(fc, rea, spatial_dims, name, "reanalysis", tolerance)
    rea = relabel(rea, matches, fc)
    fc = narrow(fc, matches)

    fc = chunk_spatially(fc, spatial_dims, chunk_size)
    rea = chunk_spatially(rea, spatial_dims, chunk_size)
    valid = fc["time"].values
    obs = rea.reindex(time=valid)
    persistence = rea.reindex(time=[frt.to_datetime64()])

    # points the forecast has data for; others are NaN in every score
    scored = fc.isel(time=0, ensemble=0, drop=True).notnull()
    has_obs = obs.where(scored).notnull().any(dim=spatial_dims)
    has_pers = persistence.where(scored).notnull().any()
    any_scored, has_obs, has_pers = dask.compute(scored.any(), has_obs, has_pers)
    if not any_scored:
        raise MissingDataError(f"{name} has no data at any point")
    if not has_obs.all():
        raise MissingDataError(
            "reanalysis has no value at any scored point for valid time(s)"
            f" {_times(valid[~has_obs.values])} of {name}. The reanalysis must cover"
            " the longest lead time, and every lead time must be a reanalysis"
            " time: score only the steps the reanalysis has (e.g. the 6-hourly"
            " ones against a 6-hourly reanalysis)"
        )
    if not has_pers:
        raise MissingDataError(
            f"reanalysis has no value at the start time {_times([frt])[0]} of"
            f" {name}, needed for persistence"
        )

    scores = {
        FORECAST: forecast_crps(fc, obs, core_dims=spatial_dims),
        PERSISTENCE: persistence_crps(obs, persistence).where(scored),
    }
    if climatology is not None:
        clim = _variable(climatology, config.variable, "climatology")
        clim = _rename_ensemble(clim, config.ensemble_dim)
        clim = _match_climatology(clim, fc, spatial_dims, name, tolerance)
        clim = chunk_spatially(clim, spatial_dims, chunk_size, extra={"doy": 1})
        clim = select_climatology(clim, valid)
        scores[CLIMATOLOGY] = forecast_crps(clim, obs, core_dims=spatial_dims).where(
            scored
        )

    computed = dask.compute(*scores.values())
    ds = xr.Dataset(
        {
            key: value.transpose("time", *spatial_dims).astype(np.float32)
            for key, value in zip(scores, computed)
        }
    )
    steps = pd.DatetimeIndex(valid) - frt
    ds = (
        ds.assign_coords(step=("time", steps.values), valid_time=("time", valid))
        .swap_dims(time="step")
        .drop_vars("time")
        .assign_coords(forecast_reference_time=frt.to_datetime64())
    )
    ds.attrs.update(structure="medium_range", variable=config.variable)
    return ds


def _write_atomic(ds, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        ds.to_netcdf(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def score_file_name(frt):
    return f"scores_{pd.Timestamp(frt):%Y%m%d%H}.nc"


def write_scores(scores, folder):
    """Write `scores` to `folder` as ``scores_YYYYMMDDHH.nc``; returns the path."""
    frt = start_time(scores, "scores")
    return _write_atomic(scores, Path(folder) / score_file_name(frt))


def _window(start, end):
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if end < start:
        raise InputContractError(
            f"window end {end:%Y-%m-%d} is before its start {start:%Y-%m-%d}"
        )
    return start, end + pd.Timedelta(days=1)


def _in_window(ds, by, start, stop):
    """Whether the score dataset `ds` has any sample in ``[start, stop)``."""
    frt = start_time(ds, "scores")
    if by == "start_time":
        return start <= frt < stop
    steps = pd.TimedeltaIndex(ds["step"].values)
    return frt + steps.max() >= start and frt + steps.min() < stop


def _check_same_grid(datasets, names, spatial_dims):
    ref, ref_name = datasets[0], names[0]
    for ds, name in zip(datasets[1:], names[1:]):
        for dim in spatial_dims:
            if not np.array_equal(ds[dim].values, ref[dim].values):
                raise mismatch_error_class(dim)(
                    f"{name} and {ref_name} have different {dim!r} values;"
                    " every score file of a window must be on the same grid"
                )


def _check_unique_start_times(datasets, names):
    seen = {}
    for ds, name in zip(datasets, names):
        frt = start_time(ds, name)
        if frt in seen:
            raise InputContractError(
                f"{seen[frt]} and {name} have the same start time"
                f" {_times([frt])[0]}; a window must have one score file per"
                " start time"
            )
        seen[frt] = name


def expected_start_times(datasets, by, start, stop, frequency):
    """Start times on the `frequency` grid with samples in ``[start, stop)``.

    By start time these are the grid times in the window. By valid time they
    are those whose lead times, the steps of the score files `datasets`, reach
    into the window, so a forecast started before the window is expected when
    its longest lead time falls in it.
    """
    lo, hi = start, stop
    if by == "valid_time":
        steps = pd.TimedeltaIndex(
            np.unique(np.concatenate([ds["step"].values for ds in datasets]))
        )
        lo, hi = start - steps.max(), stop - steps.min()
    grid = pd.date_range(lo.ceil(frequency), hi, freq=frequency)
    return grid[grid < hi]


def _check_complete(datasets, names, by, start, stop, aggregation):
    """Expected and missing start times of the window; raise if any is missing.

    With ``aggregation.allow_missing`` a missing start time is a warning.
    """
    expected = expected_start_times(datasets, by, start, stop, aggregation.start_step)
    present = {start_time(ds, name) for ds, name in zip(datasets, names)}
    missing = [t for t in expected if t not in present]
    if missing:
        shown = _times(missing[:10]) + (["..."] if len(missing) > 10 else [])
        message = (
            f"the window {start:%Y-%m-%d} to {stop - pd.Timedelta(days=1):%Y-%m-%d}"
            f" (by {by}) lacks the score files of {len(missing)} of the"
            f" {len(expected)} start times every {aggregation.start_frequency}:"
            f" {', '.join(shown)}"
        )
        if not aggregation.allow_missing:
            raise MissingDataError(
                f"{message}. Score them first, or set aggregation.allow_missing"
                " to aggregate without them"
            )
        log.warning(message)
    return expected, missing


def aggregate(paths, start, end, config):
    """Means of per-start-date scores over the window from `start` to `end`.

    `start` and `end` are inclusive dates. A sample (start date, step) is in
    the window when the calendar date of its valid time (or of its start time,
    with ``aggregation.by: start_time``) is. Each mean is taken over the same
    samples: those where every score in the files is finite. Returns a
    dataset on (``step``, *spatial dims*) with the means, the skill scores
    ``1 - forecast / benchmark`` (NaN where the benchmark mean is 0) and
    ``count``; cells with fewer than ``aggregation.min_count`` samples are NaN.

    Every start time on the ``aggregation.start_frequency`` grid with samples
    in the window must have a score file (see `expected_start_times`);
    otherwise MissingDataError is raised, or with ``aggregation.allow_missing``
    a warning is logged. The missing start times are recorded in the
    ``missing_start_times`` attribute.
    """
    spatial_dims = list(config.spatial_dims)
    by = config.aggregation.by
    start, stop = _window(start, end)
    chunks = {} if config.chunk_size is not None else None

    paths = sorted(Path(p) for p in paths)
    selected, names = [], []
    for path in paths:
        ds = xr.open_dataset(path, chunks=chunks, decode_timedelta=True)
        if _in_window(ds, by, start, stop):
            selected.append(ds)
            names.append(f"score file {path.name}")
        else:
            ds.close()
    if not selected:
        raise MissingDataError(
            f"none of the {len(paths)} score files has a sample in the window"
            f" {start:%Y-%m-%d} to {stop - pd.Timedelta(days=1):%Y-%m-%d}"
            f" (by {by})"
        )
    log.info(f"{len(selected)} of {len(paths)} score files in the window")

    try:
        for ds, name in zip(selected, names):
            for key in (FORECAST, PERSISTENCE):
                if key not in ds:
                    raise InputContractError(f"{name} has no {key!r}")
            check_dims(ds[FORECAST], name, ["step", *spatial_dims])
        with_clim = [CLIMATOLOGY in ds for ds in selected]
        if any(with_clim) and not all(with_clim):
            raise InputContractError(
                f"{CLIMATOLOGY!r} is in some score files of the window but not in"
                f" others (missing from {names[with_clim.index(False)]})"
            )
        _check_same_grid(selected, names, spatial_dims)
        _check_unique_start_times(selected, names)
        expected, missing = _check_complete(
            selected, names, by, start, stop, config.aggregation
        )

        keys = [FORECAST, PERSISTENCE] + ([CLIMATOLOGY] if all(with_clim) else [])
        combined = xr.concat(
            [ds[keys].drop_vars("valid_time", errors="ignore") for ds in selected],
            dim="forecast_reference_time",
            join="outer",
            coords="minimal",
            compat="override",
        )
        combined = chunk_spatially(combined, spatial_dims, config.chunk_size)

        frt = combined["forecast_reference_time"]
        when = frt + combined["step"] if by == "valid_time" else frt
        samples = (when >= start.to_datetime64()) & (when < stop.to_datetime64())
        for key in keys:
            samples = samples & combined[key].notnull()
        count = samples.sum("forecast_reference_time")
        enough = count >= config.aggregation.min_count

        means = {}
        for key in keys:
            total = (
                combined[key]
                .astype(np.float64)
                .where(samples)
                .sum("forecast_reference_time")
            )
            means[key] = total.where(enough) / count.where(enough)
        out = dict(means)
        for key in keys[1:]:
            bench = means[key]
            out[SKILL[key]] = 1 - means[FORECAST] / bench.where(bench > 0)

        result = xr.Dataset(
            {
                key: value.transpose("step", *spatial_dims).astype(np.float32)
                for key, value in out.items()
            }
        )
        result["count"] = count.transpose("step", *spatial_dims).astype(np.int32)
        result = result.compute()
    finally:
        for ds in selected:
            ds.close()

    result.attrs.update(
        structure="medium_range",
        window_start=f"{start:%Y-%m-%d}",
        window_end=f"{stop - pd.Timedelta(days=1):%Y-%m-%d}",
        aggregation_by=by,
        min_count=config.aggregation.min_count,
        score_files=len(selected),
        start_frequency=config.aggregation.start_frequency,
        expected_start_times=len(expected),
        missing_start_times=", ".join(_times(missing)),
    )
    return result


def write_aggregate(result, path):
    """Write the result of `aggregate` to `path`, replacing it atomically."""
    return _write_atomic(result, path)
