# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

"""Command line interface of hyve-forecast-crps."""

import argparse
import logging as log
import math
import os
import sys
from pathlib import Path

import dask
import xarray as xr
from pydantic import ValidationError

from hyve.tools.forecast_crps import medium_range
from hyve.tools.forecast_crps.chunks import chunk_spatially
from hyve.tools.forecast_crps.config import load_config
from hyve.tools.forecast_crps.crps import check_dims, require_variable
from hyve.tools.forecast_crps.errors import (
    CoordinateMismatchError,
    InputContractError,
    MissingDataError,
)
from hyve.tools.forecast_crps.matching import match_coords, narrow, relabel
from hyve.tools.forecast_crps.scores import compute_score, latest_first

# subcommands of the medium-range structure; without one the arguments are
# those of the reforecast scoring
SUBCOMMANDS = ("score", "aggregate")


def _coord_tolerance(text):
    """argparse type: a finite, non-negative float."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"{text!r} must be finite and >= 0")
    return value


def _positive_int(text):
    """argparse type: an integer >= 1."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r} must be >= 1")
    return value


def main():
    if len(sys.argv) > 1 and sys.argv[1] in SUBCOMMANDS:
        medium_range_main(sys.argv[1:])
        return

    parser = argparse.ArgumentParser(
        prog="hyve-forecast-crps",
        description=(
            "Score reforecasts. For medium-range forecasts use the 'score' and"
            " 'aggregate' subcommands (hyve-forecast-crps score --help)."
        ),
    )
    parser.add_argument(
        "--config",
        help=(
            "YAML configuration file with 'structure: reforecast'; the flags"
            " below override it"
        ),
    )
    parser.add_argument(
        "--variable", help="variable in every input file (default: dis)"
    )
    parser.add_argument("--reanalysis", required=True, help="reanalysis dataset file")
    parser.add_argument("--reforecast", required=True, help="reforecast dataset folder")
    parser.add_argument(
        "--climatology",
        help=(
            "climatology dataset file with dimensions ensemble, doy, issue_hour"
            " and the spatial dimensions (optional)"
        ),
    )
    parser.add_argument(
        "--output", help="output folder for individual crps values", default="."
    )
    parser.add_argument(
        "--spatial_dims",
        nargs="+",
        default=None,
        help=(
            "spatial dimensions of the input datasets, for example 'station',"
            " 'lat lon' or 'y x' (default: station)"
        ),
    )
    parser.add_argument(
        "--core_dim",
        default=None,
        help="single spatial dimension; same as --spatial_dims with one name",
    )
    parser.add_argument(
        "--coord_tolerance",
        type=_coord_tolerance,
        default=None,
        help=(
            "absolute tolerance for matching float coordinates (lat, lon, y, x),"
            " in the units of the coordinates; default: relative to the float"
            " precision of the coordinates"
        ),
    )
    parser.add_argument(
        "--chunk_size",
        type=_positive_int,
        default=None,
        help=(
            "read every input lazily, in chunks of about this many spatial"
            " points per chunk; peak memory is close to the data held in memory"
            " plus threads x 8 x chunk bytes, counted in float64 (default: read"
            " whole files)"
        ),
    )
    parser.add_argument(
        "--threads",
        type=_positive_int,
        default=None,
        help="number of dask threads (default: one per CPU core)",
    )
    parser.add_argument(
        "--with_init",
        action="store_true",
        default=None,
        help=(
            "the first time step of each reforecast file is the initial"
            " condition, which is then also the persistence forecast; without"
            " it the persistence date is one time step before the first step"
        ),
    )
    parser.add_argument(
        "--scheduler",
        default=None,
        choices=["synchronous", "threads"],
        help="dask scheduler (default: threads)",
    )
    parser.add_argument(
        "--log",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="set logging level",
    )

    args = parser.parse_args()
    if args.spatial_dims is not None and args.core_dim is not None:
        parser.error("use either --spatial_dims or --core_dim, not both")
    log.basicConfig(
        level=args.log, format="crps - (%(processName)-10s) %(levelname)s: %(message)s"
    )

    try:
        apply_config(args, reforecast_config(args))
        run(args)
    except (CoordinateMismatchError, InputContractError, ValidationError) as err:
        sys.stderr.write(f"Error: {err}\n")
        sys.exit(1)


def reforecast_config(args):
    """Configuration of a reforecast run: file, then flags."""
    spatial_dims = args.spatial_dims or ([args.core_dim] if args.core_dim else None)
    config = load_config(
        args.config,
        defaults={"structure": "reforecast"},
        variable=args.variable,
        spatial_dims=spatial_dims,
        coord_tolerance=args.coord_tolerance,
        chunk_size=args.chunk_size,
        threads=args.threads,
        scheduler=args.scheduler,
        with_init=args.with_init,
        climatology=args.climatology,
    )
    if config.structure != "reforecast":
        raise InputContractError(
            f"the configuration has 'structure: {config.structure}'; medium-range"
            " forecasts are scored with the 'score' and 'aggregate' subcommands"
            " (hyve-forecast-crps score --help)"
        )
    return config


def apply_config(args, config):
    """Set the settings of `config` on the parsed arguments that `run` reads."""
    args.variable = config.variable
    args.spatial_dims = list(config.spatial_dims)
    args.core_dim = None
    args.coord_tolerance = config.coord_tolerance
    args.chunk_size = config.chunk_size
    args.threads = config.threads
    args.scheduler = config.scheduler
    args.with_init = config.with_init
    args.climatology = str(config.climatology) if config.climatology else None


def spatial_dims_from_args(args):
    """Spatial dimensions requested on the command line, station by default."""
    if getattr(args, "spatial_dims", None):
        return list(args.spatial_dims)
    if getattr(args, "core_dim", None):
        return [args.core_dim]
    return ["station"]


def open_climatology(path, spatial_dims, chunk_size=None):
    """Climatology data array, chunked by day of year and spatial points.

    The climatology has no time dimension; one day of year per chunk means a
    forecast reads only the days it is scored against.
    """
    da = xr.open_dataarray(path)
    return chunk_spatially(da, spatial_dims, chunk_size, extra={"doy": 1})


def dask_settings(args):
    """dask configuration for the scheduler and thread count of `args`."""
    settings = {"scheduler": getattr(args, "scheduler", None) or "threads"}
    threads = getattr(args, "threads", None)
    if threads is not None:
        settings["num_workers"] = threads
    return settings


def run(args):
    log.info("Computing the scoring using crps approach")

    with dask.config.set(**dask_settings(args)):

        spatial_dims = spatial_dims_from_args(args)
        log.info(f"Spatial dimensions: {spatial_dims}")
        variable = getattr(args, "variable", None) or "dis"
        chunk_size = getattr(args, "chunk_size", None)
        if chunk_size is not None:
            log.info(f"Reading inputs in chunks of about {chunk_size} points")

        # read reanalysis dataset
        ds_reanalysis = chunk_spatially(
            xr.open_dataset(args.reanalysis), spatial_dims, chunk_size
        )

        require_variable(ds_reanalysis, variable, "reanalysis")

        # the reanalysis holds only time and the spatial dimensions
        check_dims(
            ds_reanalysis[variable],
            f"reanalysis {variable!r}",
            [*spatial_dims, "time"],
            hint=". Set --spatial_dims to its spatial dimensions",
        )

        # the latest reforecast file is the reference grid, as in compute_score
        reforecast_files = latest_first(args.reforecast)
        log.info("Found {} files".format(len(reforecast_files)))
        log.debug(reforecast_files)
        ref_name = f"reforecast file {os.path.basename(reforecast_files[0])}"
        ds_reforecast = xr.open_dataset(reforecast_files[0])
        require_variable(ds_reforecast, variable, ref_name)
        tolerance = getattr(args, "coord_tolerance", None)

        log.info("Reanalysis dataset from {}:".format(args.reanalysis))
        log.debug(ds_reanalysis[variable])
        log.info("Reforecast dataset from {}:".format(args.reforecast))
        log.debug(ds_reforecast[variable])

        # observations and climatology are relabelled to the forecast grid
        matches = match_coords(
            ds_reforecast,
            ds_reanalysis,
            spatial_dims,
            ref_name,
            "reanalysis",
            tolerance,
        )
        ds_reanalysis = relabel(ds_reanalysis, matches, ds_reforecast)
        ds_reforecast = narrow(ds_reforecast, matches)
        log.debug(ds_reanalysis[variable])
        log.debug(ds_reforecast[variable])

        ds_clim = None
        if args.climatology:
            ds_clim = open_climatology(args.climatology, spatial_dims, chunk_size)
            matches = match_coords(
                ds_reanalysis,
                ds_clim,
                spatial_dims,
                "reanalysis",
                "climatology",
                tolerance,
            )
            ds_clim = relabel(ds_clim, matches, ds_reanalysis)
            ds_clim = ds_clim.reindex(
                {dim: ds_reanalysis[dim].values for dim in spatial_dims}
            ).sel(issue_hour=0)
            log.debug(ds_clim)

        if args.output:
            os.makedirs(args.output, exist_ok=True)

        compute_score(
            args.output,
            args.reforecast,
            ds_reanalysis[variable],
            ds_clim,
            spatial_dims,
            args.with_init,
            tolerance,
            chunk_size=chunk_size,
            variable=variable,
        )


def _medium_range_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config", help="YAML configuration file; the flags below override it"
    )
    common.add_argument(
        "--variable", help="variable in every input file (default: dis)"
    )
    common.add_argument(
        "--spatial_dims",
        nargs="+",
        help="spatial dimensions of the inputs, for example 'point' or 'lat lon'",
    )
    common.add_argument(
        "--coord_tolerance",
        type=_coord_tolerance,
        help="absolute tolerance for matching float coordinates",
    )
    common.add_argument(
        "--chunk_size",
        type=_positive_int,
        help="read inputs lazily in chunks of about this many spatial points",
    )
    common.add_argument("--threads", type=_positive_int, help="number of dask threads")
    common.add_argument(
        "--scheduler",
        choices=["synchronous", "threads"],
        help="dask scheduler (default: threads)",
    )
    common.add_argument(
        "--log",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="set logging level",
    )

    parser = argparse.ArgumentParser(
        prog="hyve-forecast-crps",
        description=(
            "Medium-range scoring. Without a subcommand, hyve-forecast-crps"
            " scores reforecasts (see hyve-forecast-crps --help)."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    score = commands.add_parser(
        "score",
        parents=[common],
        help="score each forecast (one start date per file) per lead time",
    )
    score.add_argument(
        "--forecast",
        required=True,
        help="forecast file, or folder of forecast files (*.nc)",
    )
    score.add_argument("--reanalysis", required=True, help="reanalysis file")
    score.add_argument("--climatology", help="climatology file (optional)")
    score.add_argument(
        "--lead_time_dim",
        choices=["time", "step"],
        help=(
            "lead-time dimension of the forecast: 'time' holds valid times,"
            " 'step' lead times from forecast_reference_time (default: time)"
        ),
    )
    score.add_argument(
        "--ensemble_dim",
        help="ensemble dimension of the forecast (default: ensemble)",
    )
    score.add_argument(
        "--output",
        required=True,
        help="folder for the scores_YYYYMMDDHH.nc files",
    )

    agg = commands.add_parser(
        "aggregate",
        parents=[common],
        help="average per-start-date scores over a window of dates",
    )
    agg.add_argument(
        "--scores", required=True, help="folder of scores_YYYYMMDDHH.nc files"
    )
    agg.add_argument("--start", required=True, help="first date of the window")
    agg.add_argument("--end", required=True, help="last date of the window")
    agg.add_argument(
        "--by",
        choices=["valid_time", "start_time"],
        help="select samples by valid time or start time (default: valid_time)",
    )
    agg.add_argument(
        "--min_count",
        type=_positive_int,
        help="cells with fewer samples are NaN (default: 1)",
    )
    agg.add_argument(
        "--start_frequency",
        help=(
            "time between forecast start times, a divisor of a day such as 1D,"
            " 12h or 6h (default: 1D)"
        ),
    )
    agg.add_argument(
        "--allow_missing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "aggregate without the score files of missing start times, with a"
            " warning; --no-allow_missing overrides the configuration"
            " (default: a missing start time is an error)"
        ),
    )
    agg.add_argument("--output", required=True, help="output file")
    return parser


def medium_range_config(args):
    """Configuration of a medium-range subcommand: file, then flags."""
    overrides = {
        "variable": args.variable,
        "spatial_dims": args.spatial_dims,
        "coord_tolerance": args.coord_tolerance,
        "chunk_size": args.chunk_size,
        "threads": args.threads,
        "scheduler": args.scheduler,
    }
    if args.command == "score":
        overrides["climatology"] = args.climatology
        overrides["lead_time_dim"] = args.lead_time_dim
        overrides["ensemble_dim"] = args.ensemble_dim
    else:
        overrides["aggregation"] = {
            "by": args.by,
            "min_count": args.min_count,
            "start_frequency": args.start_frequency,
            "allow_missing": args.allow_missing,
        }
    config = load_config(
        args.config, defaults={"structure": "medium_range"}, **overrides
    )
    if config.structure != "medium_range":
        raise InputContractError(
            f"'{args.command}' needs 'structure: medium_range' in the"
            f" configuration, got {config.structure!r}"
        )
    return config


def forecast_paths(path):
    """The forecast file `path`, or the forecast files of the folder `path`."""
    path = Path(path)
    if not path.is_dir():
        return [path]
    paths = sorted(path.glob("*.nc"))
    if not paths:
        raise MissingDataError(f"no forecast file (*.nc) in {path}")
    return paths


def run_score(args, config):
    reanalysis = xr.open_dataset(args.reanalysis)
    climatology = None
    if config.climatology is not None:
        climatology = xr.open_dataset(config.climatology)
    os.makedirs(args.output, exist_ok=True)
    for path in forecast_paths(args.forecast):
        log.info(f"Scoring {path}")
        with xr.open_dataset(path) as forecast:
            scores = medium_range.score_start_date(
                forecast,
                reanalysis,
                config,
                climatology=climatology,
                name=f"forecast file {path.name}",
            )
        scores.attrs.update(
            forecast_file=str(path),
            reanalysis_file=str(args.reanalysis),
            climatology_file=str(config.climatology or ""),
        )
        written = medium_range.write_scores(scores, args.output)
        log.info(f"Wrote {written}")


def run_aggregate(args, config):
    paths = sorted(Path(args.scores).glob("scores_*.nc"))
    result = medium_range.aggregate(paths, args.start, args.end, config)
    medium_range.write_aggregate(result, args.output)
    log.info(f"Wrote {args.output}")


def medium_range_main(argv):
    """Entry point of the ``score`` and ``aggregate`` subcommands."""
    args = _medium_range_parser().parse_args(argv)
    log.basicConfig(
        level=args.log, format="crps - (%(processName)-10s) %(levelname)s: %(message)s"
    )
    try:
        config = medium_range_config(args)
        with dask.config.set(**dask_settings(config)):
            if args.command == "score":
                run_score(args, config)
            else:
                run_aggregate(args, config)
    except (
        CoordinateMismatchError,
        InputContractError,
        MissingDataError,
        ValidationError,
    ) as err:
        sys.stderr.write(f"Error: {err}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
