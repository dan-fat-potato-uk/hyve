<!--
SPDX-FileCopyrightText: 2025 European Centre for Medium-Range Weather Forecasts (ECMWF)

SPDX-License-Identifier: Apache-2.0
-->

# Hyve

<p align="center">
    <a href="https://github.com/ecmwf/codex/raw/refs/heads/main/ESEE">
        <img src="https://github.com/ecmwf/codex/raw/refs/heads/main/ESEE/foundation_badge.svg" alt="ECMWF Software EnginE">
    </a>
    <a href="https://github.com/ecmwf/codex/raw/refs/heads/main/Project Maturity">
        <img src="https://github.com/ecmwf/codex/raw/refs/heads/main/Project Maturity/emerging_badge.svg" alt="Maturity Level">
    </a>
    <a href="https://opensource.org/licenses/apache-2-0">
        <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0">
    </a>
    <a href="https://github.com/ecmwf/hyve/releases">
        <img src="https://img.shields.io/github/v/release/ecmwf/hyve?color=purple&label=Release" alt="Latest Release">
    </a>
</p>

<p align="center">
  <a href="#installation">Installation</a> *
  <a href="#development">Development</a>
</p>

> \[!IMPORTANT\]
> This software is **Emerging** and subject to ECMWF's guidelines on [Software Maturity](https://github.com/ecmwf/codex/raw/refs/heads/main/Project%20Maturity).

Hyve is a library and toolkit for hydrological verification.

## Installation

```
pip install git+https://github.com/ecmwf/hyve.git@main
```

### Development

Install project with uv

```
# (Optional) Create and activate virtual environment
uv venv .venv --python 3.10
source .venv/bin/activate

# Install hyve with development dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

## Inputs

### `hyve-forecast-crps`

The variable `dis` (or the one set with `--variable`) of each input has exactly
these dimensions (any order). The spatial dimensions are chosen with `--spatial_dims` (default `station`, for
example `lat lon` or `y x`) or `--core_dim` for a single one.

| Input        | `dis` dimensions                                  |
| ------------ | ------------------------------------------------- |
| Reanalysis   | `time`, *spatial dims*                            |
| Reforecast   | `ensemble`, `time`, *spatial dims* (one file per date) |
| Climatology  | `ensemble`, `doy`, `issue_hour`, *spatial dims*   |

The reanalysis may not have any other dimension (such as `ensemble` or
`system`); otherwise the tool exits with a dimension mismatch error. The
reforecast grid is the reference and the other inputs are matched to it, see
[Troubleshooting](#hyve-forecast-crps-stations-or-grids-cannot-be-matched).

### `hyve-forecast-crps`: memory

By default every input file is read whole. For large domains, read them lazily
in chunks over the spatial dimensions and fix the number of threads:

```shell
hyve-forecast-crps --reanalysis reanalysis.nc --reforecast forecasts/ \
    --climatology climatology.nc --spatial_dims point \
    --chunk_size 5000 --threads 4
```

`--chunk_size` is the number of spatial points per chunk; with several spatial
dimensions the first one is split into rows of whole slices of the others. The
climatology is also chunked by day of year, so a forecast reads only the days it
is scored against. Peak memory is close to the data held in memory plus
threads x 8 x chunk bytes, where a chunk holds every ensemble member and time
step of its points counted in float64 (the CRPS is computed in float64 even for
float32 files): 51 members, 40 steps and 5,000 points make 82 MB, so 4 threads
need about 2.6 GB on top of the outputs. The scores do not depend on
the chunk size. `--scheduler synchronous` runs everything in one thread, for
debugging.

### `hyve-forecast-crps`: configuration file

The settings can also come from a YAML file given with `--config`; flags of
the same name override it. Its `structure` key names the input structure, and
the command must match it: `reforecast` (the default without a subcommand) is
scored without a subcommand, `medium_range` with the `score` and `aggregate`
subcommands below. A mismatch, or a key that does not apply to the structure,
is an error.

```yaml
structure: reforecast         # default without a subcommand
variable: dis                 # in the reanalysis and reforecast files
spatial_dims: [point]         # default: [station]
coord_tolerance: 1.0e-6       # optional, for float coordinates
chunk_size: 5000              # optional, see the memory section
threads: 4                    # optional
scheduler: threads            # or synchronous
with_init: false              # as --with_init
climatology: climatology.nc   # optional benchmark
```

```shell
hyve-forecast-crps --config reforecast.yaml \
    --reanalysis reanalysis.nc --reforecast forecasts/ --output scores/
```

### `hyve-forecast-crps`: medium-range forecasts

Medium-range forecasts have one forecast per start date. They are scored in two
steps: `score` computes the CRPS of each forecast per lead time, as soon as the
reanalysis covers its longest lead time, and `aggregate` averages those scores
over a window of dates, such as a month.

```shell
hyve-forecast-crps score --config medium_range.yaml \
    --forecast forecasts/2025100100.nc --reanalysis reanalysis.nc --output scores/
hyve-forecast-crps aggregate --config medium_range.yaml \
    --scores scores/ --start 2025-10-01 --end 2025-10-31 --output crps_202510.nc
```

Without a subcommand, `hyve-forecast-crps` scores reforecasts as described above.

The configuration file holds the settings shared by both steps. Every key is
optional; flags of the same name override the file. A step ignores the keys
marked for the other one; `aggregate` also ignores `coord_tolerance`, since
every score file of a window must be on the same grid.

```yaml
structure: medium_range       # the default for score and aggregate
variable: dis                 # in the forecast, reanalysis and climatology
spatial_dims: [point]         # default: [station]
lead_time_dim: step           # or time (the default), see below; for score
ensemble_dim: number          # default: ensemble; for score
coord_tolerance: 1.0e-6       # optional, for float coordinates
chunk_size: 5000              # optional, see the memory section
threads: 4                    # optional
scheduler: threads            # or synchronous
climatology: climatology.nc   # optional benchmark, for score
aggregation:                  # for aggregate
  by: valid_time              # or start_time
  min_count: 1                # cells with fewer samples are NaN
  start_frequency: 1D         # time between start times: 1D, 12h, 6h, ...
  allow_missing: false        # true: aggregate despite missing start times
```

Inputs of `score`:

| Input       | `variable` dimensions                            | Also required |
| ----------- | ------------------------------------------------ | ------------- |
| Forecast    | *ensemble dim*, *lead-time dim*, *spatial dims*  | scalar coordinate `forecast_reference_time` (the start time) |
| Reanalysis  | `time`, *spatial dims*                           | |
| Climatology | *ensemble dim*, `doy`, `issue_hour`, *spatial dims* | as written by `hyve-clim-benchmark` |

The ensemble dimension is `ensemble_dim` (`ensemble` by default, `number` in
GRIB-derived files). The lead-time dimension depends on `lead_time_dim`:

- `time` (default): `time` holds the valid times.
- `step`: `step` holds the lead times (time differences) from
  `forecast_reference_time`, as cfgrib and earthkit write them. The valid time
  is `forecast_reference_time + step`; a `valid_time` coordinate, if present,
  is ignored.

Forecast values are taken as instantaneous at their valid time and verified
against the reanalysis at that time. Every valid time must therefore be a
reanalysis time: a forecast with 3-hourly steps against a 6-hourly reanalysis
fails, naming the unmatched times, so select the 6-hourly steps first (with
`lead_time_dim: step`, for example
`ds.sel(step=ds.step % np.timedelta64(6, "h") == np.timedelta64(0, "h"))`).

`--forecast` is one file or a folder of them. Each forecast file gives one
`scores_YYYYMMDDHH.nc` (named by its start time) with `crps_forecast`,
`crps_persistence` and, with a climatology, `crps_climatology`, in float32 on
(`step`, *spatial dims*), with the coordinates `valid_time(step)` and
`forecast_reference_time`.

- Step 0 (the initial condition) is not scored.
- Persistence is the reanalysis at the start time.
- The climatology is taken for each valid time: the day of year of the valid
  date, and the `issue_hour` slot holding the valid hour (the largest one not
  after it), so a daily climatology (`issue_hour` 0 only) serves 6-hourly steps.
  The climatology must cover every forecast point.
- The forecast grid is the reference, as for reforecasts. Missing data gives NaN
  in that cell only; a point without forecast data is NaN in every score. The
  step fails only if the reanalysis has no value at any scored point for a
  valid time, or for the start time.

`aggregate` reads the `scores_*.nc` files of a folder and keeps the samples
(start date, step) whose valid time, or start time with `by: start_time`, falls
on a date from `--start` to `--end`, both included. The window follows calendar
dates: a value valid at 1 November 00 UTC counts in November. For a monthly
window by valid time, the folder must hold the scores of the forecasts started
up to the longest lead time before the month.

The window must be complete: every start time on the `start_frequency` grid
(from 00 UTC) that has a sample in the window needs its score file. By valid
time these run from the window start minus the longest lead time to the window
end minus the shortest one, taking the lead times from the score files. A
missing start time is an error naming it; with `allow_missing: true`
(`--allow_missing`, or `--no-allow_missing` to override the file) it is a warning and the means use the files present. The
output records `expected_start_times` and `missing_start_times` as attributes.
Score files of extra start times, off the grid, are used as well. Two score
files with the same start time are an error, so a rerun saved under another
name is not counted twice.

Each cell is averaged over the
samples where every score is finite, so forecast and benchmarks always share
the same samples. The output file has, on (`step`, *spatial dims*):

| Variable            | Meaning |
| ------------------- | ------- |
| `crps_forecast`, `crps_persistence`, `crps_climatology` | mean CRPS |
| `crpss_persistence`, `crpss_climatology` | `1 - crps_forecast / benchmark`, NaN where the benchmark is 0 |
| `count`             | number of samples in each mean |

Cells with fewer than `min_count` samples are NaN. Every score file of a window
must be on the same grid. Both steps write each file atomically, so a failed
write leaves no partial file and keeps any earlier file of that name.

## Troubleshooting

### `hyve-forecast-crps`: stations or grids cannot be matched

`hyve-forecast-crps` matches the reanalysis, the reforecast files and the
(optional) climatology by their station IDs or grid coordinates. The reforecast
grid is the reference: reanalysis and climatology are relabelled to its
coordinate values (and output files carry them), and only points present in
both are scored. The latest reforecast file (by sorted name) defines the grid
and every other file must cover it. A mismatch exits with status 1 and a
one-line `Error: ...` message on stderr. The message names the dimension, both
datasets and, for float grids, the tolerance. There are three kinds:

- *Station mismatch* (`StationMismatchError`): station IDs of different kinds
  (integer vs text vs bytes, never converted), no ID in common, or a reforecast
  file lacking a reference station, or duplicate IDs. Fix the ID type or the
  duplicates upstream and rerun.
- *Grid mismatch* (`GridMismatchError`): the same problems for `lat`, `lon`,
  `y`, `x`, plus ambiguous tolerances, NaN/infinite coordinates and duplicate
  coordinates (of any type, integer or float).
- *Dimension mismatch* (`DimensionMismatchError`): a dataset has no `lat` (or
  other `--spatial_dims`) dimension, or the reanalysis has a dimension other
  than `time` and the spatial dimensions. Set `--spatial_dims` (or `--core_dim`) to
  the dimensions the files actually have.

Float coordinates (`lat`, `lon`, projected metres) match within a tolerance
relative to their precision, 4 x the machine epsilon of the coarser dtype, so a
float32 grid matches the same float64 grid. Integer coordinates (station IDs,
integer `y`/`x`) match exactly. If grids differ by more (for example a
rounded-coordinate file), pass an absolute tolerance in coordinate units:
`--coord_tolerance 0.001` (finite, >= 0).

*Ambiguous* means the tolerance is at least half the smallest grid spacing (or
two values snap to one point): a value could match two cells, so nothing is
matched. Lower `--coord_tolerance` or remove it, and check the coordinates.
The old name `hyve-crps-stations` still works but is deprecated.

## License
### License headers

Every file declares its copyright and license with [SPDX](https://spdx.dev) tags following
the [REUSE](https://reuse.software) specification, as set out in ECMWF's
[ADR-010](https://github.com/ecmwf/codex/blob/main/ADR/ADR-010-SPDX-License-Identifiers-and-REUSE-Headers.md).
Add the header to a new file with:

```
uvx reuse annotate \
    --copyright "European Centre for Medium-Range Weather Forecasts (ECMWF)" \
    --license Apache-2.0 \
    path/to/new_file.py
```

which writes, in the file's comment style and with the current year:

```
# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0
```

- `uvx` runs `reuse` without installing it; `pip install reuse` works too.
- If `reuse` does not recognise the file type, pick a comment style with `--style`, e.g. `--style python`.
- For a co-developed file, pass `--copyright` once per copyright holder.
- Files that cannot carry a comment (data, generated files) are covered in [REUSE.toml](REUSE.toml) instead.

The `reuse` pre-commit hook checks that every file is covered; run the same check by hand
with `uvx reuse lint`.

## License

Hyve is licensed under the Apache License, Version 2.0; see [LICENSE](LICENSE) and
[NOTICE](NOTICE), which also carry ECMWF's intergovernmental notice.

© 2025-2026 European Centre for Medium-Range Weather Forecasts (ECMWF).
