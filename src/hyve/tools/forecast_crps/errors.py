# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

"""Errors raised by the forecast CRPS tool.

All of them are ``ValueError`` subclasses of `CoordinateMismatchError`, which
the command line interface reports as a one-line ``Error: ...`` message.
"""


class CoordinateMismatchError(ValueError):
    """Coordinates or dimensions of two datasets cannot be matched."""


class StationMismatchError(CoordinateMismatchError):
    """Station IDs of two datasets cannot be matched."""


class GridMismatchError(CoordinateMismatchError):
    """Values of a grid (non-station) dimension cannot be matched.

    For example: no overlap, an ambiguous tolerance, two values snapping to
    one point, duplicate or non-finite coordinates, or a reforecast file that
    lacks a point of the reference grid.
    """


class DimensionMismatchError(CoordinateMismatchError):
    """A dataset lacks a required dimension or has an unexpected one."""


def mismatch_error_class(dim):
    """Error class for a mismatch along `dim`: station IDs or a grid dimension."""
    return StationMismatchError if dim == "station" else GridMismatchError


class InputContractError(ValueError):
    """An input does not follow the layout the tool expects.

    For example a medium-range forecast without ``forecast_reference_time``,
    or a climatology without an ``issue_hour`` slot for a valid time.
    """


class MissingDataError(ValueError):
    """Data needed for a score is missing.

    For example the reanalysis has no value at all for a lead time, or no
    per-start-date score file falls in an aggregation window.
    """
