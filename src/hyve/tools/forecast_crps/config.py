# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration model of hyve-forecast-crps."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Aggregation(StrictModel):
    """How `aggregate` averages per-start-date scores over a window."""

    by: Literal["valid_time", "start_time"] = Field(
        default="valid_time",
        description=(
            "a (start date, step) sample is in the window when its valid time"
            " (or its start time) falls on a date of the window"
        ),
    )
    min_count: int = Field(
        default=1,
        ge=1,
        description="cells with fewer samples than this are NaN",
    )
    start_frequency: str = Field(
        default="1D",
        description=(
            "time between forecast start times, a divisor of a day such as"
            " '1D', '12h' or '6h'; start times are on this grid from 00 UTC"
        ),
    )
    allow_missing: bool = Field(
        default=False,
        description=(
            "aggregate without the score files of missing start times, with a"
            " warning; by default a missing start time is an error"
        ),
    )

    @field_validator("start_frequency")
    @classmethod
    def _divides_a_day(cls, v: str) -> str:
        try:
            freq = pd.Timedelta(v)
        except ValueError:
            raise ValueError(f"start_frequency {v!r} is not a duration") from None
        day = pd.Timedelta(days=1)
        if freq <= pd.Timedelta(0) or day % freq != pd.Timedelta(0):
            raise ValueError(
                f"start_frequency must divide a day (e.g. '1D', '12h', '6h'), got {v!r}"
            )
        return v

    @property
    def start_step(self) -> pd.Timedelta:
        """`start_frequency` as a time difference."""
        return pd.Timedelta(self.start_frequency)


class ForecastCrpsConfig(StrictModel):
    """Settings of hyve-forecast-crps, from a YAML file and the command line."""

    structure: Literal["reforecast", "medium_range"] = Field(
        default="reforecast",
        description=(
            "reforecast: a folder of files averaged by lead position;"
            " medium_range: one forecast per start date, scored per lead time"
            " and aggregated over a window"
        ),
    )
    variable: str = Field(
        default="dis",
        description=(
            "variable in the forecast and reanalysis files, and in the"
            " climatology of the medium_range structure (a reforecast"
            " climatology is read as its single variable)"
        ),
    )
    spatial_dims: list[str] = Field(default_factory=lambda: ["station"], min_length=1)
    lead_time_dim: Literal["time", "step"] = Field(
        default="time",
        description=(
            "lead-time dimension of a medium-range forecast. time: it holds"
            " valid times; step: it holds lead times from forecast_reference_time"
        ),
    )
    ensemble_dim: str = Field(
        default="ensemble",
        min_length=1,
        description="ensemble dimension of a medium-range forecast, e.g. 'number'",
    )
    coord_tolerance: float | None = Field(default=None, ge=0)
    chunk_size: int | None = Field(default=None, ge=1)
    threads: int | None = Field(default=None, ge=1)
    scheduler: Literal["threads", "synchronous"] = "threads"
    with_init: bool = False
    climatology: Path | None = None
    aggregation: Aggregation = Field(default_factory=Aggregation)

    @field_validator("coord_tolerance")
    @classmethod
    def _finite_tolerance(cls, v: float | None) -> float | None:
        if v is not None and not math.isfinite(v):
            raise ValueError(f"coord_tolerance must be finite, got {v}")
        return v

    @model_validator(mode="after")
    def _structure_options(self) -> ForecastCrpsConfig:
        if self.structure == "medium_range" and self.with_init:
            raise ValueError(
                "with_init applies to the reforecast structure only; medium-range"
                " forecasts carry their start time in forecast_reference_time"
            )
        if self.structure == "reforecast":
            for key in ("aggregation", "lead_time_dim", "ensemble_dim"):
                if key in self.model_fields_set:
                    raise ValueError(
                        f"{key} applies to the medium_range structure only"
                    )
        return self


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """`base` updated with the values of `overrides` that are not None.

    Nested mappings (``aggregation``) are merged key by key.
    """
    merged = dict(base)
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key, {}), dict):
            nested = _merge(merged.get(key, {}), value)
            if nested:
                merged[key] = nested
        else:
            merged[key] = value
    return merged


def load_config(
    path: str | Path | None = None,
    defaults: dict[str, Any] | None = None,
    **overrides: Any,
) -> ForecastCrpsConfig:
    """Configuration from `defaults`, then the YAML file at `path`, then `overrides`.

    Overrides that are None (command line flags not given) are ignored.
    """
    data = dict(defaults or {})
    if path is not None:
        with open(path) as f:
            data = _merge(data, yaml.safe_load(f) or {})
    return ForecastCrpsConfig(**_merge(data, overrides))
