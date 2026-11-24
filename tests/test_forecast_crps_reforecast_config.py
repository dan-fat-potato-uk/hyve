# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration file of the reforecast route of hyve-forecast-crps.

Without a subcommand, hyve-forecast-crps scores reforecasts; ``--config``
gives the same settings as the flags, which override it.
"""

import dask
import pytest
import xarray as xr

# the reforecast inputs of the chunking tests
from test_forecast_crps_chunks import _write_inputs

from hyve.tools import forecast_crps
from hyve.tools.forecast_crps import cli


def _main(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["hyve-forecast-crps", *argv])
    forecast_crps.main()


def _outputs(folder):
    return {p.name: xr.open_dataarray(p) for p in sorted(folder.iterdir())}


def _assert_same_outputs(a, b):
    expected, actual = _outputs(a), _outputs(b)
    assert sorted(actual) == sorted(expected)
    assert "crps_clim_20200226.nc" in expected
    for name in expected:
        xr.testing.assert_allclose(actual[name], expected[name], rtol=1e-12, atol=0)


def _rename_variable(path, name):
    """Rewrite the netCDF file `path` with its variable ``dis`` called `name`."""
    ds = xr.load_dataset(path)
    ds.rename(dis=name).to_netcdf(path)


def test_config_file_gives_the_scores_of_the_flags(monkeypatch, tmp_path):
    reanalysis, forecasts, climatology = _write_inputs(tmp_path)
    _main(
        monkeypatch,
        "--reanalysis", str(reanalysis),
        "--reforecast", str(forecasts),
        "--climatology", str(climatology),
        "--spatial_dims", "point",
        "--chunk_size", "5",
        "--output", str(tmp_path / "flags"),
    )  # fmt: skip

    config = tmp_path / "config.yaml"
    config.write_text(
        "structure: reforecast\n"
        "spatial_dims: [point]\n"
        "chunk_size: 5\n"
        f"climatology: {climatology}\n"
    )
    _main(
        monkeypatch,
        "--config", str(config),
        "--reanalysis", str(reanalysis),
        "--reforecast", str(forecasts),
        "--output", str(tmp_path / "config"),
    )  # fmt: skip
    _assert_same_outputs(tmp_path / "flags", tmp_path / "config")


def test_configured_variable(monkeypatch, tmp_path):
    reanalysis, forecasts, climatology = _write_inputs(tmp_path / "dis")
    args = ["--climatology", str(climatology), "--spatial_dims", "point"]
    _main(
        monkeypatch,
        "--reanalysis", str(reanalysis),
        "--reforecast", str(forecasts),
        "--output", str(tmp_path / "dis_out"),
        *args,
    )  # fmt: skip

    reanalysis, forecasts, climatology = _write_inputs(tmp_path / "q")
    _rename_variable(reanalysis, "q")
    for path in forecasts.iterdir():
        _rename_variable(path, "q")
    config = tmp_path / "config.yaml"
    config.write_text("variable: q\n")
    _main(
        monkeypatch,
        "--config", str(config),
        "--reanalysis", str(reanalysis),
        "--reforecast", str(forecasts),
        "--output", str(tmp_path / "q_out"),
        *args,
    )  # fmt: skip
    _assert_same_outputs(tmp_path / "dis_out", tmp_path / "q_out")


def test_flags_override_the_config_file(monkeypatch, tmp_path):
    reanalysis, forecasts, _ = _write_inputs(tmp_path)
    seen = {}

    def fake_compute_score(out, folder, da, clim, spatial_dims, with_init, *a, **kw):
        seen.update(
            scheduler=dask.config.get("scheduler"),
            num_workers=dask.config.get("num_workers", None),
            spatial_dims=spatial_dims,
            with_init=with_init,
            chunk_size=kw["chunk_size"],
        )

    monkeypatch.setattr(cli, "compute_score", fake_compute_score)
    config = tmp_path / "config.yaml"
    config.write_text(
        "spatial_dims: [station]\n"
        "with_init: true\n"
        "threads: 3\n"
        "chunk_size: 10\n"
        "scheduler: synchronous\n"
    )
    _main(
        monkeypatch,
        "--config", str(config),
        "--reanalysis", str(reanalysis),
        "--reforecast", str(forecasts),
        "--spatial_dims", "point",
        "--threads", "2",
        "--output", str(tmp_path / "out"),
    )  # fmt: skip
    assert seen == {
        "scheduler": "synchronous",
        "num_workers": 2,
        "spatial_dims": ["point"],
        "with_init": True,
        "chunk_size": 10,
    }


def test_core_dim_overrides_the_config_file(monkeypatch, tmp_path):
    reanalysis, forecasts, _ = _write_inputs(tmp_path)
    seen = {}
    monkeypatch.setattr(
        cli, "compute_score", lambda *a, **kw: seen.update(spatial_dims=a[4])
    )
    config = tmp_path / "config.yaml"
    config.write_text("spatial_dims: [station]\n")
    _main(
        monkeypatch,
        "--config", str(config),
        "--reanalysis", str(reanalysis),
        "--reforecast", str(forecasts),
        "--core_dim", "point",
        "--output", str(tmp_path / "out"),
    )  # fmt: skip
    assert seen == {"spatial_dims": ["point"]}


@pytest.mark.parametrize(
    "text, message",
    [
        ("structure: medium_range\n", "'score' and 'aggregate' subcommands"),
        ("lead_time_dim: step\n", "lead_time_dim"),
        ("not_a_key: 1\n", "not_a_key"),
        ("spatial_dims: [point]\nvariable: q\n", "no variable 'q'"),
    ],
)
def test_invalid_configuration_exits_1(monkeypatch, tmp_path, capsys, text, message):
    reanalysis, forecasts, _ = _write_inputs(tmp_path)
    config = tmp_path / "config.yaml"
    config.write_text(text)
    with pytest.raises(SystemExit) as exc:
        _main(
            monkeypatch,
            "--config", str(config),
            "--reanalysis", str(reanalysis),
            "--reforecast", str(forecasts),
            "--output", str(tmp_path / "out"),
        )  # fmt: skip
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("Error: ")
    assert message in err
    assert not (tmp_path / "out").exists()


def test_every_reforecast_file_needs_the_variable(monkeypatch, tmp_path, capsys):
    reanalysis, forecasts, _ = _write_inputs(tmp_path)
    oldest = min(forecasts.iterdir())  # not the reference file
    _rename_variable(oldest, "q")
    with pytest.raises(SystemExit) as exc:
        _main(
            monkeypatch,
            "--reanalysis", str(reanalysis),
            "--reforecast", str(forecasts),
            "--spatial_dims", "point",
            "--output", str(tmp_path / "out"),
        )  # fmt: skip
    assert exc.value.code == 1
    assert f"{oldest.name} has no variable 'dis'" in capsys.readouterr().err


def test_help_points_to_the_medium_range_subcommands(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _main(monkeypatch, "--help")
    out = " ".join(capsys.readouterr().out.split())
    assert "--config" in out
    assert "'score' and 'aggregate' subcommands" in out
