from __future__ import annotations

import unittest
from unittest.mock import patch

from continuum.profiler import main as profile_main


class TestProfileMain(unittest.TestCase):
    def _assert_exit_code(self, exc: BaseException, expected: int) -> None:
        code = getattr(exc, "code", None)
        if code is None:
            code = getattr(exc, "exit_code", None)
        if code is None and isinstance(exc, SystemExit):
            code = exc.code
        self.assertEqual(code, expected)

    def test_benchmarks_cpu_only_and_exclude_static(self) -> None:
        mock_static = unittest.mock.Mock(return_value={"cpu": {"model": "X"}})
        mock_cpu = unittest.mock.Mock(return_value={"cpu_sustained": {"mean_iter_per_sec": 1.0}})
        registry = {"static": mock_static, "cpu": mock_cpu}
        with patch("continuum.profiler.main.AVAILABLE_BENCHMARKS", registry):
            with patch("continuum.profiler.main.render_profile_human"):
                with patch("continuum.profiler.main.typer.echo"):
                    with patch(
                        "continuum.profiler.main.build_profile_report",
                        side_effect=lambda static_profile, benchmark_results=None, benchmarks=None: {
                            "schema_version": "1.0.0",
                            "static_profile": static_profile,
                            "benchmarks": benchmarks or {},
                            "benchmark_results": benchmark_results or [],
                        },
                    ) as mock_build:
                        with self.assertRaises(BaseException) as cm:
                            profile_main.profile_command(
                                benchmarks="cpu",
                                no_static=True,
                                output_format="json",
                                no_write=True,
                            )
        self._assert_exit_code(cm.exception, 0)
        mock_static.assert_not_called()
        args, kwargs = mock_build.call_args
        self.assertEqual(args[0], {})
        self.assertIn("cpu_sustained", kwargs["benchmarks"])

    def test_no_static_produces_empty_static_profile(self) -> None:
        mock_static = unittest.mock.Mock(return_value={"cpu": {"model": "X"}})
        mock_cpu = unittest.mock.Mock(return_value={"cpu_sustained": {"iterations": 1}})
        mock_mem = unittest.mock.Mock(return_value={"memory_bandwidth": {"iterations": 1}})
        registry = {"static": mock_static, "cpu": mock_cpu, "memory": mock_mem}
        with patch("continuum.profiler.main.AVAILABLE_BENCHMARKS", registry):
            with patch("continuum.profiler.main.render_profile_human"):
                with patch("continuum.profiler.main.typer.echo"):
                    with patch(
                        "continuum.profiler.main.build_profile_report",
                        side_effect=lambda static_profile, benchmark_results=None, benchmarks=None: {
                            "schema_version": "1.0.0",
                            "static_profile": static_profile,
                            "benchmarks": benchmarks or {},
                            "benchmark_results": benchmark_results or [],
                        },
                    ) as mock_build:
                        with self.assertRaises(BaseException) as cm:
                            profile_main.profile_command(
                                no_static=True,
                                output_format="json",
                                no_write=True,
                            )
        self._assert_exit_code(cm.exception, 0)
        args, _kwargs = mock_build.call_args
        self.assertEqual(args[0], {})

    def test_unknown_benchmark_exits_with_code_2(self) -> None:
        with patch("continuum.profiler.main.typer.echo"):
            with self.assertRaises(BaseException) as cm:
                profile_main.profile_command(
                    benchmarks="static,unknown",
                    no_write=True,
                )
        self._assert_exit_code(cm.exception, 2)

    def test_output_format_json_suppresses_human_output(self) -> None:
        mock_static = unittest.mock.Mock(return_value={"cpu": {"model": "X"}})
        mock_cpu = unittest.mock.Mock(return_value={"cpu_sustained": {"iterations": 1}})
        mock_mem = unittest.mock.Mock(return_value={"memory_bandwidth": {"iterations": 1}})
        registry = {"static": mock_static, "cpu": mock_cpu, "memory": mock_mem}
        with patch("continuum.profiler.main.AVAILABLE_BENCHMARKS", registry):
            with patch("continuum.profiler.main.render_profile_human") as mock_render:
                with patch("continuum.profiler.main.typer.echo"):
                    with self.assertRaises(BaseException) as cm:
                        profile_main.profile_command(
                            output_format="json",
                            no_write=True,
                        )
        self._assert_exit_code(cm.exception, 0)
        mock_render.assert_not_called()

    def test_default_behavior_runs_all(self) -> None:
        mock_static = unittest.mock.Mock(return_value={"cpu": {"model": "X"}})
        mock_cpu = unittest.mock.Mock(return_value={"cpu_sustained": {"iterations": 1}})
        mock_mem = unittest.mock.Mock(return_value={"memory_bandwidth": {"iterations": 1}})
        registry = {"static": mock_static, "cpu": mock_cpu, "memory": mock_mem}
        with patch("continuum.profiler.main.AVAILABLE_BENCHMARKS", registry):
            with patch("continuum.profiler.main.render_profile_human") as mock_render:
                with patch("continuum.profiler.main.typer.echo"):
                    with self.assertRaises(BaseException) as cm:
                        profile_main.profile_command(
                            no_write=True,
                        )
        self._assert_exit_code(cm.exception, 0)
        mock_static.assert_called_once()
        mock_cpu.assert_called_once()
        mock_mem.assert_called_once()
        mock_render.assert_called_once()


if __name__ == "__main__":
    unittest.main()
