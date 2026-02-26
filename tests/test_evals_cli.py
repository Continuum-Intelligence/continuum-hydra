from __future__ import annotations

import json
import os
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

if find_spec("typer") is not None:
    from typer.testing import CliRunner

    from continuum.cli import app
else:  # pragma: no cover
    CliRunner = None
    app = None


@unittest.skipIf(find_spec("typer") is None, "typer is not installed in this interpreter")
class TestEvalsCli(unittest.TestCase):
    def test_evals_init_creates_core_suite(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                result = runner.invoke(app, ["evals", "init"], catch_exceptions=False)
                self.assertEqual(result.exit_code, 0)
                suite_path = Path(tmp) / "hydra-evals" / "suites" / "core.yaml"
                self.assertTrue(suite_path.exists())
                data = json.loads(suite_path.read_text(encoding="utf-8"))
                self.assertEqual(data["schema_version"], "evals.suite.v1")
            finally:
                os.chdir(previous)

    def test_evals_run_dummy_creates_artifacts_and_metrics(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                suites_dir = Path(tmp) / "hydra-evals" / "suites"
                suites_dir.mkdir(parents=True, exist_ok=True)
                suite_payload = {
                    "schema_version": "evals.suite.v1",
                    "name": "core",
                    "model_prompt_template": "Sources:\\n{sources}\\n\\nQuestion: {question}\\nAnswer:",
                    "cases": [
                        {
                            "case_id": "pass-case",
                            "question": "When was Hydra founded?",
                            "sources": [{"id": "S1", "text": "Hydra was founded in 2024."}],
                            "must_cite": True,
                            "must_be_grounded": True,
                            "metadata": {"dummy_output": "Hydra was founded in 2024. [S1]"},
                        },
                        {
                            "case_id": "fail-case",
                            "question": "What year was Hydra founded?",
                            "sources": [{"id": "S1", "text": "Hydra was founded in 2024."}],
                            "must_cite": True,
                            "must_be_grounded": True,
                            "metadata": {"dummy_output": "Hydra was founded in 2025. [S1]"},
                        },
                    ],
                }
                (suites_dir / "core.yaml").write_text(
                    json.dumps(suite_payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

                result = runner.invoke(
                    app,
                    ["evals", "run", "--suite", "core", "--model", "dummy", "--json"],
                    catch_exceptions=False,
                )
                self.assertEqual(result.exit_code, 0)
                summary = json.loads(result.stdout)
                self.assertEqual(summary["schema_version"], "evals.run.v1")
                self.assertEqual(summary["total_cases"], 2)
                self.assertEqual(summary["pass_count"], 1)
                self.assertAlmostEqual(summary["pass_rate"], 0.5)
                self.assertAlmostEqual(summary["citation_present_rate"], 1.0)
                self.assertAlmostEqual(summary["grounded_rate"], 0.5)
                self.assertEqual(summary["forbidden_hits"], 0)

                run_dir = Path(summary["run_path"])
                self.assertTrue((run_dir / "suite_snapshot.yaml").exists())
                self.assertTrue((run_dir / "results.jsonl").exists())
                self.assertTrue((run_dir / "summary.json").exists())
                self.assertTrue((run_dir / "report.md").exists())

                rows = [
                    json.loads(line)
                    for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.assertEqual([row["case_id"] for row in rows], ["fail-case", "pass-case"])
            finally:
                os.chdir(previous)

    def test_evals_run_fail_under_exits_3(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                suites_dir = Path(tmp) / "hydra-evals" / "suites"
                suites_dir.mkdir(parents=True, exist_ok=True)
                suite_payload = {
                    "schema_version": "evals.suite.v1",
                    "name": "core",
                    "model_prompt_template": "Sources:\\n{sources}\\n\\nQuestion: {question}\\nAnswer:",
                    "cases": [
                        {
                            "case_id": "only-case",
                            "question": "Q?",
                            "sources": [],
                            "must_cite": False,
                            "must_say_idk_if_insufficient": True,
                            "must_be_grounded": False,
                            "metadata": {"dummy_output": "No answer available."},
                        }
                    ],
                }
                (suites_dir / "core.yaml").write_text(json.dumps(suite_payload) + "\n", encoding="utf-8")

                result = runner.invoke(
                    app,
                    ["evals", "run", "--suite", "core", "--model", "dummy", "--fail-under", "0.9"],
                    catch_exceptions=False,
                )
                self.assertEqual(result.exit_code, 3)
            finally:
                os.chdir(previous)

    def test_evals_compare_reports_flips(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                suites_dir = Path(tmp) / "hydra-evals" / "suites"
                suites_dir.mkdir(parents=True, exist_ok=True)
                base_suite = {
                    "schema_version": "evals.suite.v1",
                    "name": "core",
                    "model_prompt_template": "Sources:\\n{sources}\\n\\nQuestion: {question}\\nAnswer:",
                    "cases": [
                        {
                            "case_id": "flip-case",
                            "question": "When?",
                            "sources": [{"id": "S1", "text": "Founded in 2024."}],
                            "must_cite": True,
                            "must_be_grounded": True,
                            "metadata": {"dummy_output": "Founded in 2024. [S1]"},
                        }
                    ],
                }
                suite_file = suites_dir / "core.yaml"
                suite_file.write_text(json.dumps(base_suite) + "\n", encoding="utf-8")

                left = runner.invoke(
                    app,
                    ["evals", "run", "--suite", "core", "--model", "dummy", "--json"],
                    catch_exceptions=False,
                )
                self.assertEqual(left.exit_code, 0)
                left_summary = json.loads(left.stdout)

                base_suite["cases"][0]["metadata"]["dummy_output"] = "Founded in 2025. [S1]"
                suite_file.write_text(json.dumps(base_suite) + "\n", encoding="utf-8")

                right = runner.invoke(
                    app,
                    ["evals", "run", "--suite", "core", "--model", "dummy", "--json"],
                    catch_exceptions=False,
                )
                self.assertEqual(right.exit_code, 0)
                right_summary = json.loads(right.stdout)

                compare = runner.invoke(
                    app,
                    ["evals", "compare", "--left", left_summary["run_id"], "--right", right_summary["run_id"]],
                    catch_exceptions=False,
                )
                self.assertEqual(compare.exit_code, 0)
                self.assertIn("pass -> fail", compare.output)
                self.assertIn("flip-case", compare.output)
                out_path = Path(tmp) / f"evals-compare-{left_summary['run_id']}-vs-{right_summary['run_id']}.md"
                self.assertTrue(out_path.exists())
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
