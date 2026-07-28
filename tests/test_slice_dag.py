import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "slice_dag", ROOT / "tools" / "slice_dag.py"
)
slice_dag = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = slice_dag
SPEC.loader.exec_module(slice_dag)

REWRITE_SPEC = importlib.util.spec_from_file_location(
    "rewrite_strict_mul", ROOT / "tools" / "rewrite_strict_mul.py"
)
rewrite_strict_mul = importlib.util.module_from_spec(REWRITE_SPEC)
assert REWRITE_SPEC.loader is not None
REWRITE_SPEC.loader.exec_module(rewrite_strict_mul)


class SliceDagParserTests(unittest.TestCase):
    def parse_fixture(self, body: str, *, pin_manifest: bool = False):
        source = (
            "def rhs_specie(state, mut ydot, X, z):\n"
            + "\n".join("    " + line for line in body.splitlines())
            + "\n\n"
        )
        return slice_dag.parse_function(
            source, "rhs_specie", pin_manifest=pin_manifest
        )

    def test_multiline_and_nonzero_suffix(self):
        dag = self.parse_fixture(
            """var T = state.T
var x1_7 = ((((
   T
))))
var x2_3 = x1_7 + 1
vset(ydot, 0, x2_3)"""
        )
        self.assertEqual([d.name for d in dag.definitions], ["x1_7", "x2_3"])
        self.assertEqual(dag.definitions[1].dependencies, ("x1_7",))
        self.assertEqual(dag.outputs[0].key, (0,))

    def test_duplicate_definition_is_rejected(self):
        with self.assertRaisesRegex(slice_dag.DagError, "duplicate"):
            self.parse_fixture(
                """var T = state.T
var x0_0 = T
var x0_0 = T
vset(ydot, 0, x0_0)"""
            )

    def test_use_before_definition_is_rejected(self):
        with self.assertRaisesRegex(slice_dag.DagError, "use-before"):
            self.parse_fixture(
                """var T = state.T
var x0_0 = x1_0
var x1_0 = T
vset(ydot, 0, x0_0)"""
            )

    def test_unsupported_statement_is_rejected(self):
        with self.assertRaisesRegex(slice_dag.DagError, "unsupported"):
            self.parse_fixture("var T = state.T\nprint(T)")

    def test_manifest_drift_is_rejected(self):
        with self.assertRaisesRegex(slice_dag.DagError, "manifest drift"):
            self.parse_fixture(
                "var T = state.T\nvar x0_0 = T\nvset(ydot, 0, x0_0)",
                pin_manifest=True,
            )


class SliceDagGenerationTests(unittest.TestCase):
    def test_repository_manifest_and_deterministic_generation(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            report_a = slice_dag.generate(
                ROOT / "reproducer.mojo", Path(first)
            )
            report_b = slice_dag.generate(
                ROOT / "reproducer.mojo", Path(second)
            )
            self.assertEqual(report_a, report_b)
            for name in (
                "slices_jac.mojo",
                "slices_rhs.mojo",
                "slices_jac_shared.mojo",
                "slices_rhs_shared.mojo",
                "slices_base_shared.mojo",
                "slices_rhs_shared.cuh",
                "slices_base_shared.cuh",
                "slice_report.json",
            ):
                self.assertEqual(
                    (Path(first) / name).read_bytes(),
                    (Path(second) / name).read_bytes(),
                )
            self.assertLessEqual(report_a["exchange_bytes"], 8192)
            self.assertEqual(report_a["exchange_bytes"], 13 * 32 * 8)
            self.assertEqual(
                report_a["input_sha256"],
                hashlib.sha256((ROOT / "reproducer.mojo").read_bytes()).hexdigest(),
            )
            parsed = json.loads((Path(first) / "slice_report.json").read_text())
            self.assertEqual(parsed["functions"]["jac_nuc"]["outputs"], 225)
            self.assertEqual(
                parsed["shared_scratch"]["jacobian"]["scratch_slots"], 515
            )
            self.assertEqual(
                parsed["shared_scratch"]["rhs"]["scratch_slots"], 174
            )
            self.assertEqual(
                parsed["shared_scratch"]["rhs"]["exchange_slots"], 13
            )
            self.assertEqual(
                parsed["shared_scratch"]["base"]["scratch_slots"], 531
            )
            self.assertEqual(
                parsed["shared_scratch"]["base"]["exchange_slots"], 15
            )
            self.assertEqual(parsed["fused_base"]["definitions"], 1113)
            self.assertEqual(
                parsed["fused_base"]["eliminated_definitions"], 562
            )
            self.assertEqual(parsed["fused_base"]["outputs"], 240)
            self.assertEqual(
                parsed["shared_scratch"]["rhs"]["exchange_producers"][0][
                    "definitions"
                ],
                93,
            )
            self.assertIn(
                "def rhs_specie_exchange_shared(",
                (Path(first) / "slices_rhs_shared.mojo").read_text(),
            )
            self.assertIn(
                "def base_exchange_shared(",
                (Path(first) / "slices_base_shared.mojo").read_text(),
            )
            cuda_base = (Path(first) / "slices_base_shared.cuh").read_text()
            cuda_rhs = (Path(first) / "slices_rhs_shared.cuh").read_text()
            self.assertIn("namespace structured_cuda_base", cuda_base)
            self.assertIn("base_exchange_shared", cuda_base)
            self.assertIn("structured_mul", cuda_base)
            self.assertIn("inline constexpr int kConcurrentWarps = 3;", cuda_base)
            self.assertIn("namespace structured_cuda_rhs", cuda_rhs)
            self.assertIn("rhs_specie_exchange_shared", cuda_rhs)
            self.assertNotIn(" if ", cuda_base)
            for cuda_source in (cuda_base, cuda_rhs):
                self.assertNotIn("cell * ScratchSlots", cuda_source)
                self.assertNotIn("cell * ExchangeSlots", cuda_source)
            self.assertEqual(
                parsed["shared_scratch"]["jacobian"]["wave_warps"],
                [[7, 1, 0], [2, 3, 4, 5, 6]],
            )

    def test_bounded_regions_reduce_shared_scratch(self):
        with tempfile.TemporaryDirectory() as output:
            report = slice_dag.generate(
                ROOT / "reproducer.mojo",
                Path(output),
                shared_region_definitions=48,
            )
            self.assertEqual(
                report["shared_scratch"]["jacobian"]["scratch_slots"], 446
            )
            self.assertEqual(
                report["shared_scratch"]["rhs"]["scratch_slots"], 108
            )
            cuda_base = (Path(output) / "slices_base_shared.cuh").read_text()
            self.assertIn("__device__ __noinline__ void _base_slice_", cuda_base)

    def test_tuned_cuda_schedule_keeps_uniform_empty_exchange_signatures(self):
        with tempfile.TemporaryDirectory() as output:
            report = slice_dag.generate(
                ROOT / "reproducer.mojo",
                Path(output),
                threshold=1400,
                shared_region_definitions=416,
                shared_wave_warps=8,
            )
            base = report["shared_scratch"]["base"]
            rhs = report["shared_scratch"]["rhs"]
            self.assertEqual(base["scratch_slots"], 190)
            self.assertEqual(rhs["scratch_slots"], 0)
            self.assertEqual(base["exchange_slots"], 0)
            self.assertEqual(rhs["exchange_slots"], 0)
            self.assertEqual(base["concurrent_warps"], 8)
            self.assertEqual(rhs["concurrent_warps"], 8)
            mojo_base = (Path(output) / "slices_base_shared.mojo").read_text()
            cuda_base = (Path(output) / "slices_base_shared.cuh").read_text()
            self.assertIn("def base_exchange_shared(", mojo_base)
            self.assertIn("exchange: SharedPointer", mojo_base)
            self.assertIn("    pass", mojo_base)
            self.assertNotIn(
                "def base_exchange_shared(\n"
                "    inputs: SharedPointer, exchange: SharedPointer, \n"
                "    scratch: SharedPointer, cell: Int, z: Float64):\n"
                "    var T",
                mojo_base,
            )
            self.assertIn("base_exchange_shared", cuda_base)
            self.assertIn("double* __restrict__ exchange", cuda_base)
            self.assertIn("inline constexpr int kConcurrentWarps = 8;", cuda_base)

    def test_shared_warp_order_must_be_a_permutation(self):
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaisesRegex(slice_dag.DagError, "permutation"):
                slice_dag.generate(
                    ROOT / "reproducer.mojo",
                    Path(output),
                    shared_warp_order=(0, 1, 2, 3, 4, 5, 6, 6),
                )

    def test_exchange_budget_is_enforced(self):
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaisesRegex(slice_dag.DagError, "exchange arena"):
                slice_dag.generate(
                    ROOT / "reproducer.mojo",
                    Path(output),
                    threshold=0,
                    max_exchange_bytes=0,
                )


class StrictMultiplyRewriteTests(unittest.TestCase):
    def test_nested_products_are_rewritten_in_evaluation_order(self):
        rewritten = rewrite_strict_mul.rewrite_expression("a * b * (c + d * e)")
        self.assertEqual(
            rewritten,
            "portable_mul(portable_mul(a, b), (c + portable_mul(d, e)))",
        )

    def test_non_products_are_unchanged(self):
        rewritten = rewrite_strict_mul.rewrite_expression("a / b + c - d")
        self.assertEqual(rewritten, "(((a / b) + c) - d)")


class CudaExpressionTests(unittest.TestCase):
    def test_nested_conditionals_and_boolean_operators(self):
        translated = slice_dag.cuda_expression(
            "portable_mul(a, b) if truthy(x and not y) else portable_sqrt(z)"
        )
        self.assertEqual(
            translated,
            "(structured_truthy((x && (!y))) ? structured_mul(a, b) : structured_sqrt(z))",
        )


if __name__ == "__main__":
    unittest.main()
