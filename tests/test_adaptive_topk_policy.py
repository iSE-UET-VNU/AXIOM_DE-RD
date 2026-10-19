import csv
import json
import tempfile
import unittest
from pathlib import Path

from ondemand.adaptive.topk_policy import apply_policy, build_policy, load_policy, predictions_from_csv
from ondemand.adaptive.pipeline import subdata_for


class AdaptiveTopKPolicyTest(unittest.TestCase):
    def test_committed_policy_matches_reproducible_oof_artifact(self):
        path = Path(__file__).resolve().parents[1] / "ondemand/adaptive/xgboost_score_modality_oof_20261016.json"
        policy = load_policy(path)
        self.assertEqual(policy["n_queries"], 220)
        self.assertAlmostEqual(policy["mean_k"], 59.43181818181818)
        self.assertAlmostEqual(policy["metrics"]["macro_recall"], 0.8522584932359281)

    def test_csv_export_and_apply_prefers_top200(self):
        with tempfile.TemporaryDirectory() as folder:
            csv_path = Path(folder) / "oof.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["qid", "chosen_k"])
                writer.writeheader()
                writer.writerows([{"qid": "q1", "chosen_k": 120}, {"qid": "q2", "chosen_k": 7}])
            predictions = predictions_from_csv(csv_path)
            policy = build_policy(predictions, name="test")
            gate = {
                "gate": {"q1": ["p1"], "q2": ["p2"]},
                "light_run_top100": {"q1": list(range(100)), "q2": list(range(100))},
                "light_run_top200": {"q1": list(range(150)), "q2": list(range(150))},
            }
            output = apply_policy(gate, policy)
            self.assertEqual(output["adaptive_k"], {"q1": 120, "q2": 7})
            self.assertEqual(output["adaptive_k_policy"]["ranking_fields"], {"light_run_top200": 2})

    def test_rejects_silent_cap(self):
        policy = build_policy({"q1": 120}, name="test")
        gate = {"gate": {"q1": ["p1"]}, "light_run_top100": {"q1": list(range(100))}}
        with self.assertRaisesRegex(ValueError, "rebuild a top-200 gate"):
            apply_policy(gate, policy)
        output = apply_policy(gate, policy, allow_cap=True)
        self.assertEqual(output["adaptive_k"]["q1"], 100)
        self.assertEqual(output["adaptive_k_policy"]["capped_queries"]["q1"]["requested_k"], 120)

    def test_policy_schema_is_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy.json"
            path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported policy schema"):
                load_policy(path)

    def test_pipeline_uses_deep_ranking_and_aligned_scores(self):
        gate = {
            "gate": {"q1": ["p1"]},
            "gate_scores": {"q1": [0.9]},
            "light_run_top100": {"q1": ["p1", "p2"]},
            "light_run_top200": {"q1": ["p1", "p2", "p3"]},
            "light_run_scores_top200": {"q1": [0.9, 0.8, 0.7]},
        }
        subdata = subdata_for(gate, "q1", "question")
        self.assertEqual(subdata.pages, ["p1", "p2", "p3"])
        self.assertEqual(subdata.scores, {"p1": 0.9, "p2": 0.8, "p3": 0.7})


if __name__ == "__main__":
    unittest.main()
