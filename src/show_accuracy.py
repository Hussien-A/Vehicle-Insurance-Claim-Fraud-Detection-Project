"""
Show the saved model's performance on the held-out TEST split.

The old version of this script scored the model on the full dataset, which includes the
training rows and therefore overstates performance. The honest numbers come from the test
split that train_local.py sets aside and uses once; they are stored in reports/metrics.json.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
METRICS = BASE / "reports" / "metrics.json"

if not METRICS.exists():
    raise SystemExit("reports/metrics.json not found. Run: python src/train_local.py")

report = json.loads(METRICS.read_text())
test = report["test"]
new = test["selected_model"]
old = test["baseline_previous_approach"]
dumb = test["baseline_always_not_fraud"]

print("=" * 62)
print(f"  Held-out test set: {report['data']['split']['test']} claims, "
      f"fraud rate {report['data']['fraud_rate']:.1%}")
print(f"  Model: {report['selection']['selected']}  (threshold {new['threshold']:.3f})")
print("=" * 62)
print(f"{'':<34}{'Accuracy':>9}{'Recall':>9}{'F1':>8}")
print(f"{'Always predict not fraud':<34}{dumb['accuracy']:>9.3f}{0:>9.3f}{0:>8.3f}")
print(f"{'Previous approach':<34}{old['accuracy']:>9.3f}{old['fraud_recall']:>9.3f}{old['fraud_f1']:>8.3f}")
print(f"{'Current model':<34}{new['accuracy']:>9.3f}{new['fraud_recall']:>9.3f}{new['fraud_f1']:>8.3f}")
print(f"\nFraud precision: {new['fraud_precision']:.3f}")
print(f"PR-AUC: {new['pr_auc']:.3f}   ROC-AUC: {new['roc_auc']:.3f}")
cm = new["confusion_matrix"]
print("\nConfusion matrix (rows = true, cols = predicted):")
print(f"  Not fraud: {cm[0][0]:>5} correct, {cm[0][1]:>5} flagged as fraud")
print(f"  Fraud:     {cm[1][1]:>5} caught,  {cm[1][0]:>5} missed")
