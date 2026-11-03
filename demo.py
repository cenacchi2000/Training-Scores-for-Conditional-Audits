"""Illustrative synthetic example; does not reproduce the 500-seed paper result."""
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from conditional_audit import overlap_weights, permutation_test

def main():
    rng = np.random.default_rng(2027)
    def sample(n):
        z = rng.integers(0, 4, n)
        y = rng.binomial(1, np.array([.05, .2, .8, .95])[z])
        x = np.column_stack([z, .7 * y + rng.normal(size=n), rng.normal(size=(n, 3))])
        return x, y, z
    xtr, ytr, ztr = sample(1200)
    xte, yte, zte = sample(600)
    report = {}
    # Same held-out records and randomization seed for both learners.
    for name, weights in [("standard", None), ("overlap", overlap_weights(ytr, ztr))]:
        model = RandomForestClassifier(n_estimators=80, min_samples_leaf=5, random_state=7)
        model.fit(xtr, ytr, sample_weight=weights)
        scores = model.predict_proba(xte)[:, 1]
        report[name] = {"marginal_auroc": roc_auc_score(yte, scores),
                        **permutation_test(scores, yte, zte)}
    print(json.dumps(report, indent=2))
if __name__ == "__main__":
    main()
