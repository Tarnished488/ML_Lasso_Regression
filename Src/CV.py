"""Week3 (ii): 5-fold cross-validation to select C for Lasso & Ridge.

Run:  python CV.py   (from Src/ directory)
Output: Figures/cv_lasso.png, cv_ridge.png
"""

import sys, numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_score

sys.path.insert(0, __file__[: __file__.rfind("\\") if "\\" in __file__ else __file__.rfind("/")])
from Data_loader import load_data, plot_cv_errorbar

X, y, _, _ = load_data()

poly = PolynomialFeatures(degree=5, include_bias=False)
Xp = poly.fit_transform(X)

# ---- C range justification (for report) ----
# Lower bound 1e-2: covers the all-zero-coefficient region (C < ~1.38 for this data)
# where Lasso predicts only the intercept (underfitting plateau).
# Upper bound 1e3: CV error curve flattens beyond C ~ 100 (overfitting region),
# so larger C adds no information.
# 30 points on a log scale gives a smooth readable curve.
C_values = np.logspace(-2, 3, 30)
print(f"C range: [{C_values[0]:.2e}, {C_values[-1]:.2e}]  ({len(C_values)} points, log-spaced)")
print("Justification: lower bound covers underfitting plateau (all coefs zero),")
print("upper bound reaches the flat overfitting region where MSE no longer decreases.")

# ---- baseline: dummy regressor (always predicts mean y) ----
dummy = DummyRegressor(strategy="mean")
dummy_scores = cross_val_score(dummy, Xp, y, cv=5, scoring="neg_mean_squared_error")
baseline_mse = -dummy_scores.mean()
print(f"\nBaseline (mean predictor) 5-fold CV MSE = {baseline_mse:.4f}")

# ---- (ii)(a) Lasso: 5-fold CV errorbar ----
lasso_mean, lasso_std = [], []
for C in C_values:
    m = Lasso(alpha=1 / (2 * C), max_iter=100000, tol=1e-6)
    scores = cross_val_score(m, Xp, y, cv=5, scoring="neg_mean_squared_error")
    lasso_mean.append(-scores.mean())
    lasso_std.append(scores.std())

plot_cv_errorbar(C_values, lasso_mean, lasso_std,
                 filename="cv_lasso.png", mark_best=True,
                 baseline=baseline_mse,
                 title="(ii)(a)  Lasso 5-fold CV error vs C")

best_lasso_idx = int(np.argmin(lasso_mean))
print(f"\nLasso best C = {C_values[best_lasso_idx]:.4g}  "
      f"(MSE = {lasso_mean[best_lasso_idx]:.4f} +/- {lasso_std[best_lasso_idx]:.4f})")
print(f"  Improvement over baseline: {(baseline_mse - lasso_mean[best_lasso_idx]) / baseline_mse * 100:.1f}%")

# ---- (ii)(c) Ridge: 5-fold CV errorbar ----
ridge_mean, ridge_std = [], []
for C in C_values:
    m = Ridge(alpha=1 / (2 * C))
    scores = cross_val_score(m, Xp, y, cv=5, scoring="neg_mean_squared_error")
    ridge_mean.append(-scores.mean())
    ridge_std.append(scores.std())

plot_cv_errorbar(C_values, ridge_mean, ridge_std,
                 filename="cv_ridge.png", mark_best=True,
                 baseline=baseline_mse,
                 title="(ii)(c)  Ridge 5-fold CV error vs C")

best_ridge_idx = int(np.argmin(ridge_mean))
print(f"\nRidge best C = {C_values[best_ridge_idx]:.4g}  "
      f"(MSE = {ridge_mean[best_ridge_idx]:.4f} +/- {ridge_std[best_ridge_idx]:.4f})")
print(f"  Improvement over baseline: {(baseline_mse - ridge_mean[best_ridge_idx]) / baseline_mse * 100:.1f}%")

# ---- (ii)(b) recommendation summary ----
print("\n--- (ii)(b) C selection summary ---")
print(f"Recommended Lasso C : {C_values[best_lasso_idx]:.4g}")
print(f"  Reason: lowest 5-fold CV mean MSE on the tested range.")
print(f"  At this C the model balances bias (underfitting at small C)")
print(f"  and variance (overfitting at large C).")
print(f"Recommended Ridge C : {C_values[best_ridge_idx]:.4g}")
print(f"  Same reasoning applied to L2 penalty.")

print("\ndone -- all figures saved to Figures/")
