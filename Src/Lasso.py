import sys, numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge

sys.path.insert(0, __file__[: __file__.rfind("\\") if "\\" in __file__ else __file__.rfind("/")])
from Data_loader import load_data, plot_data_3d, plot_fit_surface, plot_coef_vs_C, make_grid

X, y, _, _ = load_data()

poly = PolynomialFeatures(degree=5, include_bias=False)
Xp = poly.fit_transform(X)
names = poly.get_feature_names_out(["x1", "x2"])

print(f"polynomial features: {Xp.shape[1]}  (degree 5 on 2 inputs)")
print(f"features: {list(names)}")


def find_zero_C():
    lo, hi = 1e-6, 1.0
    while True:
        m = Lasso(alpha=1 / (2 * hi), max_iter=100000, tol=1e-6).fit(Xp, y)
        if np.any(m.coef_ != 0):
            break
        hi *= 10
        if hi > 1e6:
            return 1e-4
    for _ in range(40):
        mid = (lo + hi) / 2
        m = Lasso(alpha=1 / (2 * mid), max_iter=100000, tol=1e-6).fit(Xp, y)
        if np.all(m.coef_ == 0):
            lo = mid
        else:
            hi = mid
    return lo


C_zero = find_zero_C()
print(f"\nC where all coeffs first become non-zero: ~{C_zero:.6f}")

C_values = np.unique([
    C_zero * 0.5, C_zero * 0.9, C_zero, C_zero * 2, C_zero * 5,
    0.01, 0.1, 1.0, 10.0, 100.0, 1000.0,
])
print(f"C values ({len(C_values)}): {C_values}")

plot_data_3d(X, y)

lasso_coefs = []
for C in C_values:
    m = Lasso(alpha=1 / (2 * C), max_iter=100000, tol=1e-6).fit(Xp, y)
    lasso_coefs.append(m.coef_.copy())
    n_nonzero = np.count_nonzero(m.coef_)
    print(f"\n--- Lasso C={C:.6g}  (alpha={1/(2*C):.4g})  non-zero={n_nonzero}/{len(names)} ---")
    for name, val in zip(names, m.coef_):
        if val != 0:
            print(f"  {name:>12s} : {val:+.6f}")
    print(f"  intercept     : {m.intercept_:+.6f}")

lasso_coefs = np.array(lasso_coefs)

plot_coef_vs_C(C_values, lasso_coefs, names,
               filename="lasso_coef.png",
               title="(i)(b)  Lasso coefficients vs C  (polynomial degree 5)")

G1, G2, Xgrid = make_grid(-1.5, 1.5, 51)
Xgrid_p = poly.transform(Xgrid)

surface_Cs = [C_values[2], C_values[len(C_values)//2], C_values[-1]]

surfaces = []
for C in surface_Cs:
    m = Lasso(alpha=1 / (2 * C), max_iter=100000, tol=1e-6).fit(Xp, y)
    Z = m.predict(Xgrid_p).reshape(G1.shape)
    surfaces.append((f"Lasso C={C:.4g}", G1, G2, Z))

plot_fit_surface(X, y, surfaces, filename="lasso_surface.png",
                 title="(i)(c)  Lasso predictions on feature grid  (degree 5 poly)")

ridge_coefs = []
for C in C_values:
    m = Ridge(alpha=1 / (2 * C)).fit(Xp, y)
    ridge_coefs.append(m.coef_.copy())

ridge_coefs = np.array(ridge_coefs)

# Print Ridge parameters for representative C values (same as Lasso output format)
for ci in [2, len(C_values)//2, -1]:
    C = C_values[ci]
    idx = list(C_values).index(C)
    print(f"\n--- Ridge C={C:.6g}  (alpha={1/(2*C):.4g}) ---")
    for name, val in zip(names, ridge_coefs[idx]):
        if abs(val) > 1e-6:
            print(f"  {name:>12s} : {val:+.6f}")
    print(f"  intercept     : {Ridge(alpha=1/(2*C)).fit(Xp, y).intercept_:+.6f}")

plot_coef_vs_C(C_values, ridge_coefs, names,
               filename="ridge_coef.png",
               title="(i)(e)  Ridge coefficients vs C  (polynomial degree 5)")

ridge_surfaces = []
for C in surface_Cs:
    m = Ridge(alpha=1 / (2 * C)).fit(Xp, y)
    Z = m.predict(Xgrid_p).reshape(G1.shape)
    ridge_surfaces.append((f"Ridge C={C:.4g}", G1, G2, Z))

plot_fit_surface(X, y, ridge_surfaces, filename="ridge_surface.png",
                 title="(i)(e)  Ridge predictions on feature grid  (degree 5 poly)")

print("\ndone -- all figures saved to Figures/")
