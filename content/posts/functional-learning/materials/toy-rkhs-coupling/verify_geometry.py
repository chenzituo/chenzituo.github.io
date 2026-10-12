"""Independent finite-dimensional checks of the blog's projection/law identities."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

rng = np.random.default_rng(431)
# H is Euclidean in orthonormal coordinates; observations leave 4 missing modes.
O = rng.normal(size=(3, 7))
G = O@O.T
M = np.linalg.inv(G)
S = O.T@M
P = S@O
u, v = rng.normal(size=(11, 3)), rng.normal(size=(11, 3))
observed_cost = np.einsum('abi,ij,abj->ab', u[:, None]-v[None, :], M, u[:, None]-v[None, :])
field_cost = cdist(u@S.T, v@S.T, 'sqeuclidean')
row, col = linear_sum_assignment(observed_cost)
row2, col2 = linear_sum_assignment(field_cost)
F = rng.normal(size=(11, 7))
projected = F@P.T
floor = np.mean(np.sum((F-projected)**2, axis=1))
i, j = linear_sum_assignment(cdist(F, projected, 'sqeuclidean'))
best_projected_cost = np.mean(np.sum((F[i]-projected[j])**2, axis=1))
report = {
    'reobservation_max_error': float(abs(O@S-np.eye(3)).max()),
    'orthogonal_projection_max_error': float(max(abs(P@P-P).max(), abs(P-P.T).max())),
    'pairwise_isometry_max_error': float(abs(observed_cost-field_cost).max()),
    'wasserstein_squared_cost_difference': float(abs(observed_cost[row, col].mean()-field_cost[row2, col2].mean())),
    'projection_floor': float(floor),
    'projection_floor_attainment_error': float(abs(best_projected_cost-floor)),
    'source_rkhs_second_moment': float(np.trace(S@G@S.T)),
}
assert max(report[k] for k in report if 'error' in k or 'difference' in k) < 1e-10
assert abs(report['source_rkhs_second_moment']-3) < 1e-10
Path(__file__).with_name('geometry-checks.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
