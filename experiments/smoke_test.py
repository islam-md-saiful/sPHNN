"""
Quick end-to-end sanity check of the sphnn_torch port: builds a random
sPHNN (n=3 state), checks the 0-GAS guarantee holds by construction,
integrates a trajectory, and confirms the Hamiltonian is non-increasing
along it (the core physical claim of Theorem 3.1).
"""
import torch

from sphnn_torch import (
    FICNN, LyapunovNN, SkewSymmetricMatrix, SPDMatrix, MatrixFunction,
    ISPHS, ODESolver, is_zero_gas_guarantee_valid,
)

torch.manual_seed(0)
n = 3  # state dimension
m = 2  # input dimension

# --- build a random sPHNN --------------------------------------------------
ficnn = FICNN(in_size=n, out_size=1, width=16, depth=2)
H = LyapunovNN(ficnn, state_size=n, minimum_learnable=False)  # x* fixed at 0
J = SkewSymmetricMatrix(in_size=n, matrix_size=n, width=16, depth=2)
R = SPDMatrix(in_size=n, matrix_size=n, width=16, depth=2)  # strictly PD -> R(x) >> 0 everywhere
G = MatrixFunction(in_size=n, shape=(n, m), width=16, depth=2)

model = ISPHS(hamiltonian=H, poisson_matrix=J, resistive_matrix=R, input_matrix=G)

# --- structural checks ------------------------------------------------------
print("=== structural checks ===")
x = torch.randn(5, n, requires_grad=True)  # batch of 5
xdot = model(torch.tensor(0.0), x, u=None)
print("xdot shape (expect (5, 3)):", xdot.shape)

Jx = J(x)
print("J skew-symmetric? max|J + J^T| =", (Jx + Jx.transpose(-1, -2)).abs().max().item())

Rx = R(x)
eigR = torch.linalg.eigvals(Rx).real
print("R eigenvalues (all should be > 0):", eigR.detach().numpy())

print("0-GAS guarantee valid by construction:", is_zero_gas_guarantee_valid(model))

# --- integrate a trajectory and check H is non-increasing ------------------
print("\n=== dynamics check (unforced system, u=None) ===")
model_unforced = ISPHS(hamiltonian=H, poisson_matrix=J, resistive_matrix=R, input_matrix=None)
solver = ODESolver(model_unforced, method="dopri5")

x0 = torch.tensor([1.0, -0.5, 0.8])
ts = torch.linspace(0.0, 10.0, 50)
# NOTE: torch.no_grad() must NOT wrap this call -- the PHS dynamics need
# autograd internally to compute grad_x H(x) at every solver step, even
# during pure inference/rollout (this mirrors JAX's `jax.grad` being used
# inside the ODE right-hand side in the original dynax code). We simply
# don't call .backward() afterwards, and detach the final result.
traj = solver(x0, ts)  # (50, 3)
H_traj = H(traj)
traj = traj.detach()
H_traj = H_traj.detach()

print("H(t) along trajectory (should be non-increasing, ->", H(torch.zeros(n)).item(), "):")
print(H_traj.numpy())
is_nonincreasing = bool(torch.all(H_traj[1:] <= H_traj[:-1] + 1e-4))
print("H non-increasing along trajectory:", is_nonincreasing)
print("Final state (should be near 0):", traj[-1].numpy())

print("\nSMOKE TEST PASSED" if is_nonincreasing else "\nSMOKE TEST FAILED")
