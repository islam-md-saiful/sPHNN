"""
Torch reproduction of Section 4.1 / Section 2.2 of Roth et al. (2025):
a damped spinning rigid body, governed by Euler's rotation equations

    I * wdot + w x (I*w) = -mu * w

with H(w) = 0.5 * (I1 w1^2 + I2 w2^2 + I3 w3^2) the true kinetic energy.
We generate (w, wdot) pairs from the ground-truth ODE, then train an
sPHNN via derivative fitting and check that the learned energy tracks the
true kinetic energy -- including on states never seen during training.
"""
import torch
import numpy as np

from sphnn_torch import FICNN, LyapunovNN, SkewSymmetricMatrix, ConstantSPDMatrix, ISPHS, ODESolver
from sphnn_torch.training import train_derivative_fit

torch.manual_seed(0)

# ---------------------------------------------------------------- #
# 1. Ground truth: Euler's rotation equations (paper Eq. 5)         #
# ---------------------------------------------------------------- #
I = torch.tensor([1.0, 2.0, 3.0])
mu = 0.01


def euler_equations(t, w):
    tau = mu * w
    Iw = I * w
    cross = torch.linalg.cross(Iw, w) if w.dim() == 1 else torch.cross(Iw, w, dim=-1)
    return (1.0 / I) * cross - tau


def true_energy(w):
    return 0.5 * torch.sum(I * w**2, dim=-1)


# ---------------------------------------------------------------- #
# 2. Generate training data: (w, wdot) pairs sampled from 10        #
#    trajectories, matching the paper's protocol.                  #
# ---------------------------------------------------------------- #
def rk4_integrate(f, x0, ts):
    xs = [x0]
    x = x0
    for i in range(len(ts) - 1):
        dt = ts[i + 1] - ts[i]
        k1 = f(ts[i], x)
        k2 = f(ts[i] + dt / 2, x + dt / 2 * k1)
        k3 = f(ts[i] + dt / 2, x + dt / 2 * k2)
        k4 = f(ts[i] + dt, x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xs.append(x)
    return torch.stack(xs)


ts_train = torch.linspace(0, 50, 1000)
n_traj = 10
w0s = torch.rand(n_traj, 3) ** 2  # matches paper's x0s_train**2, in [0,1)^3

all_w, all_wdot = [], []
for i in range(n_traj):
    traj = rk4_integrate(euler_equations, w0s[i], ts_train)
    wdot = torch.stack([euler_equations(0.0, w) for w in traj])
    all_w.append(traj)
    all_wdot.append(wdot)

w_train = torch.cat(all_w, dim=0)        # (10*1000, 3)
wdot_train = torch.cat(all_wdot, dim=0)  # (10*1000, 3)
print(f"Training pairs: {w_train.shape[0]}")

# ---------------------------------------------------------------- #
# 3. Build sPHNN: state-dependent J(w) (angular-momentum coupling), #
#    constant (learnable) diagonal-ish PD dissipation R.            #
# ---------------------------------------------------------------- #
n = 3
ficnn = FICNN(in_size=n, out_size=1, width=16, depth=2)
H = LyapunovNN(ficnn, state_size=n, minimum_learnable=False)  # x* = 0, known equilibrium
J = SkewSymmetricMatrix(in_size=n, matrix_size=n, width=16, depth=2)
R = ConstantSPDMatrix(size=n)

model = ISPHS(hamiltonian=H, poisson_matrix=J, resistive_matrix=R, input_matrix=None)

# ---------------------------------------------------------------- #
# 4. Train via derivative fitting                                   #
# ---------------------------------------------------------------- #
hist = train_derivative_fit(
    model, x=w_train, xdot_target=wdot_train,
    num_steps=3000, lr=1e-3, verbose_every=500,
)

# ---------------------------------------------------------------- #
# 5. Evaluate: does the learned Hamiltonian track true energy along #
#    a held-out trajectory?                                         #
# ---------------------------------------------------------------- #
solver = ODESolver(model, method="dopri5")
w0_test = torch.tensor([0.6, 0.3, 0.9])
ts_test = torch.linspace(0, 100, 200)

with torch.enable_grad():
    w_pred = solver(w0_test, ts_test).detach()
w_true = rk4_integrate(euler_equations, w0_test, ts_test)

E_true = true_energy(w_true)
E_pred = H(w_pred).detach()

rmse_energy = torch.sqrt(torch.mean((E_true - E_pred) ** 2)).item()
rmse_state = torch.sqrt(torch.mean((w_true - w_pred) ** 2)).item()
print(f"\nEnergy RMSE  (predicted vs. true kinetic energy): {rmse_energy:.4f}")
print(f"State  RMSE  (predicted vs. true omega trajectory): {rmse_state:.4f}")

from sphnn_torch import is_zero_gas_guarantee_valid
print(f"0-GAS guarantee valid: {is_zero_gas_guarantee_valid(model)}")
