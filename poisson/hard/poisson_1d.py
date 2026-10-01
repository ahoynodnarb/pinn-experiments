import matplotlib.pyplot as plt
import numpy as np
import torch

if torch.mps.is_available():
    torch.set_default_device("mps")
elif torch.cuda.is_available():
    torch.set_default_device("cuda")

import deepxde as dde

# -u''(x) = f(x)
# in this case, f(x) = pi^2 * sin(pi * x) and our analytical solution is u(x) = sin(pi * x)

# seemingly deepxde maintains the principle that all values passed to functions represent individual "points"
# i.e. we can just ignore the batch dimension: that is all handled by deepxde


def f(x):
    return torch.pi**2 * torch.sin(torch.pi * x)


def residual(x, y):
    dy_xx = dde.grad.hessian(y, x)
    return -dy_xx - f(x)


# number of points sampled within the domain and on the boundary, respectively

N_COLLOCATION = 16


geometry = dde.geometry.Interval(-1, 1)
data = dde.data.PDE(
    geometry=geometry,
    pde=residual,
    bcs=[],
    num_domain=N_COLLOCATION,
    solution=None,
)

# 1 input neuron; 3 layers each 50 neurons, and 1 output neuron
layer_size = [1] + [50] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
optimizer = "adam"
lr = 1e-2
n_iters = 10000


class HardDirichlet1DFNN(dde.nn.FNN):
    def forward(self, x):
        N = super().forward(x)
        return x * (1 - x) * N


# fully connected neural network; in this case an MLP
network = HardDirichlet1DFNN(layer_size, activation, initializer)
model = dde.Model(data, network)
model.compile(optimizer, lr)
loss_history, train_state = model.train(iterations=n_iters)

x = np.random.uniform(0, 1, 1000)
u_true = np.sin(np.pi * x)
u_pred = model.predict(x[:, None]).ravel()
print(f"test MSE: {np.mean((u_true - u_pred) ** 2)}")
plt.figure()
plt.scatter(x, u_true, color="blue", label="Analytic Solution")
plt.scatter(x, u_pred, color="crimson", label="PINN Solution")
plt.show()
