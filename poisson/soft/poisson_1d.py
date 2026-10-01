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


# deepxde will provide bool whether or not the given x is on top of the boundary
def boundary(x, on_boundary: bool) -> bool:
    return on_boundary


# on the boundary, u(x) should be 0
def boundary_func(x):
    return 0


# number of points sampled within the domain and on the boundary, respectively

N_COLLOCATION = 16
N_BOUNDARY = 2
N_TEST = 100


geometry = dde.geometry.Interval(-1, 1)
boundary_condition = dde.icbc.DirichletBC(geometry, boundary_func, boundary)
data = dde.data.PDE(
    geometry,
    residual,
    boundary_condition,
    N_COLLOCATION,
    N_BOUNDARY,
    solution=None,
    num_test=N_TEST,
)

# 1 input neuron; 3 layers each 50 neurons, and 1 output neuron
layer_size = [1] + [50] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
optimizer = "adam"
lr = 1e-2
n_iters = 10000

# fully connected neural network; in this case an MLP
network = dde.nn.FNN(layer_size, activation, initializer)
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
