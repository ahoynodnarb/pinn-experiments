import jax.numpy as jnp
import matplotlib.pyplot as plt

import deepxde as dde

# -u''(x) = f(x)
# in this case, f(x) = pi^2 * sin(pi * x) and our analytical solution is u(x) = sin(pi * x)

# seemingly deepxde maintains the principle that all values passed to functions represent individual "points"
# i.e. we can just ignore the batch dimension: that is all handled by deepxde


def f(x):
    return jnp.pi**2 * jnp.sin(jnp.pi * x)


def residual(x, y):
    dy_xx, _ = dde.grad.hessian(y, x)
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

dde.saveplot(loss_history, train_state, issave=True, isplot=True)

# Optional: Restore the saved model with the smallest training loss
# model.restore(f"model/model-{train_state.best_step}.ckpt", verbose=1)
# Plot PDE residual
x = dde.geometry.uniform_points(1000, True)
y = model.predict(x, operator=residual)
plt.figure()
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("PDE residual")
plt.show()
