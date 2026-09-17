import os

import matplotlib.pyplot as plt
import numpy as np
import torch

if torch.mps.is_available():
    torch.set_default_device("mps")
elif torch.cuda.is_available():
    torch.set_default_device("cuda")

import deepxde as dde

# du_dt = alpha * du_xx
# 0 <= x < = 1, 0 <= t <= T

# analytical solution given by u(x,t) = exp(-alpha * pi^2 * t) * sin(pi * x)

alpha = 0.1
T = 1


def u_true(data):
    x, t = data.T
    return np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)


def residual(data, u):
    du_dt = dde.grad.jacobian(u, data, i=0, j=1)
    du_xx = dde.grad.hessian(u, data, i=0, j=0)
    return du_dt - alpha * du_xx


def boundary(data, on_boundary):
    return on_boundary


def boundary_func(data):
    return 0


def initial(data, on_initial):
    return on_initial


# have to make sure x's shape has the same structure as data to prevent weird broadcasting problems
def initial_func(data):
    x = data[:, 0, None]
    return np.sin(np.pi * x)


def get_model(n_collocation, n_boundary, n_initial):

    geometry = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, T)
    domains = dde.geometry.GeometryXTime(geometry, timedomain)
    boundary_condition = dde.icbc.DirichletBC(domains, boundary_func, boundary)
    initial_condition = dde.icbc.IC(domains, initial_func, initial)

    train_data = dde.data.TimePDE(
        domains,
        residual,
        [boundary_condition, initial_condition],
        num_domain=n_collocation,
        num_boundary=n_boundary,
        num_initial=n_initial,
    )

    layers = [2] + [50] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot normal"
    network = dde.nn.FNN(layers, activation, initializer)

    optimizer = "adam"
    lr = 1e-3

    model = dde.Model(train_data, network)
    model.compile(optimizer, lr)

    return model


def train_model(model, n_iters, checkpoint_path):

    checkpointer = dde.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, period=1000
    )

    loss_history, train_state = model.train(
        iterations=n_iters, callbacks=[checkpointer]
    )

    return loss_history, train_state


def plot_solutions(model):
    x = np.linspace(0, 1, 40)
    t = np.linspace(0, T, 40)
    xx, tt = np.meshgrid(x, t)

    test_points = np.vstack([xx.ravel(), tt.ravel()]).T

    u_true = u_true(test_points)
    u_pred = model.predict(test_points).ravel()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")

    ax.scatter(
        xx.ravel(),
        tt.ravel(),
        u_true,
        c="royalblue",
        marker="o",
        s=40,
        edgecolors="w",
        label="Analytic Solution",
    )
    ax.scatter(
        xx.ravel(),
        tt.ravel(),
        u_pred,
        c="crimson",
        marker="o",
        s=40,
        edgecolors="w",
        label="PINN Solution",
    )

    ax.set_title("Heat Equation 1D")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u")
    ax.legend(loc="upper left")

    plt.show()
    print("L2 relative error:", dde.metrics.l2_relative_error(u_true, u_pred))


if __name__ == "__main__":
    CHECKPOINT_PATH = "./checkpoints/"
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    N_COLLOCATION = 2540
    N_BOUNDARY = 100
    N_INITIAL = 100
    N_TRAIN_ITERS = 15000

    model = get_model(N_COLLOCATION, N_BOUNDARY, N_INITIAL)
    train_model(model, N_TRAIN_ITERS, os.path.join(CHECKPOINT_PATH, "ckpt"))
    # model.restore(os.path.join(CHECKPOINT_PATH, "ckpt-15000.pt"))
    plot_solutions(model)
