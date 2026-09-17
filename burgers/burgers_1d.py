import os

import matplotlib.pyplot as plt
import numpy as np
import torch

if torch.mps.is_available():
    torch.set_default_device("mps")
elif torch.cuda.is_available():
    torch.set_default_device("cuda")

import deepxde as dde

v = 1e-2 / np.pi


def gen_testdata():
    data = np.load("./dataset/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y


# data is just a vector being (x, t)
def residual(data, u):
    du_dx = dde.grad.jacobian(u, data, i=0, j=0)
    du_dt = dde.grad.jacobian(u, data, i=0, j=1)
    du_dxx = dde.grad.hessian(u, data, i=0, j=0)
    return du_dt + u * du_dx - v * du_dxx


def boundary_func(data):
    return 0


def boundary(data, on_boundary):
    return on_boundary


# have to make sure x's shape has the same structure as data to prevent weird broadcasting problems
def initial_func(data):
    x = data[:, 0, None]
    return -np.sin(np.pi * x)


def initial(data, on_boundary):
    return on_boundary


def get_model(n_collocation, n_boundary, n_initial):
    geometry = dde.geometry.Interval(-1, 1)
    time_domain = dde.geometry.TimeDomain(0, 0.99)
    # it would technically be possible to do this with a rectangular domain, but that messes up our boundary conditions
    # this way, deepxde can differentiate between the geometry and the time domains so that the boundary conditions are only checked against the geometry and not time
    geomtime = dde.geometry.GeometryXTime(geometry, time_domain)

    boundary_condition = dde.icbc.DirichletBC(
        geomtime,
        boundary_func,
        boundary,
    )
    initial_conditon = dde.icbc.IC(
        geomtime,
        initial_func,
        initial,
    )
    train_data = dde.data.TimePDE(
        geomtime,
        residual,
        [boundary_condition, initial_conditon],
        num_domain=n_collocation,
        num_boundary=n_boundary,
        num_initial=n_initial,
    )

    layers = [2] + [20] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot normal"
    optimizer = "adam"
    lr = 1e-3

    # fully connected neural network; in this case an MLP
    network = dde.nn.FNN(layers, activation, initializer)
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
    X, u_true = gen_testdata()
    indices = np.random.choice(X.shape[0], size=3200, replace=False)

    X = X[indices]
    u_true = u_true[indices].ravel()
    u_pred = model.predict(X).ravel()

    x, t = X.T

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")

    ax.scatter(
        x,
        t,
        u_true,
        c="royalblue",
        marker="o",
        s=40,
        label="Analytic Solution",
    )
    ax.scatter(
        x,
        t,
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

    N_COLLOCATION = 2540
    N_BOUNDARY = 80
    N_INITIAL = 160
    N_TRAIN_ITERS = 15000

    model = get_model(N_COLLOCATION, N_BOUNDARY, N_INITIAL)
    train_model(model, N_TRAIN_ITERS, os.path.join(CHECKPOINT_PATH, "ckpt"))
    # model.restore(os.path.join(CHECKPOINT_PATH, "ckpt-15000.pt"))
    plot_solutions(model)
