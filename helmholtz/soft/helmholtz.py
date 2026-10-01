import os

import numpy as np
import torch

if torch.mps.is_available():
    torch.set_default_device("mps")
elif torch.cuda.is_available():
    torch.set_default_device("cuda")

import matplotlib.pyplot as plt

import deepxde as dde

k = 2 * np.pi


class EndCheckpointer(dde.callbacks.ModelCheckpoint):
    def on_train_end(self):
        self.on_epoch_end()


def solution(data):
    x, y = data.T
    return np.sin(k * x) * np.sin(k * y)


def f(data):
    x, y = data.T
    x = x[:, None]
    y = y[:, None]
    return -(k**2) * torch.sin(k * x) * torch.sin(k * y)


# data is just a vector being (x, t)
def residual(data, u):
    du_xx = dde.grad.hessian(u, data, i=0, j=0)
    du_yy = dde.grad.hessian(u, data, i=1, j=1)
    lap = du_xx + du_yy
    return lap + k**2 * u - f(data)


def boundary_func(data):
    return 0


def boundary(data, on_boundary):
    return on_boundary


def get_network():

    layers = [2] + [150] * 3 + [1]
    activation = "sin"
    initializer = "Glorot normal"
    # fully connected neural network; in this case an MLP
    network = dde.nn.FNN(layers, activation, initializer)

    return network


def run_training_instance(
    model, optimizer, ckpt_path, iters=None, lr=None, restore_path=None
):
    checkpointer = EndCheckpointer(ckpt_path, verbose=1, period=1000)
    model.compile(optimizer=optimizer, lr=lr)
    if restore_path is not None:
        model.restore(restore_path)
    model.train(iterations=iters, callbacks=[checkpointer])


def train_model(
    net,
    n_collocation,
    n_boundary,
    n_iters,
    checkpoint_path,
    pre=True,
    post=True,
    restore_path=None,
):
    ckpt_path_pre = os.path.join(checkpoint_path, "ckpt-pre")
    ckpt_path_post = os.path.join(checkpoint_path, "ckpt-post")

    geometry = dde.geometry.Rectangle([0, 0], [1, 1])

    boundary_condition = dde.icbc.DirichletBC(
        geometry,
        boundary_func,
        boundary,
    )
    train_data = dde.data.PDE(
        geometry,
        residual,
        [boundary_condition],
        num_domain=n_collocation,
        num_boundary=n_boundary,
    )

    lr = 1e-4
    model = dde.Model(train_data, net)

    if pre:
        run_training_instance(
            model,
            "adam",
            ckpt_path_pre,
            iters=n_iters,
            lr=lr,
            restore_path=restore_path,
        )
        restore_path = None

    if post:
        run_training_instance(
            model,
            "L-BFGS",
            ckpt_path_post,
            restore_path=restore_path,
        )
        restore_path = None

    return model


def plot_solutions(model):
    x = np.linspace(0, 1, 40)
    y = np.linspace(0, 1, 40)
    xx, yy = np.meshgrid(x, y)

    test_points = np.vstack([xx.ravel(), yy.ravel()]).T

    u_pred = model.predict(test_points).ravel()
    u_true = solution(test_points)

    print("Mean squared testing error: ", np.mean((u_true - u_pred) ** 2))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")

    ax.scatter(
        xx.ravel(),
        yy.ravel(),
        u_true,
        c="royalblue",
        marker="o",
        s=40,
        edgecolors="w",
        label="Analytic Solution",
    )
    ax.scatter(
        xx.ravel(),
        yy.ravel(),
        u_pred,
        c="crimson",
        marker="o",
        s=40,
        edgecolors="w",
        label="PINN Solution",
    )

    ax.set_title("Helmholtz Equation 2D")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    ax.legend(loc="upper left")

    plt.show()


if __name__ == "__main__":
    CHECKPOINT_PATH = "./checkpoints/"
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    N_COLLOCATION = 2540
    N_BOUNDARY = 300
    N_TRAIN_ITERS = 15000

    net = get_network()
    model = train_model(
        net,
        N_COLLOCATION,
        N_BOUNDARY,
        N_TRAIN_ITERS,
        CHECKPOINT_PATH,
    )
    plot_solutions(model)
