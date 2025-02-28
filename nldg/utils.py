import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.stats import ortho_group


def generate_data_example_1(
        seed: int,
        plot: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates data from a specific DAG.

    Args:
        seed: The random seed.
        plot: If true, plots the underlying causal graph.

    Returns:
        A tuple of 2 pandas dataframes:
        - data_train: A dataframe containing the training data.
        - data_test: A dataframe containing the testing data.
    """
    np.random.seed(seed)

    ## Train data generation
    n1 = 1000
    n2 = 1000
    n3 = 1000
    n_train = n1 + n2 + n3

    E_train = np.concatenate([np.repeat(0, n1), np.repeat(1, n2), np.repeat(2, n3)])

    X1 = np.concatenate([
        np.random.normal(0, 1, n1),
        3 * np.random.normal(0, 1, n2) + 1,
        1.5 * np.random.normal(0, 1, n3) + 0.5
    ])

    X4 = np.concatenate([
        np.random.normal(0, 1, n1),
        3 * np.random.normal(0, 1, n2),
        2.5 * np.random.normal(0, 1, n3) + 1
    ])

    Y_train = 1.5 * X1 + X4 + 0.2 * np.random.normal(0, 1, n_train)

    X2 = 2 * X1 + np.random.normal(0, 1, n_train)

    X3 = np.concatenate([
        1.5 * Y_train[:n1] - X2[:n1] + 0.4 * np.random.normal(0, 1, n1),
        -0.3 * Y_train[n1:(n1+n2)] + X2[n1:(n1+n2)] + 0.4 * np.random.normal(0, 1, n2),
        0.5 * Y_train[(n1+n2):] + 2 * X2[(n1+n2):] + 0.4 * np.random.normal(0, 1, n3)
    ])

    X6 = np.random.normal(0, 1, n_train)

    X5 = np.concatenate([
        4 * Y_train[:n1] - X6[:n1] + 0.6 * np.random.normal(0, 1, n1),
        -2 * Y_train[n1:(n1+n2)] + 2 * X6[n1:(n1+n2)] + np.random.normal(0, 1, n2),
        1.5 * Y_train[(n1+n2):] + 1.3 * X6[(n1+n2):] + 2 * np.random.normal(0, 1, n3)
    ])

    X_train = np.column_stack([X1, X2, X3, X4, X5, X6])
    data_train = pd.DataFrame(np.column_stack([E_train, X_train, Y_train]),
                              columns=['E', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'Y'])

    ## Test data generation
    np.random.seed(seed)
    n2 = 1000
    n4 = 1000
    n_test = n2 + n4

    E_test = np.concatenate([np.repeat(1, n2), np.repeat(3, n4)])

    X1 = np.concatenate([
        3 * np.random.normal(0, 1, n2) + 1,
        -2.5 * np.random.normal(0, 1, n4) + 1.5
    ])

    X4 = np.concatenate([
        3 * np.random.normal(0, 1, n2),
        5 * np.random.normal(0, 1, n4) - 2
    ])

    Y_test = 1.5 * X1 + X4 + 0.2 * np.random.normal(0, 1, n_test)

    X2 = 2 * X1 + np.random.normal(0, 1, n_test)

    X3 = np.concatenate([
        -0.3 * Y_test[:n2] + X2[:n2] + 0.4 * np.random.normal(0, 1, n2),
        4.5 * Y_test[n2:n_test] - 2.5 * X2[n2:n_test] + 0.4 * np.random.normal(0, 1, n4)
    ])

    X6 = np.random.normal(0, 1, n_test)

    X5 = np.concatenate([
        -2 * Y_train[:n2] + 2 * X6[:n2] + np.random.normal(0, 1, n2),
        -0.5 * Y_train[n2:n_test] - 2.6 * X6[n2:n_test] + 2 * np.random.normal(0, 1, n3)
    ])

    X_test = np.column_stack([X1, X2, X3, X4, X5, X6])
    data_test = pd.DataFrame(np.column_stack([E_test, X_test, Y_test]),
                             columns=['E', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'Y'])

    if plot:
        G = nx.DiGraph()

        nodes = ["E", "X1", "X2", "X3", "X4", "X5", "X6", "Y"]
        G.add_nodes_from(nodes)

        edges = [
            ("E", "X1"), ("E", "X3"), ("E", "X4"), ("E", "X5"),
            ("X1", "X2"), ("X1", "Y"),
            ("X2", "X3"),
            ("X4", "Y"),
            ("X6", "X5"),
            ("Y", "X3"), ("Y", "X5")
        ]
        G.add_edges_from(edges)

        pos = {
            "E": (-2, 0.5),
            "X1": (-1, 1), "X4": (1, 1),
            "X2": (-1, 0.5), "X3": (-1, 0),
            "Y": (0, 0.5),
            "X5": (1, 0), "X6": (2, 0.5)
        }

        plt.figure(figsize=(6, 4))
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=12, arrowsize=20)
        plt.show()

    return data_train, data_test


def gen_data_isd(
    n_train: int = 1000,
    n_test: int = 500,
    rng_train: np.random.Generator = np.random.default_rng(42),
    rng_test: np.random.Generator = np.random.default_rng(42),
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Generates data to be used for Random Forests with Invariant Subspace Decomposition.

    Args:
        n_train: Number of train data.
        n_test: Number of test data.
        rng_train: Random number generator used for the train data.
        rng_test: Random number generator used for the test data.

    Returns:
        A tuple of 4 elements:
        - data_train: Data frame containing the train data.
        - data_test: Data frame containing the test data.
        - Sigma_list_train: Covariance matrix of the features of the train data for each environment.
        - Sigma_list_test: Covariance matrix of the features of the test data for each environment.
    """
    n_envs_train = 4
    block_sizes = [2, 2, 2]
    p = sum(block_sizes)
    X = np.zeros((n_train, p))
    mu_x = np.zeros(p)
    Y = np.zeros((n_train))
    y_mean = 0
    eps = 0.8 * rng_train.normal(size=(n_train))
    E = np.zeros((n_train))
    envs_train = [0, 1, 2, 3]
    n_e = int(n_train / n_envs_train)
    starts = [j * n_e for j in range(n_envs_train)]
    Sigma_list_train = np.zeros((n_envs_train, p, p))
    OM = ortho_group.rvs(dim=p, random_state=rng_train)
    rng_sigma = np.random.default_rng(0)

    for i, st in enumerate(starts):
        E[st:st + n_e] = envs_train[i]
        A = block_diag(*[rng_sigma.random((bs, bs)) for bs in block_sizes])
        Sigma = OM.T @ (A @ A.T + 0 * np.eye(p)) @ OM
        Sigma_list_train[i, :, :] = Sigma
        X[st:st + n_e, :] = rng_train.multivariate_normal(mean=mu_x, cov=Sigma, size=n_e)
        if i == 0:
            Y[st:st + n_e] = np.sin(X[st:st + n_e, 0]) + np.cos(X[st:st + n_e, 1])
        elif i == 1:
            Y[st:st + n_e] = np.sin(X[st:st + n_e, 0]) + X[st:st + n_e, 1] ** 2
        elif i == 2:
            Y[st:st + n_e] = -np.sin(X[st:st + n_e, 0]) + np.cos(X[st:st + n_e, 1])
        else:
            Y[st:st + n_e] = np.cos(X[st:st + n_e, 0]) + X[st:st + n_e, 1] ** 3
    Y = Y + y_mean + eps

    data_train = pd.DataFrame(np.column_stack([E, X, Y]),
                              columns=['E', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'Y'])

    n_envs_test = 2
    X = np.zeros((n_test, p))
    Y = np.zeros((n_test))
    eps = 0.8 * rng_test.normal(size=(n_test))
    E = np.zeros((n_test))
    envs_test = [2, 4]
    n_e = int(n_test / n_envs_test)
    starts = [j * n_e for j in range(n_envs_test)]
    Sigma_list_test = np.zeros((n_envs_test, p, p))
    OM = ortho_group.rvs(dim=p, random_state=rng_test)
    rng_sigma = np.random.default_rng(12)

    for i, st in enumerate(starts):
        E[st:st + n_e] = envs_test[i]
        A = block_diag(*[rng_sigma.random((bs, bs)) for bs in block_sizes])
        Sigma = OM.T @ (A @ A.T + 0 * np.eye(p)) @ OM
        Sigma_list_test[i, :, :] = Sigma
        X[st:st + n_e, :] = rng_test.multivariate_normal(mean=mu_x, cov=Sigma, size=n_e)
        if i == 0:
            Y[st:st + n_e] = np.sin(X[st:st + n_e, 0]) + X[st:st + n_e, 1] ** 2  # same as second env in train
        else:
            Y[st:st + n_e] = np.sin(X[st:st + n_e, 0]) ** 2 - X[st:st + n_e, 1] ** 2
    Y = Y + y_mean + eps

    data_test = pd.DataFrame(np.column_stack([E, X, Y]),
                             columns=['E', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'Y'])

    return data_train, data_test, Sigma_list_train, Sigma_list_test


def gen_data_maximin(
    n_train: int = 1000,
    n_test: int = 500,
    rng_train: np.random.Generator = np.random.default_rng(42),
    rng_test: np.random.Generator = np.random.default_rng(42),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates data to be used for Random Forests with minimizing the maximum error across environments.

    Args:
        n_train: Number of train data.
        n_test: Number of test data.
        rng_train: Random number generator used for the train data.
        rng_test: Random number generator used for the test data.

    Returns:
        A tuple of 2 dataframes:
        - data_train: Data frame containing the train data.
        - data_test: Data frame containing the test data.
    """
    #n_envs_train = 4
    n_envs_train = 5
    #block_sizes = [2, 2, 2]
    block_sizes = [3, 3, 3]
    p = sum(block_sizes)

    X = np.zeros((n_train, p))
    mu_x = np.zeros(p)
    Y = np.zeros((n_train))
    y_mean = 0
    eps = 0.8 * rng_train.normal(size=(n_train))

    E = np.zeros((n_train))
    envs_train = np.arange(n_envs_train)
    n_e = int(n_train / n_envs_train)
    starts = [j * n_e for j in range(n_envs_train)]
    OM = ortho_group.rvs(dim=p, random_state=rng_train)
    rng_sigma = np.random.default_rng(0)

    A = block_diag(*[rng_sigma.random((bs, bs)) for bs in block_sizes])
    #Sigma = OM.T @ (A @ A.T + 0 * np.eye(p)) @ OM
    Sigma = OM.T @ (A @ A.T + 0.2 * np.eye(p)) @ OM

    for i, st in enumerate(starts):
        E[st:st + n_e] = envs_train[i]
        X[st:st + n_e, :] = rng_train.multivariate_normal(mean=mu_x, cov=Sigma, size=n_e)
        if i == 0:
            #Y[st:st + n_e] = np.sin(X[st:st + n_e, 0]) + np.cos(X[st:st + n_e, 1])
            Y[st:st + n_e] = 1.5 * np.sin(X[st:st + n_e, 0]) + 2.7 * X[st:st + n_e, 1] ** 2
        elif i == 1:
            #Y[st:st + n_e] = np.sin(X[st:st + n_e, 0]) + X[st:st + n_e, 1] ** 2
            Y[st:st + n_e] = 0.3 * X[st:st + n_e, 0] - 1.4 * np.abs(X[st:st + n_e, 1]) + X[st:st + n_e, 2] ** 3
        elif i == 2:
            #Y[st:st + n_e] = -np.sin(X[st:st + n_e, 0]) + np.cos(X[st:st + n_e, 1])
            Y[st:st + n_e] = 5 * X[st:st + n_e, 0] - 3.5 * X[st:st + n_e, 1] + np.sin(X[st:st + n_e, 2])
        elif i == 3:
            #Y[st:st + n_e] = np.cos(X[st:st + n_e, 0]) + X[st:st + n_e, 1] ** 3
            Y[st:st + n_e] = -3.7 * np.log(np.abs(X[st:st + n_e, 0]) + 1) + 0.2 * X[st:st + n_e, 1] ** 2
        else:
            Y[st:st + n_e] = 2 * np.tanh(X[st:st + n_e, 0]) - 3 * X[st:st + n_e, 1] + X[st:st + n_e, 2]
    Y = Y + y_mean + eps

    data_train = pd.DataFrame(np.column_stack([E, X, Y]),
                              columns=['E'] + [f'X{i+1}' for i in range(p)] + ['Y'])

    #n_envs_test = 2
    n_envs_test = 3
    X = np.zeros((n_test, p))
    Y = np.zeros((n_test))
    eps = 0.8 * rng_test.normal(size=(n_test))

    E = np.zeros((n_test))
    #envs_test = [2, 4]
    envs_test = [2, 4, 5]
    n_e = int(n_test / n_envs_test)
    starts = [j * n_e for j in range(n_envs_test)]
    ##OM = ortho_group.rvs(dim=p, random_state=rng_test)

    ##A = block_diag(*[rng_sigma.random((bs, bs)) for bs in block_sizes])
    ##Sigma = OM.T @ (A @ A.T + 0 * np.eye(p)) @ OM

    for i, st in enumerate(starts):
        E[st:st + n_e] = envs_test[i]
        X[st:st + n_e, :] = rng_test.multivariate_normal(mean=mu_x, cov=Sigma, size=n_e)
        if i == 0:
            #Y[st:st + n_e] = np.sin(X[st:st + n_e, 0]) + X[st:st + n_e, 1] ** 2  # same as second env in train
            Y[st:st + n_e] = 0.3 * X[st:st + n_e, 0] - 1.4 * np.abs(X[st:st + n_e, 1]) + X[st:st + n_e, 2] ** 3
        elif i == 1:
            #Y[st:st + n_e] = X[st:st + n_e, 0] + X[st:st + n_e, 1]
            Y[st:st + n_e] = -8.2 * X[st:st + n_e, 0] + 5.1 * X[st:st + n_e, 1] - 2 * X[st:st + n_e, 2]
        else:
            Y[st:st + n_e] = 4 * np.tanh(X[st:st + n_e, 0]) - 2 * X[st:st + n_e, 1] + np.cos(X[st:st + n_e, 2])
    Y = Y + y_mean + eps

    data_test = pd.DataFrame(np.column_stack([E, X, Y]),
                             columns=['E'] + [f'X{i+1}' for i in range(p)] + ['Y'])

    return data_train, data_test
