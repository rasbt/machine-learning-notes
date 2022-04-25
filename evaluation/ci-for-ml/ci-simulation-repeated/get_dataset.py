from sklearn.datasets import make_classification


def get_dataset(random_seed):

    X, y = make_classification(
        n_samples=10_002_000,
        n_features=5,
        n_redundant=2,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=random_seed,
        flip_y=0.25,
    )

    X_train = X[:1_000]
    y_train = y[:1_000]

    X_test = X[1_000:2_000]
    y_test = y[1_000:2_000]

    X_huge_test = X[2_000:]
    y_huge_test = y[2_000:]

    return X_train, y_train, X_test, y_test, X_huge_test, y_huge_test
