import argparse
from get_dataset import get_dataset
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def run_method(num_repetitions):
    is_inside_list = []

    for i in range(num_repetitions):

        X_train, y_train, X_test, y_test, X_huge_test, y_huge_test = get_dataset(
            random_seed=i
        )

        clf = DecisionTreeClassifier(random_state=123, max_depth=3)
        clf.fit(X_train, y_train)

        acc_test_true = clf.score(X_huge_test, y_huge_test)

        #####################################################
        # Compute CI
        #####################################################

        rng = np.random.RandomState(seed=12345)
        idx = np.arange(y_train.shape[0])

        bootstrap_train_accuracies = []
        bootstrap_rounds = 200
        weight = 0.632

        for i in range(bootstrap_rounds):

            train_idx = rng.choice(idx, size=idx.shape[0], replace=True)
            valid_idx = np.setdiff1d(idx, train_idx, assume_unique=False)

            boot_train_X, boot_train_y = X_train[train_idx], y_train[train_idx]
            boot_valid_X, boot_valid_y = X_train[valid_idx], y_train[valid_idx]

            clf.fit(boot_train_X, boot_train_y)
            train_acc = clf.score(X_train, y_train)
            valid_acc = clf.score(boot_valid_X, boot_valid_y)
            acc = weight * train_acc + (1.0 - weight) * valid_acc

            bootstrap_train_accuracies.append(acc)

        ci_lower = np.percentile(bootstrap_train_accuracies, 2.5)
        ci_upper = np.percentile(bootstrap_train_accuracies, 97.5)

        # Check CI
        is_inside = acc_test_true >= ci_lower and acc_test_true <= ci_upper

        is_inside_list.append(is_inside)

    return is_inside_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--repetitions",
        required=True,
        type=int,
    )

    args = parser.parse_args()
    is_inside_list = run_method(args.repetitions)

    print(
        f"{np.mean(is_inside_list)*100}% of 95% confidence"
        " intervals contain the true accuracy."
    )
