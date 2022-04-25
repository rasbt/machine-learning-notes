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

        predictions_test = clf.predict(X_test)

        rng = np.random.RandomState(seed=12345)
        idx = np.arange(y_test.shape[0])

        test_accuracies = []

        for i in range(200):

            pred_idx = rng.choice(idx, size=idx.shape[0], replace=True)
            acc_test_boot = np.mean(predictions_test[pred_idx] == y_test[pred_idx])
            test_accuracies.append(acc_test_boot)

        ci_lower = np.percentile(test_accuracies, 2.5)
        ci_upper = np.percentile(test_accuracies, 97.5)

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
