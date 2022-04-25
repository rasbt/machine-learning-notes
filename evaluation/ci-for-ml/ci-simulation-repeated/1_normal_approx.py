import argparse
from get_dataset import get_dataset
from sklearn.tree import DecisionTreeClassifier
import scipy.stats
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

        confidence = 0.95  # Change to your desired confidence level
        z_value = scipy.stats.norm.ppf((1 + confidence) / 2.0)
        acc_test = clf.score(X_test, y_test)
        ci_length = z_value * np.sqrt((acc_test * (1 - acc_test)) / y_test.shape[0])

        ci_lower = acc_test - ci_length
        ci_upper = acc_test + ci_length

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
