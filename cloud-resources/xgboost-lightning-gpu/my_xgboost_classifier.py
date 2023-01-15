from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from joblib import dump


def run_classifier(save_as="my_model.joblib", use_gpu=False):
    digits = datasets.load_digits()
    features, targets = digits.images, digits.target
    features = features.reshape(-1, 8*8)

    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=123)

    if use_gpu:
        model = XGBClassifier(tree_method='gpu_hist', gpu_id=0)
    else:
        model = XGBClassifier()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100.0:.2f}%")

    dump(model, filename=save_as)


if __name__ == "__main__":
    run_classifier()
