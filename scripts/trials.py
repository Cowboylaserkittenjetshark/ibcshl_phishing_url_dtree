import logging
from joblib import parallel_backend
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    HalvingRandomSearchCV,
)
from sklearn.metrics import (
    classification_report,
    accuracy_score,
)
import common.setup
from common.data import fetched_features


def __feature_importance(model, X):
    def sortSecond(val):
        return val[1]

    values = model.feature_importances_
    features = list(X)
    importances = [(features[i], values[i]) for i in range(len(features))]
    importances.sort(reverse=True, key=sortSecond)
    return importances


def __score_model(clf, X_train, y_train, X_valid, y_valid, X_test, y_test):
    score = clf.score(X_valid, y_valid)
    test_accuracy = accuracy_score(y_test, clf.predict(X_test))
    test_report = classification_report(y_test, clf.predict(X_test))
    cv_score = cross_val_score(clf, X_train, y_train, cv=7)
    report = classification_report(y_valid, clf.predict(X_valid))

    logging.info("Score: %s\n", score)
    logging.info("Test accuracy: %s\n", test_accuracy)
    logging.info("Test classification report:\n%s\n", test_report)
    logging.info("Cross val score: %s\n", cv_score)
    logging.info("Classification report:\n%s\n", report)


def __build_header(name, param_grid, with_fetched, feature_importance_model):
    options = " ("
    if with_fetched:
        options = options + "With fetched features"
    else:
        options = options + "Local features"
    if param_grid:
        options = options + ", tuned"
    else:
        options = options + ", untuned"
    if feature_importance_model:
        options = options + ", feature selection"
    options = options + ")"

    return name + options


def run_trial(
    name,
    classifier,
    X,
    y,
    params=None,
    with_fetched=False,
    feature_importance_model=None,
    fit_all_finally=False
):
    with parallel_backend("threading", n_jobs=-1):
        header = __build_header(
            name,
            params,
            with_fetched,
            feature_importance_model,
        )
        logging.info("==== %s ====", header)

        if not with_fetched:
            logging.debug("Dropping fetched features")
            X = X.drop(fetched_features, axis=1)

        if feature_importance_model:
            logging.debug("Selecting top 20 features")
            feature_importance = __feature_importance(
                feature_importance_model, X
            )
            X = X[[col[0] for col in feature_importance[:20]]]

        logging.debug("Splitting data")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=0.2
        )
        print(X_train.columns)

        clf = None
        if params:
            clf = (
                HalvingRandomSearchCV(classifier, params, verbose=0)
                .fit(X_train, y_train)
                .best_estimator_
            )
        else:
            clf = classifier.fit(X_train, y_train)

        if hasattr(clf, "feature_importances_"):
            feat_importance = __feature_importance(clf, X)
            logging.debug(
                "Feature importance: %s",
                str(feat_importance),
            )

        __score_model(
            clf,
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test,
        )

        if fit_all_finally:
            clf = classifier.fit(X, y)

        return clf
