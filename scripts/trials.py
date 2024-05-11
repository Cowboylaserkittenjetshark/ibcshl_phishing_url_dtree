import logging
from joblib import parallel_backend
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, cross_val_score, HalvingRandomSearchCV
from sklearn.metrics import classification_report, accuracy_score
from common import fetched_features

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
    cv_score = cross_val_score(clf, X_train, y_train, cv=7)
    report = classification_report(y_valid, clf.predict(X_valid))
    test_report = classification_report(y_test, clf.predict(X_test))
    test_accuracy = accuracy_score(y_test, clf.predict(X_test))

    logging.info('Score: %s\n', score)
    logging.info('Cross val score: %s\n', cv_score)
    logging.info('Classification report:\n%s\n', report)
    logging.info('Test classification report:\n%s\n', test_report)
    logging.info('Test accuracy: %s\n', test_accuracy)

def __build_header(name, param_grid, with_fetched):
    options = " ("
    if with_fetched:
        options = options + "With fetched features"
    else:
        options = options + "Local features"
    if param_grid:
        options = options + ", tuned"
    else:
        options = options + ", untuned"
    options = options + ')'

    return name + options

def run_trial(name, classifier, X, y, params=None, with_fetched=False):
    with parallel_backend('threading', n_jobs=-1):
        header = __build_header(name, params, with_fetched)
        logging.info('==== %s ====', header)

        if not with_fetched:
            logging.debug('Dropping fetched features')
            X = X.drop(fetched_features, axis = 1)
            
        logging.debug('Splitting data')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)

        clf = None
        if params:
            clf = HalvingRandomSearchCV(classifier, params, verbose=1).fit(X_train, y_train)
        else:
            clf = classifier.fit(X_train, y_train)

        if hasattr(clf, 'feature_importances_'):
            feat_importance = __feature_importance(clf, X)
            logging.debug('Feature importance: %s', str(feat_importance))

        __score_model(clf, X_train, y_train, X_valid, y_valid, X_test, y_test)

        return clf
