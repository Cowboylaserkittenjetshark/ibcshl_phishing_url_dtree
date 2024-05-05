from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from joblib import parallel_backend
from common import X, y
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

def feature_importance(model):
    def sortSecond(val):
        return val[1]

    values = model.feature_importances_
    features = list(X)
    importances = [(features[i], values[i]) for i in range(len(features))]
    importances.sort(reverse=True, key=sortSecond)
    return importances

with parallel_backend('threading', n_jobs=-1):
    logging.debug('Splitting data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
    test_size=0.2)

    logging.info('==== Decision Tree ====')
    dt_clf = DecisionTreeClassifier().fit(X_train, y_train)

    dt_score = dt_clf.score(X_valid, y_valid)
    dt_cross_val_score = cross_val_score(dt_clf, X_train, y_train, cv=7)
    dt_report = classification_report(y_test, dt_clf.predict(X_test))
    dt_feat_importance = feature_importance(dt_clf)

    logging.info('Score: %s\n', dt_score)
    logging.info('Cross val score: %s\n', dt_cross_val_score)
    logging.info('Classification report:\n%s\n', dt_report)
    logging.debug('Feature importance: %s', str(dt_feat_importance))

    logging.info('==== Decision Tree with feature selection')
    X_train_cut = X_train[[col[0] for col in dt_feat_importance[:15]]]
    X_test_cut  = X_test[[col[0] for col in dt_feat_importance[:15]]]
    X_valid_cut = X_valid[[col[0] for col in dt_feat_importance[:15]]]

    dt_clf = DecisionTreeClassifier().fit(X_train_cut, y_train)

    dt_score = dt_clf.score(X_valid_cut, y_valid)
    dt_cross_val_score = cross_val_score(dt_clf, X_train_cut, y_train, cv=7)
    dt_report = classification_report(y_test, dt_clf.predict(X_test_cut))

    logging.info('Score: %s\n', dt_score)
    logging.info('Cross val score: %s\n', dt_cross_val_score)
    logging.info('Classification report:\n%s\n', dt_report)

    logging.info('==== Random Forest ====')
    rf_clf = RandomForestClassifier().fit(X_train, y_train)

    rf_score = rf_clf.score(X_valid, y_valid)
    rf_cross_val_score = cross_val_score(rf_clf, X_train, y_train, cv=7)
    rf_report = classification_report(y_valid, rf_clf.predict(X_valid))
    rf_feat_importance = feature_importance(rf_clf)

    logging.info('Score: %s\n', rf_score)
    logging.info('Cross val score: %s\n', rf_cross_val_score)
    logging.info('Classification report:\n%s\n', rf_report)
    logging.debug('Feature importance: %s', str(rf_feat_importance))

    logging.info('==== Random Forest with feature selection ====')
    X_train_cut = X_train[[col[0] for col in rf_feat_importance[:15]]]
    X_test_cut  = X_test[[col[0] for col in rf_feat_importance[:15]]]
    X_valid_cut = X_valid[[col[0] for col in rf_feat_importance[:15]]]

    rf_clf = RandomForestClassifier().fit(X_train_cut, y_train)

    rf_score = rf_clf.score(X_valid_cut, y_valid)
    rf_cross_val_score = cross_val_score(rf_clf, X_train_cut, y_train, cv=7)
    rf_report = classification_report(y_valid, rf_clf.predict(X_valid_cut))

    logging.info('Score: %s\n', rf_score)
    logging.info('Cross val score: %s\n', rf_cross_val_score)
    logging.info('Classification report:\n%s\n', rf_report)

    # logging.info('==== Histogram Gradient Boosting ====')
    # hgb_clf = HistGradientBoostingClassifier().fit(X_train, y_train)

    # hgb_score = hgb_clf.score(X_valid, y_valid)
    # hgb_cross_val_score = cross_val_score(hgb_clf, X_train, y_train, cv=7)
    # hgb_report = classification_report(y_test, hgb_clf.predict(X_test))

    # logging.info('Score: %s\n', hgb_score)
    # logging.info('Cross val score: %s\n', hgb_cross_val_score)
    # logging.info('Classification report:\n%s\n', hgb_report)
