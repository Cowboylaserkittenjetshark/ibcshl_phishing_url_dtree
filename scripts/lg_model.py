from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, cross_val_score, HalvingGridSearchCV
from sklearn.metrics import classification_report
from joblib import parallel_backend
from common import X, y, fetched_features
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(message)s'
)

with parallel_backend('threading', n_jobs=-1):
    logging.debug('Splitting data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
    test_size=0.2)

    logging.info('==== Logistic Regression ====')
    lg_clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=42)).fit(X_train, y_train)

    lg_score = lg_clf.score(X_valid, y_valid)
    lg_cross_val_score = cross_val_score(lg_clf, X_train, y_train, cv=7)
    lg_report = classification_report(y_valid, lg_clf.predict(X_valid))

    logging.info('Score: %s\n', lg_score)
    logging.info('Cross val score: %s\n', lg_cross_val_score)
    logging.info('Classification report:\n%s\n', lg_report)

    lg_test_report = classification_report(y_valid, lg_clf.predict(X_valid))
    logging.info('Test classification report:\n%s\n', lg_report)
    
    logging.info('==== Logistic Regression without fetched features ====')
    X_train = X_train.drop(fetched_features, axis = 1)
    X_test = X_test.drop(fetched_features, axis = 1)
    X_valid = X_valid.drop(fetched_features, axis = 1)
    lg_clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=42)).fit(X_train, y_train)

    lg_score = lg_clf.score(X_valid, y_valid)
    lg_cross_val_score = cross_val_score(lg_clf, X_train, y_train, cv=7)
    lg_report = classification_report(y_valid, lg_clf.predict(X_valid))

    logging.info('Score: %s\n', lg_score)
    logging.info('Cross val score: %s\n', lg_cross_val_score)
    logging.info('Classification report:\n%s\n', lg_report)

    lg_test_report = classification_report(y_valid, lg_clf.predict(X_valid))
    logging.info('Test classification report:\n%s\n', lg_report)
