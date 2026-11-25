"""Train on the training partition only; reserve test data for evaluation."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from src.data import ROOT, FEATURES

SEED = 42


def split_data(df):
    return train_test_split(df[FEATURES], df.PE, test_size=.2, random_state=SEED)


def candidates():
    estimators = {'Linear Regression': LinearRegression(), 'Ridge': Ridge(),
                  'Random Forest': RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1),
                  'Gradient Boosting': GradientBoostingRegressor(random_state=SEED)}
    return {name: Pipeline([('scale', StandardScaler()), ('model', model)])
            for name, model in estimators.items()}


def scores(y, pred):
    return {'R2': r2_score(y, pred), 'RMSE': float(np.sqrt(mean_squared_error(y, pred))),
            'MAE': mean_absolute_error(y, pred)}


def compare(split):
    x, xt, y, yt = split
    cv = KFold(5, shuffle=True, random_state=SEED)
    rows = []
    for name, model in candidates().items():
        errors = -cross_val_score(model, x, y, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=2)
        model.fit(x, y)
        rows.append({'Model': name, 'CV RMSE': errors.mean(), 'CV SD': errors.std(),
                     **scores(yt, model.predict(xt))})
    return pd.DataFrame(rows).sort_values('CV RMSE').reset_index(drop=True)


def tune(split, name):
    x, xt, y, yt = split
    grids = {'Linear Regression': {'model__fit_intercept': [True, False]},
             'Ridge': {'model__alpha': [.1, 1, 10]},
             'Random Forest': {'model__n_estimators': [100, 160], 'model__min_samples_leaf': [1, 2]},
             'Gradient Boosting': {'model__n_estimators': [100, 200], 'model__max_depth': [3, 4]}}
    search = GridSearchCV(candidates()[name], grids[name], cv=KFold(5, shuffle=True, random_state=SEED),
                          scoring='neg_root_mean_squared_error', n_jobs=2)
    search.fit(x, y)
    return search


def diagnostics(split, model):
    _, xt, _, yt = split
    pred = model.predict(xt)
    perm = permutation_importance(model, xt, yt, scoring='neg_root_mean_squared_error', n_repeats=8, random_state=SEED)
    estimator = model.named_steps['model']
    native = estimator.feature_importances_ if hasattr(estimator, 'feature_importances_') else np.abs(estimator.coef_)
    imp = pd.DataFrame({'Feature': FEATURES, 'Permutation RMSE increase': perm.importances_mean,
                        'Native importance': native})
    pairs = pd.DataFrame({'Actual': yt.to_numpy(), 'Predicted': pred, 'Residual': yt.to_numpy()-pred})
    return imp, pairs


def save_results(df, comparison, search, name, imp, pairs):
    folder = ROOT / 'models'
    folder.mkdir(exist_ok=True)
    joblib.dump(search.best_estimator_, folder / 'best_model.joblib', compress=3)
    comparison.to_csv(folder / 'comparison.csv', index=False)
    imp.to_csv(folder / 'importance.csv', index=False)
    pairs.to_csv(folder / 'predictions.csv', index=False)
    result = {'model': name, 'parameters': search.best_params_, 'CV RMSE': -search.best_score_,
              **scores(pairs.Actual, pairs.Predicted)}
    (folder / 'metrics.json').write_text(json.dumps(result, indent=2))
    return result
