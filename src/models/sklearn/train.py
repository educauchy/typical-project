import warnings
import sys
import logging
from urllib.error import HTTPError
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_curve,\
                            accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from hyperopt import hp, fmin, tpe, Trials
from typing import Dict, Any, List


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


# objective function for hyperopt
def objective(params):
    full_pipeline.set_params(**params)
    cv = KFold(n_splits=config['model']['cv']['folds'], shuffle=True)
    score = cross_val_score(full_pipeline, train_x, train_y, cv=cv, scoring=config['model']['main_metric'], n_jobs=1)
    return score.mean()

def eval_metrics(actual, pred_labels, pred_probs, task_type='classification') -> Dict[str, Any]:
    metrics = {}
    if task_type == 'classification':
        metrics['accuracy'] = accuracy_score(actual, pred_labels)
        metrics['roc_auc'] = roc_auc_score(actual, pred_probs)
        metrics['f1_score'] = f1_score(actual, pred_labels)
        metrics['precision'] = precision_score(actual, pred_labels)
        metrics['recall'] = recall_score(actual, pred_labels)
    elif task_type == 'regression':
        metrics['rmse'] = np.sqrt(mean_squared_error(actual, pred_labels))
        metrics['mae'] = mean_absolute_error(actual, pred_labels)
        metrics['r2'] = r2_score(actual, pred_labels)
    return metrics


# load config
try:
    with open('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)
except yaml.YAMLError as exc:
    logging.error(exc)
    sys.exit(1)
except Exception as e:
    logging.error(e)
    logging.error('Error reading the config file')
    sys.exit(1)


# задайте tracking uri как http://localhost:5555
# создайте try-except конструкция на создание нового-установку существующего эксперимента

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(config['tracking']['random_state'])

    train_url = config['data']['train']
    try:
        data = pd.read_csv(train_url, sep=";")
    except HTTPError as e:
        logger.exception("Unable to download training CSV. Error: %s", e)
        raise Exception("Unable to download training CSV. Error: %s", e)

    train, test = train_test_split(data, test_size=0.2)
    train_x = data.drop(["quality"], axis=1)
    test_x = data.drop(["quality"], axis=1)
    train_y = data[["quality"]]
    test_y = data[["quality"]]

    # здесь создайте новый run и запустите его
    # не забудьте, что запуск нового run - это контекстный менеджер
    # весь код ниже должен быть вложенным в него

    # здесь залогируйте random_state

    pipeline_steps = []

    # NUMERIC STEPS
    numeric_features = train_x.select_dtypes(include=['int64', 'float64']).columns
    numeric_transformer_steps = []

    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    numeric_transformer_steps.append(('imputer', imputer))

    scaler = StandardScaler()
    numeric_transformer_steps.append(('scaler', scaler))

    numeric_transformer = Pipeline(numeric_transformer_steps)

    # CATEGORICAL STEPS
    categorical_features = train_x.select_dtypes(include=['object', 'bool']).columns
    categorical_transformer_steps = []

    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    categorical_transformer_steps.append(('imputer', imputer))

    categorical_transformer = Pipeline(categorical_transformer_steps)

    # JOIN NUMERIC AND CATEGORICAL
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    pipeline_steps.append(('preprocess', preprocessor))

    curr_model = LogisticRegression()
    pipeline_steps.append(('model', curr_model))
    full_pipeline = Pipeline(steps=pipeline_steps)

    # здесь залогируйте параметры: 'num_feature_imputer', 'num_feature_scaler', 'cat_feature_imputer'

    space = {}
    for param, values in config['model']['params_hyperopt'].items():
        if len(values) == 3 and str(values[0]) != values[0]:
            space[f'model__{param}'] = hp.quniform(label=param,
                                                   low=values[0],
                                                   high=values[1],
                                                   q=values[2])
            # здесь залогируйте values[0] значение как {param}_min
            # а values[1] значение как {param}_max
            # таким образом, у вас будут залогированы верхнее и нижнее
            # значения оптимизируемого параметра
        else:
            values = tuple(values)
            space[f'model__{param}'] = hp.choice(label=param, options=values)

    trials = Trials()
    best_params = fmin(objective,
                       space,
                       algo=tpe.suggest,
                       max_evals=config['model']['cv']['evals'],
                       trials=trials)

    # здесь залогируйте параметры: 'cv_folds', 'cv_evals'

    best_params_ = {}
    for key, value in best_params.items():
        if round(value) == value:
            value = int(value)

        best_params_[f'model__{key}'] = value

    best_params_to_log = {}
    for key, value in best_params.items():
        best_params_to_log[f'{key}_optimal'] = value
    # здесь залогируйте словарь с параметрами best_params_to_log

    full_pipeline.set_params(**best_params_)
    full_pipeline.fit(train_x, train_y)

    # Predict
    predicted_labels = full_pipeline.predict(test_x)
    predicted_probs = full_pipeline.predict_proba(test_x)[:, 1]

    quality_metrics = eval_metrics(test_y, predicted_labels, predicted_probs)
    # здесь залогируйте словарь с метриками качествами quality_metrics

    # ROC AUC curve
    run_id = '0' # получите run_id из текущего запуска, замените '0' на нужный код
    roc_curve_path = config['tracking']['FIGURES_PATH'] + f'/roc_auc_{run_id}.png'
    fpr, tpr, thresholds = roc_curve(y_true=test_y, y_score=predicted_probs, pos_label=1)
    plt.plot(fpr, tpr)
    plt.title('ROC-AUC curve (AUC={roc_auc:.2f})'.format(roc_auc=quality_metrics['roc_auc']))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.savefig(roc_curve_path)

    # здесь залогируйте артефакты: roc_curve_path, 'config.yaml', 'MLProject', 'conda_environment.yaml'

    # здесь залогируйте модель с параметрами:
    # - sk_model - обученный пайплайн выше
    # - artifact_path - из конфига, config['tracking']['ARTIFACT_PATH']
    # - registered_model_name - из конфига, config['tracking']['MODEL']
    # - signature = infer_signature(train_x, full_pipeline.predict(train_x))
    # - input_example = train_x.iloc[0].to_dict()
    # - pip_requirements = requirements.txt
    # и присвойте ее переменной lm

    # зарегистрируйте текущую модель в Model Registry с помощью register_model
    # в качестве model_uri параметра используйте model_uri переменную ниже
    run_id = '0' # получите run_id из текущего запуска, замените '0' на нужный код
    artifact_path = '' # получите artifact_path из модели, сохраненной в переменную lm, замените '' на нужный код
    model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id,
                                                        artifact_path=artifact_path)
    # а в качестве имени (name) используйте MODEL_NAME из конфига
