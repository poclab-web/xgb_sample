from required_functions.load_model import load_model
from load_data import data, load_data
import pickle
from analyzer import XGBoostAnalyzer


def xgb_pred_val(csv_path, model_path):
    """ xgboostの回帰タスク

    Args: 
        csv_path: データ入力元のcsvパス
        model_path: モデルのパス pickle形式
    """
    # モデルの読み込み
    model = pickle.load(model_path)

    # x, y定義
    x_true, y_true = load_data(csv_path)
    y_pred = model.predict(x_true) # 予測

    # analyzer
    analyzer = XGBoostAnalyzer(model)
    # r2 score...
    r2 = analyzer.calc_r2Score(y_true, y_pred, to_image=True)

    # importance

