from load_data import load_data
import pickle
from analyzer import XGBoostAnalyzer
from setting import setting


def xgb_pred_val(csv_path, model_path):
    """ xgboostの回帰タスク

    Args: 
        csv_path: データ入力元のcsvパス
        model_path: モデルのパス pickle形式
    """
    # モデルの読み込み
    f = open(model_path, 'rb')
    model = pickle.load(f)

    x_range, y_column = setting['x_range'], setting["y_column"]

    # x, y定義
    x_true, y_true = load_data(csv_path, x_range, y_column)
    y_pred = model.predict(x_true) # 予測

    # analyzer
    analyzer = XGBoostAnalyzer(model)
    # r2 score...
    r2 = analyzer.calc_r2Score(y_true, y_pred, to_image=True)

    print(r2)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, help='csv data of test data')
    parser.add_argument('--model_path', type=str, help='path of model')

    args = parser.parse_args()

    csv_path = args.csv_path
    model_path = args.model_path
    xgb_pred_val(csv_path, model_path)
    