import enum
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from TechnicalAnalysis import get_all_features
from Clients import FtxClient
from Services import BuyService, SellService, TradeExecutionService, LibService
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


class RiskLevel(enum.Enum):
    Low = 1
    Medium = 2
    High = 3


class RiskManager:

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.lib = LibService(api_key=api_key)
        self.execution_service = TradeExecutionService(api_key=self.api_key)

        attrs = vars(self.lib)
        print(json.dumps(attrs['json_data'], indent=4, sort_keys=True))

        self.client = self.lib.get_exchange_client()
        self.balances = None
        self.portfolio_size = None
        self.max_exposure = None
        self.futures = None
        self.futures_prices = None
        self.min_size = 10.00  # 10.00 USD

    def synchronise(self):
        self.lib = LibService(api_key=self.api_key)
        self.futures = self.client.list_futures()
        self.balances = self.client.get_balances()
        self.portfolio_size = self.lib.get_liquid_capital(balances=self.balances)
        self.max_exposure = self.portfolio_size * self.lib.get_leverage()
        print(f'Portfolio size: {self.portfolio_size}')
        print(f'Exposure limit: {self.max_exposure}')

        required_pairs = ['ETH-PERP', 'BTC-PERP']
        trade_pairs = self.lib.get_pairs(side='buy') + self.lib.get_pairs(side='sell')

        # get only the markets that can be traded
        self.futures = [i for i in self.futures if
                        i['name'] in trade_pairs or i['name'] in required_pairs]

        self.futures_prices = {future['name']: future['mark'] for future in self.futures}

    def get_buy_service(self):
        return BuyService(api_key=self.api_key, max_exposure=self.max_exposure, min_size=self.min_size)

    def get_sell_service(self):
        return SellService(api_key=self.api_key, max_exposure=self.max_exposure, min_size=self.min_size)

    def manage_risk(self):
        risk_level: RiskLevel = self.prep_risk_evaluation()

        if risk_level == RiskLevel.Low:
            self.lib.set_risk_level(risk_level='high')
            print(f'High market risk detected.')

            self.lib.set_aggressive_buy(aggressive_buy_enabled=False)
            print(f'Aggressive buy disabled')

            self.lib.set_aggressive_sell(aggressive_sell_enabled=True)
            print(f'Aggressive sell enabled')

        if risk_level == RiskLevel.Medium:
            self.lib.set_risk_level(risk_level='medium')
            print(f'Normal market risk.')

            self.lib.set_aggressive_buy(aggressive_buy_enabled=False)
            print(f'Aggressive buy enabled')

            self.lib.set_aggressive_sell(aggressive_sell_enabled=False)
            print(f'Aggressive sell disabled')

        if risk_level == RiskLevel.High:
            self.lib.set_risk_level(risk_level='low')
            print(f'Low market risk,')

            self.lib.set_aggressive_buy(aggressive_buy_enabled=True)
            print(f'Aggressive buy enabled')

            self.lib.set_aggressive_sell(aggressive_sell_enabled=False)
            print(f'Aggressive sell disabled')

    def prep_risk_evaluation(self) -> RiskLevel:
        btc_mark_price = self.futures_prices['BTC-PERP']
        prediction_1hr = self.evaluate_market_risk(resolution=3600)
        prediction_4hr = self.evaluate_market_risk(resolution=14400)

        is_1hr_bullish = prediction_1hr['median'] > btc_mark_price
        is_4hr_bullish = prediction_4hr['median'] > btc_mark_price

        self.lib.write_to_csv(prediction=prediction_1hr['median'], timestamp=datetime.now(tz=None), horizon=3600)
        self.lib.write_to_csv(prediction=prediction_4hr['median'], timestamp=datetime.now(tz=None), horizon=14400)

        if is_1hr_bullish and is_4hr_bullish:
            return RiskLevel.Low
        elif is_1hr_bullish or is_4hr_bullish:
            return RiskLevel.Medium
        else:
            return RiskLevel.High

    # noinspection PyTypeChecker
    def evaluate_market_risk(self, resolution: int) -> dict:

        print(f'attempting to evaluate risk for the following timeframe: {resolution}')

        price_matrix = {}

        for pair in self.futures:
            pair_symbol = pair['name']
            if pair_symbol not in price_matrix:
                price_matrix[pair_symbol] = self.lib.get_prices(client=self.client, market=pair_symbol,
                                                                resolution=resolution)

        df_training: pd.DataFrame = None
        btc_predict_df: pd.DataFrame = None
        btc_current_close = 0
        btc_min = 9999999
        btc_max = 0
        look_ahead = 50

        for market, price_series in price_matrix.items():

            df: pd.DataFrame = get_all_features(prices=price_series)

            if market == 'BTC-PERP':
                # we get the second to last candle since the last candle may not have closed
                btc_current_close = df['close'].values[-2]
                btc_max = df.max()['close']
                btc_min = df.min()['close']

                # get the last 50 candles
                btc_predict_df = df.iloc[-1 * look_ahead:]
                btc_predict_df = btc_predict_df.drop(['startTime', 'time'], axis=1)
                btc_predict_df = btc_predict_df.dropna()

            df['futureClose'] = df['close'].shift(-1 * look_ahead)

            # drop first and last 50 rows from df to remove zeros from average calculations, etc.
            df = df.iloc[look_ahead:]
            df = df.iloc[:-1 * look_ahead]
            df = df.drop(['startTime', 'time'], axis=1)

            # min-max normalisation
            normalized_df: pd.DataFrame = (df - df.min()) / (df.max() - df.min())
            if df_training is None:
                df_training = normalized_df
            else:
                df_training = df_training.append(normalized_df, ignore_index=True)

        df_training = df_training.dropna()
        X_train, X_test, y_train, y_test = train_test_split(df_training.drop(['futureClose'], axis=1),
                                                            df_training.futureClose, train_size=0.98)

        clf = GradientBoostingRegressor(learning_rate=0.2, n_estimators=1000)
        clf.fit(X_train, y_train)

        error_rate = self.get_performance(clf=clf, x_test=X_test, y_test=y_test)
        print(f'Model error rate = {error_rate}')
        print(f'Sample size of = {len(df_training.index)}')
        if len(btc_predict_df.index) <= 1:
            print(f'Warning prediction array length is {len(btc_predict_df.index)}')
        print(f'BTC candles predict into future {len(btc_predict_df.index)}')

        # self.plot_performance(clf=clf, x_test=X_test, y_test=y_test)

        btc_predict_normalized_df = (btc_predict_df - btc_min) / (btc_max - btc_min)
        btc_predictions: np.ndarray = clf.predict(btc_predict_normalized_df)

        max_prediction = (np.max(btc_predictions) * (btc_max - btc_min)) + btc_min
        min_prediction = (np.min(btc_predictions) * (btc_max - btc_min)) + btc_min
        median_prediction = (np.median(btc_predictions) * (btc_max - btc_min)) + btc_min
        print(
            f'max prediction: {max_prediction}\n'
            f'min prediction: {min_prediction}\n'
            f'median prediction: {median_prediction}')

        if btc_max < btc_min:
            raise Exception(f'Error when in defining BTC price. btc_max = {btc_max}, btc_min = {btc_min}')
        if btc_current_close == 0:
            raise Exception(f'BTC price invalid. btc_current_close = {btc_current_close}')

        return {
            'upper_bound': max_prediction,
            'lower_bound': min_prediction,
            'median': median_prediction
        }

    @staticmethod
    def get_performance(clf, x_test, y_test):
        predicted = clf.predict(x_test)
        expected = y_test
        return np.sqrt(np.mean((predicted - expected) ** 2))

    @staticmethod
    def plot_performance(clf, x_test, y_test, title):
        # for testing
        predicted = clf.predict(x_test)
        expected = y_test

        plt.figure(figsize=(4, 3))
        plt.title(title)
        plt.scatter(expected, predicted)
        plt.plot([0, 1], [0, 1], '--k')
        plt.axis('tight')
        plt.xlabel('True price')
        plt.ylabel('Predicted price')
        plt.tight_layout()
        print("RMS: %r " % np.sqrt(np.mean((predicted - expected) ** 2)))

        plt.show()
