import json
import random
import time
from datetime import datetime
from typing import List

from Clients import BaseClient, FtxClient, KrakenClient, Exchange
from TechnicalAnalysis import oversold, overbought, get_rsi


class Trade:
    def __init__(self, pair, target, stoploss, side, order_type, size, price, delete):
        self.pair = pair
        self.target = target
        self.stoploss = stoploss
        self.side = side
        self.type = order_type
        self.size = size
        self.price = price
        self.delete = delete


class BaseService:
    _type = 'limit'

    def __init__(self, api_key: str, max_exposure: float, min_size: float):
        self.api_key = api_key
        self.lib = LibService(api_key=self.api_key)
        self.execution_service = TradeExecutionService(api_key=self.api_key)

        self.exchange = self.lib.exchange
        self.client = self.lib.get_exchange_client()

        self.buy_side = 'buy'
        self.sell_side = 'sell'
        self.max_exposure = max_exposure
        self.min_size = min_size
        self.futures = None

    def create_trade(self, pair: str, prices: List[dict], side: str) -> Trade:
        mark_price: float = self.lib.get_latest_price(pair=pair, futures=self.get_futures())
        current_price: float = TradeExecutionService.adjust_price(price=mark_price, side=side)
        stop_loss: float = self.lib.get_stop(prices=prices, side=side)['close']
        target: float = self.lib.calc_target(current=current_price, stop=stop_loss)
        position_size = self.lib.calc_kelly(current=current_price, stop=stop_loss, target=target)
        size: float = (position_size * self.max_exposure) / (current_price * len(self.lib.get_pairs(side=self.buy_side)))
        min_position_size: float = self.min_size / current_price

        if size < min_position_size:
            size = min_position_size

        print(f'Attempting to {side} {size} of {pair} at {current_price}')

        return Trade(pair=pair, target=target, stoploss=stop_loss, side=side,
                     order_type=self._type, size=size, price=current_price,
                     delete=False)

    # this method is primarily here to save on API calls
    def get_futures(self):
        if self.futures is None:
            self.futures = self.lib.get_futures(client=self.client)
        return self.futures


class BuyService(BaseService):

    def __init__(self, api_key: str, max_exposure: float, min_size: float):
        super().__init__(api_key=api_key, max_exposure=max_exposure, min_size=min_size)

    def generate_trades(self) -> List[Trade]:

        trades = []
        markets = self.lib.get_pairs(side=self.buy_side)

        # if pyramiding is not enabled then cancel orders
        if not self.lib.pyramiding_enabled:
            self.execution_service.cancel_open_orders_for_markets(markets=markets)

        open_orders: list = self.client.get_open_orders()
        max_orders = self.lib.max_open_orders

        while len(open_orders) > max_orders:
            print(f'{len(open_orders)} orders reached out of a maximum of {max_orders}')
            cancelled_order = self.execution_service.cancel_oldest_order(orders=open_orders)
            open_orders.remove(cancelled_order)

            # wait 5 minutes for orders to execute before continuing
            LibService.patiently_wait(cycle_time=5 * 60, interval=1)
            open_orders = self.client.get_open_orders()

        positions = self.client.get_positions(show_avg_price=True)
        net_exposure = LibService.get_net_exposure(positions=positions)
        total_exposure = LibService.get_total_exposure(positions=positions)

        print(f'Net exposure: {net_exposure}')
        print(f'Total exposure: {total_exposure}')

        # if the exposure is too high then skip
        if total_exposure > self.max_exposure:
            print(
                f'Buy Service: Skipping. Current exposure of {total_exposure} '
                f'out of max exposure limit of {self.max_exposure}')
            return trades

        for pair in markets:
            prices: List[dict] = self.lib.get_prices(client=self.client, market=pair)

            rsi = get_rsi(prices)

            # if aggressive buying is enabled check RSI at only 40
            if self.lib.aggressive_buy_enabled:
                rsi_threshold = 40
            else:
                rsi_threshold = 30

            if not oversold(series=rsi, threshold=rsi_threshold):
                print(f'Cannot buy {pair} due to RSI too high at {rsi.values[-1]}')
                continue

            if not self.lib.pyramiding_enabled and net_exposure > 0:
                print(f'Net exposure {net_exposure}. Cannot buy {pair}')
                continue

            trades.append(self.create_trade(pair=pair, prices=prices, side=self.buy_side))

        return trades


class SellService(BaseService):

    def __init__(self, api_key: str, max_exposure: float, min_size: float):
        super().__init__(api_key, max_exposure, min_size)

    def generate_trades(self) -> List[Trade]:

        trades = []
        futures = self.get_futures()
        open_positions = self.client.get_positions(show_avg_price=True)
        open_positions = [x for x in open_positions if x['netSize'] != 0]
        net_exposure = LibService.get_net_exposure(positions=open_positions)
        open_orders = self.client.get_open_orders()

        resolutions = [60, 300, 900, 3600, 14400, 86400, 259200]
        resolution = resolutions[resolutions.index(self.lib.default_resolution) + 1]

        # skip if the net exposure is less than 1x leverage
        if net_exposure < self.max_exposure / self.lib.get_leverage():
            print(f'Sell Service: Skipping. Exposure too low')
            return trades

        for position in open_positions:
            pair = position['future']
            open_sells: List[dict] = [x for x in open_orders if x['market'] == pair and x['side'] == self.sell_side]
            cumulative_sell_size = sum([x['remainingSize'] for x in open_sells])
            prices: List[dict] = self.lib.get_prices(client=self.client, market=pair, resolution=resolution)
            break_even_price = position['recentBreakEvenPrice']
            recent_avg_open_price = position['recentAverageOpenPrice']
            mark_price = self.lib.get_latest_price(pair=pair, futures=futures)

            rsi = get_rsi(prices)

            if position['netSize'] < cumulative_sell_size:
                print(f'Cannot reduce size. Cumulative sell size greater than position size')
                continue

            if mark_price < break_even_price:
                print(f'Unable to reduce size. Position not in profit.')
                continue

            if mark_price < recent_avg_open_price:
                print(f'Unable to reduce size: Mark price less than recent opens')
                continue

            if self.lib.aggressive_sell_enabled:
                rsi_threshold = 60
            else:
                rsi_threshold = 70

            if not overbought(series=rsi, threshold=rsi_threshold):
                print(f'Cannot sell {pair} due to RSI too low at {rsi.values[-1]}')
                continue

            trades.append(self.create_trade(pair=pair, prices=prices, side=self.sell_side))

        return trades


class TradeExecutionService:

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.lib = LibService(api_key=api_key)
        self.client = self.lib.get_exchange_client()

    def execute_trades(self, trades: List[Trade]):
        for trade in trades:
            self.execute_trade(pair=trade.pair, side=trade.side,
                               trade_type=trade.type, size=trade.size, price=trade.price)
            # wait 1 second before proceeding to avoid blowing the rate limit
            LibService.patiently_wait(cycle_time=1, interval=1, logging=False)

    def execute_trade(self, pair: str, side: str = 'buy',
                      trade_type: str = 'limit', size: float = None,
                      price: float = None):

        try:
            # execute the order in ftx
            self.client.place_order(market=pair,
                                    side=side,
                                    price=price,
                                    type=trade_type,
                                    size=size)
        except Exception as e:
            print(e)

    def update_stop(self, pair: str, stop: float):
        self.lib.set_stop(pair=pair, stop=stop)

    def cancel_random_order(self, orders: list, max_orders: int):
        order_to_cancel = orders[random.randint(0, max_orders - 1)]
        self.client.cancel_order(str(order_to_cancel['id']))
        return order_to_cancel

    def cancel_oldest_order(self, orders: list):
        orders = sorted(orders, key=lambda k: k['id'], reverse=False)
        order_to_cancel = orders[0]
        self.client.cancel_order(str(order_to_cancel['id']))
        return order_to_cancel

    def cancel_open_orders_for_markets(self, markets: list):
        for pair in markets:
            self.client.cancel_orders(market_name=pair, limit_orders=True)

    @staticmethod
    def adjust_price(price: float, side: str):
        if side == 'buy':
            return price * 0.995

        if side == 'sell':
            return price * 1.005


class LibService:

    def __init__(self, api_key: str = None):
        with open('config.json') as config_file:
            self.json_data = json.load(config_file)
            self.api_key = api_key
            self.api_secret = self.json_data[self.api_key]['api-secret']
            self.subaccount = self.json_data[self.api_key]['subaccount']
            self.strategies = self.json_data[self.api_key]['strategy']
            self.staked_tokens = self.json_data[self.api_key]['staked']
            self.leverage = self.json_data[self.api_key]['leverage']
            self.pyramiding_enabled = self.json_data[self.api_key]['pyramiding_enabled']
            self.aggressive_buy_enabled = self.json_data[self.api_key]['aggressive_buy_enabled']
            self.aggressive_sell_enabled = self.json_data[self.api_key]['aggressive_sell_enabled']
            self.max_open_orders = self.json_data[self.api_key]['max_open_orders']
            self.default_resolution = self.json_data[self.api_key]['default_resolution']
            self.cycle_time = self.json_data[self.api_key]['cycle_time']
            self.risk_level = self.json_data[self.api_key]['risk_level']
            self.exchange = str(self.json_data[self.api_key]['exchange']).upper()
            self.del_queue = []

    def get_leverage(self):
        return self.leverage

    def get_positions(self) -> dict:
        return self.json_data[self.api_key]['trades']

    def get_subaccount_name(self) -> str:
        return self.subaccount

    def get_api_secret(self) -> str:
        return self.api_secret

    def get_staked_tokens(self) -> dict:
        return self.staked_tokens

    def get_stop(self, prices: List[dict], side: str) -> dict:
        if side == 'buy':
            return self.get_lowest(prices=prices)

        if side == 'sell':
            return self.get_highest(prices=prices)

    def get_pairs(self, side: str = 'buy'):
        return self.strategies[side]

    # the client is meant to be an exchange cli e.g. FtxClient
    def get_prices(self, client, market: str, resolution: int = None) -> List[dict]:

        if resolution is None:
            resolution = self.default_resolution
        prices = client.get_historical_prices(market=market, resolution=resolution)

        # wait 1 second before proceeding to avoid blowing the rate limit
        LibService.patiently_wait(cycle_time=1, interval=1, logging=False)

        return prices

    def get_liquid_capital(self, balances: list) -> float:
        total_capital = 0

        if not balances:
            raise Exception('Balances cannot be empty')

        for balance in balances:
            if balance['total'] == 0:
                continue
            total_capital += balance['usdValue']

        return total_capital - self.get_staked_amount(balances=balances)

    def get_staked_amount(self, balances: list) -> float:
        total = 0
        estimated_prices = self.get_estimated_prices(balances=balances)
        for pair, quantity in self.get_staked_tokens().items():
            if pair in estimated_prices:
                total += estimated_prices[pair] * quantity
        return total

    def set_risk_level(self, risk_level: str = 'medium'):
        self.json_data[self.api_key]['risk_level'] = risk_level
        self.write_to_file(self.json_data)

    def set_aggressive_buy(self, aggressive_buy_enabled: bool = False):
        self.json_data[self.api_key]['aggressive_buy_enabled'] = aggressive_buy_enabled
        self.write_to_file(self.json_data)

    def set_aggressive_sell(self, aggressive_sell_enabled: bool = False):
        self.json_data[self.api_key]['aggressive_sell_enabled'] = aggressive_sell_enabled
        self.write_to_file(self.json_data)

    def set_enable_pyramiding(self, pyramiding_enabled: bool = False):
        self.json_data[self.api_key]['pyramiding_enabled'] = pyramiding_enabled
        self.write_to_file(self.json_data)

    def set_default_resolution(self, resolution: int):
        self.json_data[self.api_key]['default_resolution'] = resolution
        self.write_to_file(self.json_data)

    def set_cycle_time(self, cycle_time: int):
        self.json_data[self.api_key]['cycle_time'] = cycle_time
        self.write_to_file(self.json_data)

    def set_stop(self, pair: str, stop: float):
        self.json_data[self.api_key]['trades'][pair]['stop'] = stop
        self.write_to_file(self.json_data)

    def set_trade(self, pair: str, exposure: float, side: str, target: float, stop: float):
        self.json_data[self.api_key]['trades'][pair] = {
            "exposure": exposure,
            "side": side,
            "target": target,
            "stop": stop
        }

        self.write_to_file(self.json_data)

    def calc_kelly(self, current: float, stop: float, target: float, probability: float = 0.51) -> float:
        return self._calc_kelly(probability, (target - current) / (current - stop))

    def get_exchange_client(self) -> BaseClient:
        if self.exchange == Exchange.FtxClient.value:
            return FtxClient(api_key=self.api_key,
                             api_secret=self.get_api_secret(),
                             subaccount_name=self.get_subaccount_name())

        if self.exchange == Exchange.KrakenClient.value:
            return KrakenClient(api_key=self.api_key,
                                api_secret=self.get_api_secret())

    @staticmethod
    def _calc_kelly(p: float = 0.0, b: float = 1) -> float:
        return p + ((p - 1) / b)

    @staticmethod
    def calc_target(current: float, stop: float) -> float:
        return current + (current - stop)

    @staticmethod
    def get_highest(prices: List[dict] = None) -> dict:
        return max(prices, key=lambda x: x['close'])

    @staticmethod
    def get_lowest(prices: List[dict] = None) -> dict:
        return min(prices, key=lambda x: x['close'])

    # the client is meant to be an exchange cli e.g. FtxClient
    @staticmethod
    def get_futures(client):
        return client.list_futures()

    @staticmethod
    def get_latest_price(pair: str, futures: dict) -> float:
        future_data = [i for i in futures if i['name'] == pair]

        if not future_data:
            return -1

        return future_data[0]['last']

    @staticmethod
    def get_estimated_prices(balances: list) -> dict:
        estimated_prices = {}
        for balance in balances:
            ticker = balance['coin']
            usd_value = balance['usdValue']
            total = balance['total']
            if total == 0:
                continue
            estimated_prices[ticker] = usd_value / total
        return estimated_prices

    @staticmethod
    def write_to_csv(prediction: float, timestamp: datetime, horizon: int) -> None:
        with open("predictions.csv", 'a') as predictions:
            predictions.write(f'{prediction},{timestamp},{horizon}\n')

    @staticmethod
    def write_to_file(data):
        with open('config.json', 'w') as json_file:
            json.dump(data, json_file)

    # cycle time in seconds
    @staticmethod
    def patiently_wait(cycle_time: int, interval: int, logging: bool = True):
        for i in range(0, cycle_time, interval):
            if logging:
                print(f'{i} seconds complete out of {cycle_time}', end="\r", flush=True)
            time.sleep(interval)

        if logging:
            print(f'{cycle_time} seconds complete out of {cycle_time} seconds')

    @staticmethod
    def get_net_exposure(positions: list) -> float:
        return sum(x['cost'] for x in positions)

    @staticmethod
    def get_total_exposure(positions: list) -> float:
        return sum(abs(x['cost']) for x in positions)
