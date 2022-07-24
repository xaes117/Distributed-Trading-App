import enum
import urllib.parse
from typing import Optional, Dict, Any, List
import time
import base64
import hashlib
import urllib.request

import requests
from requests import Request, Session, Response
import hmac
from ciso8601 import parse_datetime


class Exchange(enum.Enum):
    FtxClient = 'FTX'
    KrakenClient = 'KRAKEN'


class BaseClient:
    _ENDPOINT = 'https://ftx.com/api/'

    def __init__(self) -> None:
        self._session = Session()
        self._api_key = 'zGrqo8xm5joiIXI04YpCLUufuF_yRGQ_y8y5tRuE'
        self._api_secret = 'iwjVf1OjFkqmwhEAi35ksEkApimTwSJBc5p1_qbt'
        self._subaccount_name = 'read-only'

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request('GET', path, params=params)

    def _post(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request('POST', path, json=params)

    def _delete(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request('DELETE', path, json=params)

    def _request(self, method: str, path: str, **kwargs) -> Any:
        request = Request(method, self._ENDPOINT + path, **kwargs)
        self._sign_request(request)
        response = self._session.send(request.prepare())
        return self._process_response(response)

    def _sign_request(self, request: Request) -> None:
        ts = int(time.time() * 1000)
        prepared = request.prepare()
        signature_payload = f'{ts}{prepared.method}{prepared.path_url}'.encode()
        if prepared.body:
            signature_payload += prepared.body
        signature = hmac.new(self._api_secret.encode(), signature_payload, 'sha256').hexdigest()
        request.headers['FTX-KEY'] = self._api_key
        request.headers['FTX-SIGN'] = signature
        request.headers['FTX-TS'] = str(ts)
        if self._subaccount_name:
            request.headers['FTX-SUBACCOUNT'] = urllib.parse.quote(self._subaccount_name)

    def _process_response(self, response: Response) -> Any:
        try:
            data = response.json()
        except ValueError:
            response.raise_for_status()
            raise
        else:
            if not data['success']:
                raise Exception(data['error'])
            return data['result']

    def list_futures(self) -> List[dict]:
        return self._get('futures')

    def list_markets(self) -> List[dict]:
        return self._get('markets')

    def get_orderbook(self, market: str, depth: int = None) -> dict:
        return self._get(f'markets/{market}/orderbook', {'depth': depth})

    def get_historical_prices(self, market: str = None, resolution: float = None) -> List[dict]:
        return self._get(f'markets/{market}/candles', {
            'resolution': resolution
        })

    def get_trades(self, market: str) -> dict:
        pass

    def get_account_info(self) -> dict:
        pass

    def get_open_orders(self, market: str = None) -> List[dict]:
        pass

    def get_order_history(self, market: str = None, side: str = None, order_type: str = None,
                          start_time: float = None,
                          end_time: float = None) -> List[dict]:
        pass

    def get_conditional_order_history(self, market: str = None, side: str = None, type: str = None,
                                      order_type: str = None, start_time: float = None, end_time: float = None) -> List[
        dict]:
        pass

    def modify_order(
            self, existing_order_id: Optional[str] = None,
            existing_client_order_id: Optional[str] = None, price: Optional[float] = None,
            size: Optional[float] = None, client_order_id: Optional[str] = None,
    ) -> dict:
        pass

    def get_conditional_orders(self, market: str = None) -> List[dict]:
        pass

    def place_order(self, market: str, side: str, price: float, size: float, type: str = 'limit',
                    reduce_only: bool = False, ioc: bool = False, post_only: bool = False,
                    client_id: str = None) -> dict:
        pass

    def place_conditional_order(
            self, market: str, side: str, size: float, type: str = 'stop',
            limit_price: float = None, reduce_only: bool = False, cancel: bool = True,
            trigger_price: float = None, trail_value: float = None
    ) -> dict:
        pass

    def cancel_order(self, order_id: str) -> dict:
        pass

    def cancel_orders(self, market_name: str = None, conditional_orders: bool = False,
                      limit_orders: bool = False) -> dict:
        pass

    def get_fills(self) -> List[dict]:
        pass

    def get_balances(self) -> List[dict]:
        pass

    def get_deposit_address(self, ticker: str) -> dict:
        pass

    def get_positions(self, show_avg_price: bool = False) -> List[dict]:
        pass

    def get_position(self, name: str, show_avg_price: bool = False) -> dict:
        pass

    def get_all_trades(self, market: str, start_time: float = None, end_time: float = None) -> List:
        pass


class FtxClient(BaseClient):
    _ENDPOINT = 'https://ftx.com/api/'

    def __init__(self, api_key=None, api_secret=None, subaccount_name=None) -> None:
        super().__init__()
        self._session = Session()
        self._api_key = api_key
        self._api_secret = api_secret
        self._subaccount_name = subaccount_name

    def get_trades(self, market: str) -> dict:
        return self._get(f'markets/{market}/trades')

    def get_account_info(self) -> dict:
        return self._get(f'account')

    def get_open_orders(self, market: str = None) -> List[dict]:
        return self._get(f'orders', {'market': market})

    def get_order_history(self, market: str = None, side: str = None, order_type: str = None,
                          start_time: float = None,
                          end_time: float = None) -> List[dict]:
        return self._get(f'orders/history',
                         {'market': market, 'side': side, 'orderType': order_type, 'start_time': start_time,
                          'end_time': end_time})

    def get_conditional_order_history(self, market: str = None, side: str = None, type: str = None,
                                      order_type: str = None, start_time: float = None, end_time: float = None) -> List[
        dict]:
        return self._get(f'conditional_orders/history',
                         {'market': market, 'side': side, 'type': type, 'orderType': order_type,
                          'start_time': start_time, 'end_time': end_time})

    def modify_order(
            self, existing_order_id: Optional[str] = None,
            existing_client_order_id: Optional[str] = None, price: Optional[float] = None,
            size: Optional[float] = None, client_order_id: Optional[str] = None,
    ) -> dict:
        assert (existing_order_id is None) ^ (existing_client_order_id is None), \
            'Must supply exactly one ID for the order to modify'
        assert (price is None) or (size is None), 'Must modify price or size of order'
        path = f'orders/{existing_order_id}/modify' if existing_order_id is not None else \
            f'orders/by_client_id/{existing_client_order_id}/modify'
        return self._post(path, {
            **({'size': size} if size is not None else {}),
            **({'price': price} if price is not None else {}),
            **({'clientId': client_order_id} if client_order_id is not None else {}),
        })

    def get_conditional_orders(self, market: str = None) -> List[dict]:
        return self._get(f'conditional_orders', {'market': market})

    def place_order(self, market: str, side: str, price: float, size: float, type: str = 'limit',
                    reduce_only: bool = False, ioc: bool = False, post_only: bool = False,
                    client_id: str = None) -> dict:
        return self._post('orders', {'market': market,
                                     'side': side,
                                     'price': price,
                                     'size': size,
                                     'type': type,
                                     'reduceOnly': reduce_only,
                                     'ioc': ioc,
                                     'postOnly': post_only,
                                     'clientId': client_id,
                                     })

    def place_conditional_order(
            self, market: str, side: str, size: float, type: str = 'stop',
            limit_price: float = None, reduce_only: bool = False, cancel: bool = True,
            trigger_price: float = None, trail_value: float = None
    ) -> dict:
        """
        To send a Stop Market order, set type='stop' and supply a trigger_price
        To send a Stop Limit order, also supply a limit_price
        To send a Take Profit Market order, set type='trailing_stop' and supply a trigger_price
        To send a Trailing Stop order, set type='trailing_stop' and supply a trail_value
        """
        assert type in ('stop', 'take_profit', 'trailing_stop')
        assert type not in ('stop', 'take_profit') or trigger_price is not None, \
            'Need trigger prices for stop losses and take profits'
        assert type not in ('trailing_stop',) or (trigger_price is None and trail_value is not None), \
            'Trailing stops need a trail value and cannot take a trigger price'

        return self._post('conditional_orders',
                          {'market': market, 'side': side, 'triggerPrice': trigger_price,
                           'size': size, 'reduceOnly': reduce_only, 'type': 'stop',
                           'cancelLimitOnTrigger': cancel, 'orderPrice': limit_price})

    def cancel_order(self, order_id: str) -> dict:
        return self._delete(f'orders/{order_id}')

    def cancel_orders(self, market_name: str = None, conditional_orders: bool = False,
                      limit_orders: bool = False) -> dict:
        return self._delete(f'orders', {'market': market_name,
                                        'conditionalOrdersOnly': conditional_orders,
                                        'limitOrdersOnly': limit_orders,
                                        })

    def get_fills(self) -> List[dict]:
        return self._get(f'fills')

    def get_balances(self) -> List[dict]:
        return self._get('wallet/balances')

    def get_deposit_address(self, ticker: str) -> dict:
        return self._get(f'wallet/deposit_address/{ticker}')

    def get_positions(self, show_avg_price: bool = False) -> List[dict]:
        return self._get('positions', {'showAvgPrice': show_avg_price})

    def get_position(self, name: str, show_avg_price: bool = False) -> dict:
        return next(filter(lambda x: x['future'] == name, self.get_positions(show_avg_price)), None)

    def get_all_trades(self, market: str, start_time: float = None, end_time: float = None) -> List:
        ids = set()
        limit = 100
        results = []
        while True:
            response = self._get(f'markets/{market}/trades', {
                'end_time': end_time,
                'start_time': start_time,
            })
            deduped_trades = [r for r in response if r['id'] not in ids]
            results.extend(deduped_trades)
            ids |= {r['id'] for r in deduped_trades}
            print(f'Adding {len(response)} trades with end time {end_time}')
            if len(response) == 0:
                break
            end_time = min(parse_datetime(t['time']) for t in response).timestamp()
            if len(response) < limit:
                break
        return results


class KrakenClient(BaseClient):
    _ENDPOINT = 'https://api.kraken.com'

    def __init__(self, api_key=None, api_secret=None) -> None:
        super().__init__()
        self._session = Session()
        self._api_key = api_key
        self._api_secret = api_secret

    def get_kraken_signature(self, urlpath, data, secret):
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()

        mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
        sigdigest = base64.b64encode(mac.digest())
        return sigdigest.decode()

    def kraken_request(self, uri_path, data, api_key, api_sec):
        headers = {'API-Key': api_key, 'API-Sign': self.get_kraken_signature(uri_path, data, api_sec)}
        # get_kraken_signature() as defined in the 'Authentication' section
        req = requests.post((self._ENDPOINT + uri_path), headers=headers, data=data)
        return req

    def get_open_orders(self, market: str = None) -> List[dict]:
        return self.kraken_request(uri_path='/0/private/OpenOrders', data={
            "nonce": str(int(1000 * time.time())),
            "trades": True
        }, api_key=self._api_key, api_sec=self._api_secret).json()

    def place_order(self, market: str, side: str, price: float, size: float, type: str = 'limit',
                    reduce_only: bool = False, ioc: bool = False, post_only: bool = False,
                    client_id: str = None) -> dict:
        return self.kraken_request('/0/private/AddOrder', {
            "nonce": str(int(1000 * time.time())),
            "ordertype": type,
            "type": side,
            "volume": size,
            "pair": market,
            "price": price
        }, self._api_key, self._api_secret).json()

    def cancel_order(self, order_id: str) -> dict:
        return self.kraken_request('/0/private/CancelOrder', {
            "nonce": str(int(1000 * time.time())),
            "txid": "OG5V2Y-RYKVL-DT3V3B"
        }, self._api_key, self._api_key).json()

    def get_balances(self) -> List[dict]:
        return self.kraken_request('/0/private/Balance', {
            "nonce": str(int(1000 * time.time()))
        }, self._api_key, self._api_secret).json()

    def get_positions(self, show_avg_price: bool = False) -> List[dict]:
        return self.kraken_request('/0/private/OpenPositions', {
            "nonce": str(int(1000 * time.time())),
            "docalcs": True
        }, self._api_key, self._api_secret).json()
