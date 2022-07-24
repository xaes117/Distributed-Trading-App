import json

from RiskManager import RiskManager
from Services import LibService
from datetime import datetime
import traceback

import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    with open('config.json') as config_file:
        json_data = json.load(config_file)

    api_key = list(json_data.keys())[0]
    risk_manager = RiskManager(api_key=api_key)

    time_completed = 0

    while True:
        try:
            print(f'\n{datetime.now()}')
            risk_manager.synchronise()
            buy_service = risk_manager.get_buy_service()
            buy_trades: list = buy_service.generate_trades()
            risk_manager.execution_service.execute_trades(trades=buy_trades)

            sell_service = risk_manager.get_sell_service()
            sell_trades: list = sell_service.generate_trades()
            risk_manager.execution_service.execute_trades(trades=sell_trades)

            # evaluate risk every 4 hours
            if time_completed == 0:
                print('---------------------------------')
                print('Initiating risk evaluation')
                risk_manager.manage_risk()
                print('Risk evaluation complete')
                print('---------------------------------')

        except Exception as e:
            print(traceback.format_exc())
            with open('error_log.txt', 'a+') as error_log:
                error_log.write(str(e))
                error_log.write('\n')
            # wait for 15 minutes
            LibService.patiently_wait(cycle_time=60 * 15, interval=1)
        finally:
            cycle_time = risk_manager.lib.cycle_time
            LibService.patiently_wait(cycle_time=cycle_time, interval=1, logging=True)
            time_completed = time_completed + cycle_time
            if time_completed > 14400:
                time_completed = 0
