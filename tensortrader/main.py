import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pymongo
import yaml
from binance.client import Client

from tensortrader.tasks.task_utils import create_logging
from tensortrader.Trader.trader import BinanceTrader

# export PYTHONPATH="${PYTHONPATH}:/mnt/d/Tensor/tensortrader-system"
# run -->
#  conda activate Tensor
#  cd tensortrader/ & python main.py BTCUSDT


def main(symbol):

    path = f"/mnt/d/Tensor/tensortrader-system/ \
            tensortrader/config/trading/{symbol}.yml"
    CONF = yaml.safe_load(Path(path).read_text())

    print(CONF)

    # -----------------------------
    # Parameters
    # -----------------------------
    symbol = CONF["symbol"]
    units = CONF["units"]
    max_trades = CONF["max_trades"]
    signal_loc = CONF["signal_loc"]
    config_loc = CONF["config_loc"]
    path_logs = CONF["path_logs"]
    model = CONF["model"]
    bar_length = CONF["bar_length"]
    position = CONF["position"]
    max_trade_time = CONF["max_trade_time"]  # minutes
    target_usdt = CONF["target_usdt"]  # USDT
    stop_usdt = CONF["stop_usdt"]  # USDT
    database_loc = CONF["database_loc"]

    # -----------------------------
    # Logging Config
    # -----------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    trading_dir = os.path.join(path_logs, f"Trading_Execution_LOG_{symbol}")

    if not os.path.exists(trading_dir):
        os.mkdir(trading_dir)

    print("Storing Price Return data at", trading_dir)

    LOG_FILENAME = os.path.join(trading_dir, f"{timestamp}_Trading_Execution.log")

    print("Logging data at ", LOG_FILENAME)

    logger = create_logging(LOG_FILENAME)

    # -----------------------------
    # Trader
    # -----------------------------
    with open(config_loc) as f:
        SECRETS = json.load(f)
    f.close()

    api_key = SECRETS.get("key_test")
    api_secret = SECRETS.get("secret_test")

    try:
        binance_client = Client(
            api_key=api_key, api_secret=api_secret, tld="com", testnet=True
        )
        logger.info("Connection to Binance Test API sucessfully created.")
    except Exception as e:
        logger.error(f"{e}")

    try:
        MONGO_PASSWORD = SECRETS.get("MONGO_PASSWORD")
        MONGO_USER = SECRETS.get("MONGO_USER")
        MONGO_URL = "mongodb+srv://{}:{}@tensor-database.rjyvv.mongodb.net \
                    /?retryWrites=true&w=majority".format(
            MONGO_USER, MONGO_PASSWORD
        )

        mongo_client = pymongo.MongoClient(MONGO_URL)

        mongodb_database = mongo_client["Trading_Execution"]
        mongodb_collection = mongodb_database[f"{symbol}"]

        logger.info("Connection to MONGO DB sucessfully created.")
    except Exception as e:
        logger.error(f"{e}")

    trader = BinanceTrader(
        symbol=symbol,
        database_loc=database_loc,
        mongodb_collection=mongodb_collection,
        signal_loc=signal_loc,
        bar_length=bar_length,
        client=binance_client,
        model=model,
        units=units,
        position=position,
        max_trades=max_trades,
        max_trade_time=max_trade_time,
        target_usdt=target_usdt,
        stop_usdt=stop_usdt,
        logger=logger,
    )

    trader.start_trading()

    # trader.start_streaming()


if __name__ == "__main__":

    symbol = sys.argv[1]
    # symbol = 'BTCUSDT'
    main(symbol)
