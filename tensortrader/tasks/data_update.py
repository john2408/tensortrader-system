from ETL.ETL_func import *
from binance import Client
import logging
from datetime import datetime
from constants import *

# export PYTHONPATH="${PYTHONPATH}:/mnt/d/Tensor/tensortrader/tensortrader"
# Logging Config
current_date = datetime.now().strftime("%Y-%m-%d-%H-%M")
LOG_FILENAME = os.path.join( Path(os.getcwd()).parents[0].parents[0], 'logs',  f"{current_date}_Daily_candles_update.log")

print("Logging data at ", LOG_FILENAME)

logging.basicConfig(filename = LOG_FILENAME, 
                    level = logging.DEBUG, 
                    format= '%(asctime)s %(message)s', 
                    datefmt= '%m/%d/%Y %I:%M:%S %p')

config_path = join(Path(os.getcwd()).parents[0].parents[0], 'config.json')

with open(config_path) as f:
    config = json.load(f)


class task():

    def __init__(self, description, storage_folder):
        self._description = description
        self.storage_folder = storage_folder

    def run(self):
        raise NotImplementedError

    @property     
    def description(self):
        return self._description

    def __repr__(self) -> str:
        return self.description

class ETL_update_task(task):
    """_summary_

    Args:
        task (_type_): _description_
    """


    def __init__(self, description, 
                    storage_folder, 
                    config,
                    load_size_days,
                    start_time_stamp, 
                    end_timestamp,
                    interval,
                    symbols
                    ):
        """_summary_

        Args:
            description (_type_): _description_
            storage_folder (_type_): _description_
            config (_type_): _description_
            load_size_days (_type_): _description_
            start_time_stamp (_type_): _description_
            end_timestamp (_type_): _description_
            interval (_type_): _description_
            symbols (_type_): _description_
        """
        super().__init__(description, 
                        storage_folder )

        self.config = config
        self.load_size_days = load_size_days
        self.start_time_stamp = start_time_stamp
        self.end_timestamp = end_timestamp
        self.interval = interval
        self.symbols = symbols

    

    def run(self, verbose = 0):
        """_summary_

        Args:
            verbose (int, optional): _description_. Defaults to 0.
        """

        logging.info("Creating new ETL Object for Binance")
        new_ETL = ETL_Binance(self.symbols, 
                        self.storage_folder,
                        self.load_size_days, 
                        self.end_timestamp, 
                        self.start_time_stamp, 
                        total_days = None)

        new_ETL.connect_API(self.config)
        logging.info("Sucessfully connected to Binance API")

        new_ETL.update_data(self.interval, verbose = 1)
        logging.info("Data Sucessfully Updated")


if __name__ == "__main__":

    
    load_size_days = 5
    start_time_stamp = None
    end_timestamp = datetime.utcnow() - timedelta(minutes = 1) # in order to get all complete finished candles
    storage_folder = '/mnt/c/Tensor/Database/Cryptos/'

    #storage_folder = '/mnt/Data/Tensor_Invest_Fund/data/Cryptos/'
    verbose = 1
    description = "Daily update minutely candle bars data for selected portfolio"

    logging.info(f"Updating candles data for selected portfolio")
    ETL_update = ETL_update_task(description, 
                                storage_folder, 
                                config,
                                load_size_days,
                                start_time_stamp, 
                                end_timestamp,
                                INTERVAL,
                                SYMBOLS)

    ETL_update.run()

    
