import logging
import sys
from os.path import join

# https://stackoverflow.com/questions/54591352/python-logging-new-log-file-each-loop-iteration
def create_logging(logger_name :str, 
                level : int = logging.INFO) -> logging.Logger:
    """Method to return a custom logger with the given name and level

    Args:
        logger_name (str): Log file location
        level (int, optional): Loggin level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Logger object
    """
    logger = logging.getLogger(logger_name)

    logger.setLevel(level)
    format_string = "%(asctime)s %(message)s"
    log_format = logging.Formatter(fmt = format_string, 
                                datefmt= '%Y-%m-%d %I:%M:%S %p')
    
    # Creating and adding the console handler
    # console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setFormatter(log_format)
    # logger.addHandler(console_handler)

    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger


# #Sample Usage
# if __name__ == "__main__":
#     for item in range(5):
#         location = join("/mnt/d/Tensor/tensortrader/backtests/tmp",f"Logger{item}.log" )
#         logger = logger_obj(location)
#         logger.info(f"This is information for item  {item}")