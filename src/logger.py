import logging
import os
from datetime import datetime

Log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"


logs_path = os.path.join(os.getcwd(),"logs",Log_file)

os.makedirs(logs_path,exist_ok=True)

log_file_path = os.path.join(logs_path,Log_file)

# print(log_file_path)

logging.basicConfig(

    filename = log_file_path,
    format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO
)

# if __name__ == "__main__":

#     logging.info("Logging has started")
#     logging.info("how is your day")

