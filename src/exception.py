import sys
import logging
from src.logger import logging


# when an error raises this function is called

def error_message_details(error,error_detail:sys): 
    # gives you detail on which file the exceptiion has occured and on which line the exception has occured
   _,_,exc_tb =  error_detail.exc_info()

   file_name = exc_tb.tb_frame.f_code.co_filename


   error_message = "Error occured in python script name [{0}] line numnber [{1}] error message [{2}] ".format(
      
      file_name, exc_tb.tb_lineno,str(error)
   )

   return error_message

class CustomException(Exception):
   
        def __init__(self,error_message,error_detail:str):
              
            super().__init__(error_message)

            self.error_message = error_message_details(error_message,error_detail=error_detail)

        def __str__(self):
              
            return self.error_message 


# if __name__ == "__main__":
     
#     try:
#           a=1/0

#     except Exception as e:
         
#         logging.info("custom exception divide by zero error")

#         raise CustomException(e,sys)
          