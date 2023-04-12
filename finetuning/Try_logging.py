import logging

logging.basicConfig(
    filename='log/sample.log',
    # format='%(asctime)s:%(name)s: %(message)s',
    format='%(asctime)s: %(message)s',
)

logging.info("Test log")
logging.warning("Test log")
logging.critical("Test log")