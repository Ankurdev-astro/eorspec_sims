import logging
import sys

# Define a ML_pipeline FYST log level
ML_PIPELINE_LEVEL = 25 # Choose a level between INFO (20) and WARNING (30)
logging.addLevelName(ML_PIPELINE_LEVEL, "ML_PIPELINE")

def ml_pipeline(self, message, *args, **kws):
    if self.isEnabledFor(ML_PIPELINE_LEVEL):
        self._log(ML_PIPELINE_LEVEL, message, args, **kws)
        
logging.Logger.ml_pipeline = ml_pipeline

# Func to initialize and return the logger
def get_ccat_logger(rank=0):
    logger = logging.getLogger(__name__)
    logger.setLevel(ML_PIPELINE_LEVEL)
    handler = logging.StreamHandler(sys.stdout) 
    formatter = logging.Formatter(f"[Rank {rank}] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
