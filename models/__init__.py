from models.CDO import *
import logging
logger = logging.getLogger(__name__)

def get_model_from_args(**kwargs)->CDOModel:
    model = CDOModel(**kwargs)
    return model