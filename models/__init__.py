from models.CDO import *
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_model_from_args(**kwargs)->CDOModel:
    model = CDOModel(**kwargs)
    return model