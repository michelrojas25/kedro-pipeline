from typing import Dict
from kedro.pipeline import Pipeline
from prediccion_ventas.pipelines.data_engineering import pipeline as de_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    de = de_pipeline.create_pipeline()
    return {
        "__default__": de,
        "de": de
    }

