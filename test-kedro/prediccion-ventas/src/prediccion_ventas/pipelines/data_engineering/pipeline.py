from kedro.pipeline import Pipeline, node
from .nodes import load_datasets, preprocess_data

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=load_datasets,
                inputs=dict(order_items="order_items", products="products", payments="payments"),
                outputs="datasets",
                name="load_datasets_node"
            ),
            node(
                func=preprocess_data,
                inputs=dict(datasets="datasets", threshold="params:threshold"),
                outputs="preprocessed_data",
                name="preprocess_data_node"
            )
        ]
    )

