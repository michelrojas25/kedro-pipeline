from kedro.pipeline import Pipeline, node
from .nodes import load_datasets, preprocess_data, generar_graficos

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
                outputs=["preprocessed_data", "features"],
                name="preprocess_data_node"
            ),
            node(
                func=generar_graficos,
                inputs=["preprocessed_data", "features"],
                outputs=dict(
                    dist_payment_value="dist_payment_value",
                    correlation_matrix="correlation_matrix",
                    boxplots="boxplots"
                ),
                name="generar_graficos_node"
            )
        ]
    )

