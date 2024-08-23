# Predicción de Ventas

## Descripción General

Este proyecto utiliza Kedro para la predicción de ventas, basado en datos históricos de transacciones. Kedro proporciona una estructura modular y escalable para organizar y ejecutar pipelines de datos.


## Instalación de Dependencias
1. Creacion del entorno virtual con conda.

    ```bash
     conda create --name kedro-env python==3.12.4
    
     # una vez este creado lo activamos
     conda activate kedro-env
    ```
2. Clonamos le repositorio.
    ```bash
     git clone
    ```

2. Instalamos las dependencias.

    ```bash 
     pip install -r requirements.txt
    ```
3. Levantamos Kedro console y kedro interface

    ``` bash
     kedro run

     # levantmos la interface
     kedro viz
     
     # si todo sale bien deberia aparecernos
     Kedro Viz started successfully.

     ✨ Kedro Viz is running at
     http://127.0.0.1:4141/
    ```
