#!/bin/sh

PIPELINE=
#PIPELINE=src/aml_pipelines/run_step_1_training_locally.py
#PIPELINE=src/aml_pipelines/run_step_1_training_remotely.py
#PIPELINE=src/aml_pipelines/run_step_1_inference_locally.py
#PIPELINE=src/aml_pipelines/run_step_1_inference_remotely.py
#PIPELINE=src/aml_pipelines/run_step_2_extraction_locally.py
#PIPELINE=src/aml_pipelines/run_step_2_extraction_remotely.py
#PIPELINE=src/aml_pipelines/run_step_2_images_upload_locally.py
#PIPELINE=src/aml_pipelines/run_step_3_training_locally.py
#PIPELINE=src/aml_pipelines/run_step_3_training_remotely.py
#PIPELINE=src/aml_pipelines/run_step_3_inference_locally.py
#PIPELINE=src/aml_pipelines/run_step_3_inference_remotely.py
#PIPELINE=src/aml_pipelines/run_step_4_locally.py
#PIPELINE=src/aml_pipelines/run_step_4_remotely.py
#PIPELINE=src/aml_pipelines/run_pipeline_remotely.py

if [ ${#PIPELINE} -gt 0 ]
then
    echo "Will execute $PIPELINE..."
    python $PIPELINE
else
    echo "No pipeline selected!"
fi