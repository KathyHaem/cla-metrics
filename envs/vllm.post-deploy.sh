#!env bash

# Force the installation of transformers 5.0.0, which is required for the models, and is not officially supported by vLLM 0.19.0
python3 -m pip install --upgrade "transformers==5.13.0"