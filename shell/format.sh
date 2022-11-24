#!/bin/bash
# isort --sl src/keras_explainable
black --line-length 90 src/keras_explainable
flake8 src/keras_explainable
