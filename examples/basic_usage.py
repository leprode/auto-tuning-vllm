#!/usr/bin/env python3
"""
Basic usage example for auto-tune-vllm.

This example shows how to set up and run a vLLM optimization study
using the Python API instead of the CLI.
"""

import json

import optuna

from auto_tune_vllm import (
    # RayExecutionBackend,
    LocalExecutionBackend,
    StudyConfig,
    StudyController,
)


def main():
    # Create study configuration
    config = StudyConfig.from_file("examples/study_config_vllm.yaml")

    # Choose execution backend
    # Option 1: Ray distributed execution
    # backend = RayExecutionBackend(
    #     resource_requirements={"num_gpus": 1, "num_cpus": 4}
    # )

    # Option 2: Local execution for testing
    backend = LocalExecutionBackend(max_concurrent=2)  # Use this for testing

    # Create study controller (this will properly create the Optuna study with correct sampler)
    controller = StudyController.create_from_config(backend=backend, config=config)

    # Run optimization
    print("Starting vLLM optimization study...")
    
    controller.run_optimization(n_trials=config.optimization.n_trials, max_concurrent_trials=config.optimization.max_concurrent_trials)

    results = controller.get_optimization_results()

    # Print results
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()

