from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

from .config import BenchmarkConfig
from .providers import BenchmarkProvider

logger = logging.getLogger(__name__)


class VllmBenchmark(BenchmarkProvider):
    """vLLM benchmark implementation using vLLM's built-in 'vllm bench serve' command."""
    
    def start_benchmark(
        self, model_url: str, config: BenchmarkConfig
    ) -> subprocess.Popen:
        """
        Start vLLM benchmark subprocess (non-blocking).

        Args:
            model_url: URL of the vLLM server (e.g., "http://localhost:8000/v1")
            config: Benchmark configuration

        Returns:
            Popen process handle for polling by caller
        """
        self._logger.info(f"Starting vLLM benchmark for {config.model}")
        
        # Create results file path
        self._results_file = self._get_results_file_path()
        
        # Build vLLM benchmark command
        cmd = self._build_vllm_benchmark_command(model_url, config, self._results_file)
        
        # Validate binary and basic inputs
        import shutil

        if shutil.which("vllm") is None:
            raise RuntimeError(
                "vLLM CLI not found on PATH. "
                "Please install vllm package."
            )
        if not (
            model_url.startswith("http://") or model_url.startswith("https://")
        ):
            raise ValueError(
                f"Invalid model_url: {model_url!r} (expected http/https)"
            )
        
        # Run vLLM benchmark
        self._logger.info(f"Running: {' '.join(cmd)}")
        self._logger.info(f"Results will be saved to: {self._results_file}")
        
        # Use Popen so we can terminate if vLLM dies
        # start_new_session=True puts it in its own process group for clean termination
        env = os.environ.copy()
        
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True
        )
        
        # Store PID and PGID immediately for cleanup, even if process handle is lost
        self._process_pid = self._process.pid
        try:
            self._process_pgid = os.getpgid(self._process_pid)
            self._logger.debug(
                f"Started vLLM benchmark process {self._process_pid} "
                f"in process group {self._process_pgid}"
            )
        except (OSError, ProcessLookupError):
            self._logger.warning(
                f"Failed to get process group for vLLM benchmark process "
                f"{self._process_pid}"
            )
            self._process_pgid = None
        
        return self._process

    def parse_results(self) -> Dict[str, Any]:
        """
        Parse vLLM benchmark results from output file.

        Returns:
            Dictionary with benchmark results. Must include metrics that can be
            converted to objective values for Optuna.
        """
        results_file = self._results_file
        
        if not os.path.exists(results_file):
            raise RuntimeError(f"vLLM benchmark results file not found: {results_file}")
        
        try:
            with open(results_file) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in results file: {e}")
        
        return self._parse_vllm_results(data)
    
    def _get_results_file_path(self) -> str:
        """
        Get the results file path, creating directory structure if needed.
        """
        if self._trial_context is None:
            # Fallback to temporary file if no trial context
            self._logger.warning("No trial context set, using temporary file")
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                return f.name

        try:
            # Create directory structure:
            # /tmp/auto-tune-vllm-local-run/logs/{study_name}/benchmark_results/
            study_name = self._trial_context["study_name"]
            trial_id = self._trial_context["trial_id"]

            # Use /tmp as base directory for consistency with existing log structure
            base_dir = Path("/tmp/auto-tune-vllm-local-run/logs")
            benchmark_dir = base_dir / study_name / "benchmark_results"

            # Create directory if it doesn't exist
            benchmark_dir.mkdir(parents=True, exist_ok=True)

            # Create results file with trial-specific name
            results_file = benchmark_dir / f"{trial_id}_vllm_benchmark_results.json"

            return str(results_file)

        except Exception as e:
            self._logger.warning(
                f"Failed to create results path: {e}, using temporary file"
            )
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                return f.name
    
    def _build_vllm_benchmark_command(
        self, model_url: str, config: BenchmarkConfig, results_file: str
    ) -> list[str]:
        """Build vLLM benchmark command arguments."""
        # Extract directory and filename from results_file path
        results_path = Path(results_file)
        results_dir = str(results_path.parent)
        results_filename = results_path.name
        
        # Remove trailing slash and '/v1' suffix if present to prevent duplication
        base_url = model_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]  # Remove '/v1' suffix
            
        cmd = [
            "vllm",
            "bench",
            "serve",
            "--backend",
            "vllm",
            "--base-url",
            base_url,  # Use cleaned base URL
            "--model",
            config.model,
            "--num-prompts",
            str(config.samples if config.samples is not None else 1000),
            "--request-rate",
            str(config.rate),
            "--save-result",  # This is a boolean flag
            "--result-dir",
            results_dir,
            "--result-filename",
            results_filename,
        ]
        
        # Add tokenizer (use model as default tokenizer)
        cmd.extend(["--tokenizer", config.model])
        
        # Configure dataset
        if config.use_synthetic_data:
            # Use random dataset with synthetic data
            cmd.extend([
                "--dataset-name", 
                "random",
                "--random-input-len",
                str(config.prompt_tokens),
                "--random-output-len", 
                str(config.output_tokens)
            ])
        elif config.dataset:
            if config.dataset.startswith("hf://"):
                # HuggingFace dataset
                dataset_name = config.dataset[5:]  # Remove "hf://" prefix
                cmd.extend(["--dataset-name", "sharegpt", "--dataset-path", dataset_name])
            else:
                # Local file
                if not os.path.exists(config.dataset):
                    raise FileNotFoundError(f"Dataset file not found: {config.dataset}")
                cmd.extend(["--dataset-name", "sharegpt", "--dataset-path", config.dataset])
        
        # Add max concurrency if specified
        if hasattr(config, 'max_concurrency') and config.max_concurrency is not None:
            cmd.extend(["--max-concurrency", str(config.max_concurrency)])
        elif config.concurrency is not None:
            cmd.extend(["--max-concurrency", str(config.concurrency)])
        
        # Add other optional parameters
        if hasattr(config, 'seed') and config.seed is not None:
            cmd.extend(["--seed", str(config.seed)])
        
        return cmd
    
    def _parse_vllm_results(self, data: dict) -> Dict[str, Any]:
        """Parse vLLM benchmark JSON results data structure."""
        try:
            # Extract key metrics from vLLM benchmark results
            results = {}
            
            # Map vLLM metrics to consistent naming scheme used by GuideLLM
            # TTFT (Time To First Token) metrics
            if "mean_ttft_ms" in data:
                results["time_to_first_token_ms_mean"] = data["mean_ttft_ms"]
            if "median_ttft_ms" in data:
                results["time_to_first_token_ms_median"] = data["median_ttft_ms"]
            if "std_ttft_ms" in data:
                results["time_to_first_token_ms_std_dev"] = data["std_ttft_ms"]
            if "p99_ttft_ms" in data:
                results["time_to_first_token_ms_99"] = data["p99_ttft_ms"]
            
            # TPOT (Time Per Output Token) metrics
            if "mean_tpot_ms" in data:
                results["time_per_output_token_ms_mean"] = data["mean_tpot_ms"]
            if "median_tpot_ms" in data:
                results["time_per_output_token_ms_median"] = data["median_tpot_ms"]
            if "std_tpot_ms" in data:
                results["time_per_output_token_ms_std_dev"] = data["std_tpot_ms"]
            if "p99_tpot_ms" in data:
                results["time_per_output_token_ms_99"] = data["p99_tpot_ms"]
            
            # ITL (Inter-Token Latency) metrics
            if "mean_itl_ms" in data:
                results["inter_token_latency_ms_mean"] = data["mean_itl_ms"]
            if "median_itl_ms" in data:
                results["inter_token_latency_ms_median"] = data["median_itl_ms"]
            if "std_itl_ms" in data:
                results["inter_token_latency_ms_std_dev"] = data["std_itl_ms"]
            if "p99_itl_ms" in data:
                results["inter_token_latency_ms_99"] = data["p99_itl_ms"]
            
            # Request latency (approximated from TTFT)
            if "mean_ttft_ms" in data:
                results["request_latency_mean"] = data["mean_ttft_ms"]
            if "median_ttft_ms" in data:
                results["request_latency_median"] = data["median_ttft_ms"]
            if "p99_ttft_ms" in data:
                results["request_latency_99"] = data["p99_ttft_ms"]
            
            # Throughput metrics
            if "request_throughput" in data:
                results["requests_per_second"] = data["request_throughput"]
            if "output_throughput" in data:
                results["output_tokens_per_second"] = data["output_throughput"]
                results["output_tokens_per_second_mean"] = data["output_throughput"]
            
            # Token count metrics
            if "total_input_tokens" in data:
                results["prompt_token_count"] = data["total_input_tokens"]
            if "total_output_tokens" in data:
                results["output_token_count"] = data["total_output_tokens"]
            
            # Error rate calculation
            if "completed" in data and "failed" in data and "num_prompts" in data:
                total = data["completed"] + data["failed"]
                results["error_rate"] = data["failed"] / total if total > 0 else 0.0
            
            return results
            
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Invalid vLLM benchmark data structure: {e}")