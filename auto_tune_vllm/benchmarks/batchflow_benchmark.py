from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional
from pprint import pformat

import requests
from .config import BenchmarkConfig
from .providers import BenchmarkProvider

logger = logging.getLogger(__name__)


class BatchFlowBenchmark(BenchmarkProvider):
    """BatchFlow benchmark implementation."""
    
    def __init__(self):
        super().__init__()
        self._batch_flow_process: Optional[subprocess.Popen] = None
        self._batch_flow_port = 8080
        self._batch_flow_endpoint = f"http://127.0.0.1:{self._batch_flow_port}/v1"
        self._auth_header = "Bearer sk-8467763010886377472"
        self._batch_id: Optional[str] = None
        self._input_file_id: Optional[str] = None
        
    def start_benchmark(
        self, model_url: str, config: BenchmarkConfig
    ) -> subprocess.Popen:
        """
        Start BatchFlow service, upload file with replaced model name, 
        create batch job, and store process for polling.

        Args:
            model_url: URL of the vLLM server 
            config: Benchmark configuration

        Returns:
            Popen process handle for polling by caller
        """
        self._logger.info(f"Starting BatchFlow benchmark for {config.model}")
        self._logger.info(f"config info:\n{pformat(config.__dict__)}")
        
        # Store model URL and name for later use
        self._model_url = model_url
        self._model_name = config.model
        
        # Create results file path
        self._results_file = self._get_results_file_path()
        
        # Create trial directory
        trial_dir = self._get_trial_directory()
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate inputs
        if not (
            model_url.startswith("http://") or model_url.startswith("https://")
        ):
            raise ValueError(
                f"Invalid model_url: {model_url!r} (expected http/https)"
            )
        
        # Start BatchFlow process
        self._start_batchflow_service(trial_dir, config)
        
        # Wait for service to be ready
        self._wait_for_service()
        
        # Upload file with replaced model name
        dataset_file = self._prepare_dataset(config)
        self._upload_file(dataset_file, config.model)
        
        # Create batch job
        self._create_batch_job()
        
        # Create a polling script that checks batch job status
        polling_script = trial_dir / "batchflow_poller.py"
        with open(polling_script, "w") as f:
            f.write(f"""
import time
import requests
import json
import sys

def get_batch_status(batch_id):
    url = "http://127.0.0.1:{self._batch_flow_port}/v1/batches/" + batch_id
    headers = {{"Authorization": "{self._auth_header}"}}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def main():
    batch_id = "{self._batch_id}"
    results_file = "{self._results_file}"
    
    while True:
        try:
            status = get_batch_status(batch_id)
            batch_status = status["status"]
            print(f"Batch status: {{batch_status}}")
            
            if batch_status == "completed":
                # Save results
                with open(results_file, "w") as f:
                    json.dump(status, f, indent=2)
                print("Batch job completed. Results saved.")
                break
            elif batch_status in ["failed", "expired", "cancelled"]:
                print(f"Batch job ended with status: {{batch_status}}")
                sys.exit(1)
                
        except Exception as e:
            print(f"Error checking batch status: {{e}}")
            sys.exit(1)
            
        time.sleep(5)  # Poll every 5 seconds

if __name__ == "__main__":
    main()
""")

        # Create a process that polls the batch job status
        self._process = subprocess.Popen(
            ["python", str(polling_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True
        )
        
        # Store PID and PGID immediately for cleanup
        self._process_pid = self._process.pid
        try:
            self._process_pgid = os.getpgid(self._process_pid)
            self._logger.debug(
                f"Started BatchFlow polling process {self._process_pid} "
                f"in process group {self._process_pgid}"
            )
        except (OSError, ProcessLookupError):
            self._logger.warning(
                f"Failed to get process group for BatchFlow polling process "
                f"{self._process_pid}"
            )
            self._process_pgid = None
            
        self._logger.info(f"Results will be saved to: {self._results_file}")
        
        return self._process

    def parse_results(self) -> Dict[str, Any]:
        """
        Parse BatchFlow benchmark results from output file.

        Returns:
            Dictionary with benchmark results. Must include metrics that can be
            converted to objective values for Optuna.
        """
        results_file = self._results_file
        
        if not os.path.exists(results_file):
            raise RuntimeError(f"BatchFlow benchmark results file not found: {results_file}")
        
        try:
            with open(results_file) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in results file: {e}")
        
        return self._parse_batchflow_results(data)
    
    def terminate_benchmark(self):
        """Terminate the running batchflow process and polling process."""
        # Terminate the polling process first
        super().terminate_benchmark()
        
        # Cancel the batch job if it exists and is not completed
        if self._batch_id:
            try:
                self._cancel_batch_job()
            except Exception as e:
                self._logger.warning(f"Failed to cancel batch job: {e}")
        
        # Stop BatchFlow service
        self._stop_batchflow_service()
    
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
            results_file = benchmark_dir / f"{trial_id}_batchflow_benchmark_results.json"

            return str(results_file)

        except Exception as e:
            self._logger.warning(
                f"Failed to create results path: {e}, using temporary file"
            )
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                return f.name
                
    def _get_trial_directory(self) -> Path:
        """
        Get the trial directory path for storing temporary files.
        """
        if self._trial_context is None:
            # Fallback to temporary directory if no trial context
            self._logger.warning("No trial context set, using temporary directory")
            return Path(tempfile.mkdtemp(prefix="batchflow_benchmark_"))

        try:
            # Create directory structure:
            # {logging.file_path}/{study_name}/{trial_id}/
            study_name = self._trial_context["study_name"]
            trial_id = self._trial_context["trial_id"]
            
            # Use logging file path from config or default to /tmp
            log_file_path = self._trial_context.get("log_file_path")
            if log_file_path:
                base_dir = Path(log_file_path)
            else:
                base_dir = Path("/tmp/auto-tune-vllm-local-run/logs")
            
            trial_dir = base_dir / study_name / trial_id

            # Create directory if it doesn't exist
            trial_dir.mkdir(parents=True, exist_ok=True)

            return trial_dir

        except Exception as e:
            self._logger.warning(
                f"Failed to create trial directory path: {e}, using temporary directory"
            )
            return Path(tempfile.mkdtemp(prefix="batchflow_benchmark_"))
    
    def _parse_batchflow_results(self, data: dict) -> Dict[str, Any]:
        """Parse BatchFlow benchmark JSON results data structure."""
        try:
            # Extract key metrics from BatchFlow benchmark results
            results = {}
            
            # Check if the batch job completed successfully
            if data.get("status") != "completed":
                raise RuntimeError(f"Batch job did not complete successfully. Status: {data.get('status')}")
                
            # Extract timing information
            created_at = data.get("created_at", 0)
            completed_at = data.get("completed_at", 0)
            
            if completed_at <= created_at:
                raise RuntimeError("Invalid timing data in batch flow results")
                
            duration = completed_at - created_at
            results["duration"] = duration
            
            # Extract token counts
            total_prompt_tokens = data.get("total_prompt_tokens", 0)
            total_completion_tokens = data.get("total_completion_tokens", 0)
            
            # Calculate throughput metrics
            # Output throughput (completion tokens per second)
            if duration > 0 and total_completion_tokens > 0:
                output_throughput = total_completion_tokens / duration
                results["output_tokens_per_second"] = output_throughput
                results["output_tokens_per_second_mean"] = output_throughput
                
            # Request throughput (requests per second)
            request_counts = data.get("request_counts", {})
            total_requests = request_counts.get("total", 0)
            if duration > 0 and total_requests > 0:
                request_throughput = total_requests / duration
                results["requests_per_second"] = request_throughput
                
            # Token count metrics
            results["prompt_token_count"] = total_prompt_tokens
            results["output_token_count"] = total_completion_tokens
            
            # Error rate calculation
            completed_requests = request_counts.get("completed", 0)
            failed_requests = request_counts.get("failed", 0)
            total = completed_requests + failed_requests
            
            if total > 0:
                results["error_rate"] = failed_requests / total
            else:
                results["error_rate"] = 0.0
            
            # Completion stats
            results["completed_requests"] = completed_requests
            results["failed_requests"] = failed_requests
            results["total_requests"] = total
            
            return results
            
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Invalid BatchFlow benchmark data structure: {e}")
            
    def _start_batchflow_service(self, trial_dir: Path, config: BenchmarkConfig):
        """Start the BatchFlow service as a subprocess."""
        self._logger.info("Starting BatchFlow service...")
        
        # Determine concurrency from config
        concurrency = 32  # default value
        if hasattr(config, 'max_concurrency') and config.max_concurrency is not None:
            concurrency = config.max_concurrency
        elif hasattr(config, 'concurrency') and config.concurrency is not None:
            concurrency = config.concurrency
        
        # Process model URL to remove trailing /v1 if present
        model_url = self._model_url
        if model_url.endswith("/v1"):
            model_url = model_url[:-3]
        
        # Set up environment variables for BatchFlow
        env = os.environ.copy()
        env.update({
            "BO_LOG_LEVEL": "DEBUG",
            "BO_ADMIN_AUTH_HEADER": self._auth_header,
            "BO_DATA_DIR": str(trial_dir / "batchflow_data"),
            "BO_DB_TYPE": "sqlite",
            "BO_ADDR": f":{self._batch_flow_port}",
            "BO_SERVER_HANDLER_TIMEOUT": "4h",
            "BO_HTTP_TIMEOUT": "4h",
            "BO_COMPLETION_WINDOW": "960h",
            # Use the determined concurrency value
            "BO_MODEL_BACKENDS": f"{self._model_name}={model_url}|{concurrency}",
            "BO_MODEL_BACKENDS_REFRESH_INTERVAL": "60s"
        })
        
        # Create data directory
        Path(env["BO_DATA_DIR"]).mkdir(parents=True, exist_ok=True)
        
        # Find batchflow executable
        batchflow_executable = self._find_batchflow_executable()
        
        # Log the command and environment variables before starting
        cmd = [batchflow_executable]
        self._logger.info(f"BatchFlow startup command: {cmd}")
        self._logger.info(f"BatchFlow environment variables: {env}")
        
        # Start BatchFlow process
        self._batch_flow_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 将stderr合并到stdout，便于日志追踪
            start_new_session=True,
            text=True,  # 直接以文本模式输出，方便后续读取
            bufsize=1   # 行缓冲，确保日志能及时输出
        )
        
        self._logger.info(f"Started BatchFlow service with PID {self._batch_flow_process.pid}")
        
    def _stop_batchflow_service(self):
        """Stop the BatchFlow service."""
        if self._batch_flow_process:
            self._logger.info("Stopping BatchFlow service...")
            try:
                # Try graceful shutdown first
                self._batch_flow_process.terminate()
                try:
                    self._batch_flow_process.wait(timeout=10)
                    self._logger.info("BatchFlow service stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if it didn't terminate gracefully
                    self._batch_flow_process.kill()
                    self._batch_flow_process.wait()
                    self._logger.info("BatchFlow service forcefully terminated")
            except ProcessLookupError:
                self._logger.info("BatchFlow process already terminated")
            finally:
                self._batch_flow_process = None
                
    def _find_batchflow_executable(self, path: Optional[str] = None) -> str:
        """Find the BatchFlow executable."""
        # Check if batchflow is in PATH
        batchflow_path = shutil.which("batchflow")
        if batchflow_path:
            return batchflow_path
            
        # Check common locations
        common_paths = [
            path,
            "./batchflow/batchflow",
            "~/go/bin/batchflow",
            "/usr/local/bin/batchflow"
        ]
        
        # Add path relative to current module
        module_dir = Path(__file__).parent
        batchflow_relative_path = module_dir / "batchflow" / "batchflow"
        common_paths.insert(1, str(batchflow_relative_path))
        
        for path in common_paths:
            if path is None:
                continue
            expanded_path = os.path.expanduser(path)
            if os.path.isfile(expanded_path) and os.access(expanded_path, os.X_OK):
                return expanded_path
                
        raise RuntimeError(
            "Could not find BatchFlow executable. Please ensure it is installed "
            "and available in PATH or in a common location."
        )
        
    def _wait_for_service(self, timeout: int = 60):
        """Wait for the BatchFlow service to be ready."""
        self._logger.info("Waiting for BatchFlow service to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://127.0.0.1:{self._batch_flow_port}/health")
                if response.status_code == 200:
                    self._logger.info("BatchFlow service is ready")
                    return
            except requests.RequestException:
                pass
                
            time.sleep(1)
            
        raise RuntimeError(f"BatchFlow service did not start within {timeout} seconds")
        
    def _prepare_dataset(self, config: BenchmarkConfig) -> Path:
        """Prepare dataset file with model name replaced."""
        trial_dir = self._get_trial_directory()
        dataset_path = trial_dir / "dataset.jsonl"
        
        # Determine source dataset file
        if config.dataset:
            source_dataset = Path(config.dataset)
        else:
            # Use default sample dataset
            source_dataset = Path(__file__).parent / "sample-gen-math-1000.jsonl"
            
        if not source_dataset.exists():
            raise FileNotFoundError(f"Dataset file not found: {source_dataset}")
            
        # Read and process the dataset, replacing model names
        count = 0
        with open(source_dataset, "r") as infile, open(dataset_path, "w") as outfile:
            for line in infile:
                # Check if we've reached the sample limit
                if hasattr(config, 'samples') and config.samples is not None and count >= config.samples:
                    break
                    
                try:
                    data = json.loads(line)
                    # Replace model name in the body if it exists
                    if "body" in data and "model" in data["body"]:
                        data["body"]["model"] = config.model
                    
                    # Replace max_tokens with output_tokens from config if specified
                    if hasattr(config, 'output_tokens') and config.output_tokens is not None:
                        if "body" in data and "max_tokens" in data["body"]:
                            data["body"]["max_tokens"] = config.output_tokens
                    
                    outfile.write(json.dumps(data) + "\n")
                    count += 1
                except json.JSONDecodeError:
                    # If line is not valid JSON, write as is
                    outfile.write(line)
                    count += 1
                    
        self._logger.info(f"Prepared dataset with {count} samples, model name replaced: {dataset_path}")
        return dataset_path
        
    def _upload_file(self, file_path: Path, model_name: str):
        """Upload file to BatchFlow service."""
        self._logger.info("Uploading file to BatchFlow service...")
        
        url = f"{self._batch_flow_endpoint}/files"
        headers = {"Authorization": self._auth_header}
        
        with open(file_path, "rb") as f:
            files = {
                "file": f,
                "purpose": (None, "batch")
            }
            response = requests.post(url, headers=headers, files=files)
            
        response.raise_for_status()
        result = response.json()
        self._input_file_id = result["id"]
        
        self._logger.info(f"File uploaded with ID: {self._input_file_id}")
        
    def _create_batch_job(self):
        """Create a batch job in BatchFlow service."""
        if not self._input_file_id:
            raise RuntimeError("No input file ID available. Upload a file first.")
            
        self._logger.info("Creating batch job...")
        
        url = f"{self._batch_flow_endpoint}/batches"
        headers = {
            "Authorization": self._auth_header,
            "Content-Type": "application/json"
        }
        data = {
            "input_file_id": self._input_file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "metadata": {
                "customer_id": "user_123456789",
                "batch_description": "Auto-tuning job"
            }
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        self._batch_id = result["id"]
        
        self._logger.info(f"Batch job created with ID: {self._batch_id}")
        
    def _cancel_batch_job(self):
        """Cancel the batch job."""
        if not self._batch_id:
            return
            
        self._logger.info(f"Cancelling batch job {self._batch_id}...")
        
        url = f"{self._batch_flow_endpoint}/batches/{self._batch_id}"
        headers = {"Authorization": self._auth_header}
        
        # Note: This would depend on the actual BatchFlow API for cancellation
        # For now we'll just log that we're trying to cancel
        self._logger.warning("Batch job cancellation is not yet implemented in BatchFlow API")
        
    def _get_batch_status(self) -> dict:
        """Get the status of the batch job."""
        if not self._batch_id:
            raise RuntimeError("No batch job ID available. Create a batch job first.")
            
        url = f"{self._batch_flow_endpoint}/batches/{self._batch_id}"
        headers = {"Authorization": self._auth_header}
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
        
    def check_completion(self) -> bool:
        """
        Check if the batch job is completed and save results if it is.
        This method should be called periodically by the caller.
        """
        try:
            status = self._get_batch_status()
            batch_status = status["status"]
            self._logger.info(f"Batch status: {batch_status}")
            
            if batch_status == "completed":
                # Save results
                with open(self._results_file, "w") as f:
                    json.dump(status, f, indent=2)
                self._logger.info("Batch job completed. Results saved.")
                return True
            elif batch_status in ["failed", "expired", "cancelled"]:
                raise RuntimeError(f"Batch job ended with status: {batch_status}")
                
            return False
        except Exception as e:
            self._logger.error(f"Error checking batch status: {e}")
            raise