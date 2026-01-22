"""SWE-Bench Task utilities for OpenHands SDK.

This module provides utilities for running SWE-Bench tasks with the OpenHands SDK.
"""

import pandas as pd
import numpy as np
from typing import Any
import os
import json
import tempfile
import time
import re
import asyncio
import logging

# OpenHands SDK imports
from openhands.sdk import get_logger
from openhands.sdk.event import MessageEvent
from openhands.sdk.llm import Message, TextContent
from openhands.workspace import APIRemoteWorkspace

from skyrl_agent.tasks.base import BaseTask
from skyrl_agent.dispatcher.async_utils import call_sync_from_async

logger = logging.getLogger(__name__)

DOCKER_IMAGE_PREFIX = os.environ.get("EVAL_DOCKER_IMAGE_PREFIX", "docker.io/xingyaoww/")
logger.info(f"Using docker image prefix: {DOCKER_IMAGE_PREFIX}")


def ensure_serializable(obj):
    """Recursively convert non-serializable objects to JSON serializable formats."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: ensure_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [ensure_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(ensure_serializable(item) for item in obj)
    return obj


class EvalException(Exception):
    pass


def assert_and_raise(condition: bool, msg: str):
    """Raise an EvalException if the condition is not met."""
    if not condition:
        raise EvalException(msg)


RUN_WITH_BROWSING = os.environ.get("RUN_WITH_BROWSING", "false").lower() == "true"


def _get_swebench_workspace_dir_name(instance: pd.Series, dataset: str) -> str:
    if "r2e-gym" in dataset:
        return "/testbed"
    return "/workspace/" + f'{instance.repo}__{getattr(instance, "version", "null")}'.replace("/", "__")


def get_instance_docker_image(instance: pd.Series, dataset: str) -> str:
    """Get the docker image for a given SWE-Bench instance."""
    if "r2e-gym" in dataset:
        # R2E images: instance_id already contains the full image path
        # Format: {user}/{repo}:{commit_hash} e.g. namanjain12/orange3_final:2d9617bd0cb1f0ba61771258410ab8fae8e7e24d
        return f"docker.io/{instance['instance_id']}"

    # Standard SWE-Bench image naming
    repo = instance.repo.replace("/", "__")
    version = getattr(instance, "version", "null")
    return f"{DOCKER_IMAGE_PREFIX}sweb.eval.x86_64.{repo}__{version}:latest"


async def initialize_workspace_commands(workspace: APIRemoteWorkspace, instance: pd.Series, dataset: str):
    """Run initialization commands on the workspace.

    Args:
        workspace: The SDK workspace to initialize.
        instance: The SWE-Bench instance.
        dataset: The dataset name.
    """
    logger.info("-" * 30)
    logger.info("BEGIN Workspace Initialization")
    logger.info("-" * 30)

    workspace_dir_name = _get_swebench_workspace_dir_name(instance, dataset)

    # Set instance id and git configuration
    cmd = f"""echo 'export SWE_INSTANCE_ID={instance['instance_id']}' >> ~/.bashrc && \
        echo 'export PIP_CACHE_DIR=~/.cache/pip' >> ~/.bashrc && \
        echo "alias git='git --no-pager'" >> ~/.bashrc && \
        git config --global core.pager "" && \
        git config --global diff.binary false"""

    result = workspace.execute_command(cmd, timeout=600)
    assert_and_raise(result.exit_code == 0, f"Failed to configure environment: {result.output}")

    # Export USER
    result = workspace.execute_command("export USER=$(whoami); echo USER=${USER}", timeout=600)
    assert_and_raise(result.exit_code == 0, f"Failed to export USER: {result.output}")

    # Create instance data directory
    result = workspace.execute_command("mkdir -p /swe_util/eval_data/instances", timeout=600)
    assert_and_raise(result.exit_code == 0, f"Failed to create /swe_util/eval_data/instances: {result.output}")

    # Copy tools to the workspace
    script_dir = os.path.dirname(__file__)
    for tool_name in ["search", "str_replace_editor"]:
        tool_path = os.path.join(script_dir, f"scripts/tools/{tool_name}.py")
        if os.path.exists(tool_path):
            workspace.file_upload(tool_path, f"/usr/local/bin/{tool_name}")
            workspace.execute_command(f"chmod +x /usr/local/bin/{tool_name}", timeout=60)

    # Write instance JSON
    swe_instance_json_name = "swe-bench-instance.json"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, swe_instance_json_name)
        with open(temp_file_path, "w") as f:
            if not isinstance(instance, dict):
                json.dump([instance.to_dict()], f)
            else:
                json.dump([instance], f)

        workspace.file_upload(temp_file_path, f"/swe_util/eval_data/instances/{swe_instance_json_name}")

    # Copy and run entry script
    if "r2e-gym" in dataset:
        entry_script = os.path.join(script_dir, "scripts/setup/instance_r2e_entry.sh")
        if os.path.exists(entry_script):
            workspace.file_upload(entry_script, "/swe_util/instance_r2e_entry.sh")
            workspace.execute_command("chmod +x /swe_util/instance_r2e_entry.sh", timeout=60)
            result = workspace.execute_command("source ~/.bashrc && source /swe_util/instance_r2e_entry.sh", timeout=600)
            assert_and_raise(result.exit_code == 0, f"Failed to source entry script: {result.output}")
    else:
        entry_script = os.path.join(script_dir, "scripts/setup/instance_swe_entry.sh")
        if os.path.exists(entry_script):
            workspace.file_upload(entry_script, "/swe_util/instance_swe_entry.sh")
            workspace.execute_command("chmod +x /swe_util/instance_swe_entry.sh", timeout=60)
            result = workspace.execute_command("source ~/.bashrc && source /swe_util/instance_swe_entry.sh", timeout=600)
            assert_and_raise(result.exit_code == 0, f"Failed to source entry script: {result.output}")

    # Change to workspace directory
    result = workspace.execute_command(f"cd {workspace_dir_name}", timeout=600)
    assert_and_raise(result.exit_code == 0, f"Failed to cd to {workspace_dir_name}: {result.output}")

    # Git operations for swe-smith
    if dataset == "swe-smith":
        result = workspace.execute_command("git fetch", timeout=600)
        assert_and_raise(result.exit_code == 0, f"Failed to git fetch: {result.output}")

        result = workspace.execute_command(f'git checkout {instance["instance_id"]}', timeout=600)
        assert_and_raise(result.exit_code == 0, f'Failed to git checkout: {result.output}')

    # Reset and clean git for non-r2e datasets
    if "r2e-gym" not in dataset:
        result = workspace.execute_command("git reset --hard", timeout=600)
        assert_and_raise(result.exit_code == 0, f"Failed to git reset: {result.output}")

        result = workspace.execute_command('for remote_name in $(git remote); do git remote remove "${remote_name}"; done', timeout=600)
        assert_and_raise(result.exit_code == 0, f"Failed to remove git remotes: {result.output}")

    logger.info("-" * 30)
    logger.info("END Workspace Initialization")
    logger.info("-" * 30)


class SWEBenchTask(BaseTask):
    """SWE-Bench task implementation using OpenHands SDK."""

    @classmethod
    async def initialize_runtime(cls, *args, **kwargs):
        """Initialize the runtime for the task.

        For SWEBench, runtime initialization is handled by OpenHands SDK.
        """
        pass

    @classmethod
    def get_instruction(cls, instance: pd.Series, dataset: str) -> str:
        """Get the instruction for a SWE-Bench instance.

        Args:
            instance: The SWE-Bench instance.
            dataset: The dataset name.

        Returns:
            The instruction string.
        """
        workspace_dir_name = _get_swebench_workspace_dir_name(instance, dataset)
        instruction = f"""
<uploaded_files>
{workspace_dir_name}
</uploaded_files>

I've uploaded a python code repository in the directory {workspace_dir_name}. Consider the following issue description:

<issue_description>
{instance.problem_statement}
</issue_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?
I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Your task is to make the minimal changes to non-test files in the {workspace_dir_name} directory to ensure the <issue_description> is satisfied.

Follow these steps to resolve the issue:
1. First, explore the codebase to locate and understand the code relevant to the <issue_description>.
- Use efficient search commands to identify key files and functions.
- You should err on the side of caution and look at various relevant files and build your understanding of
    - how the code works
    - what are the expected behaviors and edge cases
    - what are the potential root causes for the given issue

2. Assess whether you can reproduce the issue:
    - Create a script at {workspace_dir_name}/reproduce_issue.py that demonstrates the error.
    - Execute this script to confirm the error behavior.
    - You should reproduce the issue before fixing it.
    - Your reproduction script should also assert the expected behavior for the fixed code.

3. Analyze the root cause:
    - Identify the underlying problem based on your code exploration and reproduction results.
    - Critically analyze different potential approaches to fix the issue.
    - You NEED to explicitly reason about multiple approaches to fix the issue. Next, find the most elegant and effective solution among them considering the tradeoffs (correctness, generality, side effects, etc.).
    - You would need to reason about execution paths, edge cases, and other potential issues. You should look at the unit tests to understand the expected behavior of the relevant code.

4. Implement your solution:
    - Make targeted changes to the necessary files following idiomatic code patterns once you determine the root cause.
    - You should be thorough and methodical.

5. Verify your solution:
    - Rerun your reproduction script to confirm the error is fixed.
    - If verification fails, iterate on your solution until successful. If you identify the reproduction script is buggy, adjust it as needed.

6. Run unit tests:
    - Find and run the relevant unit tests relevant to the performed fix.
    - You should run the unit tests to ensure your solution is correct and does not cause any regressions.
    - In cases where the unit tests are do not pass, you should consider whether the unit tests does not reflect the *new* expected behavior of the code. If so, you can test it by writing additional edge test cases.
    - Use the existing test runner to run the unit tests you identify as relevant to the changes you made. For example:
        - `python -m pytest -xvs sympy/physics/units/tests/test_dimensions_transcendental.py`
        - `python -m pytest tests/test_domain_py.py::test_pymethod_options`
        - `./tests/runtests.py constraints.tests.CheckConstraintTests -v 2`
    - RUN ALL relevant unit tests to ensure your solution is correct and does not cause any regressions.

7. Test edge cases:
    - Identify potential edge cases that might challenge your solution.
    - Create additional test cases in a separate file {workspace_dir_name}/edge_case_tests.py.
    - Execute these tests to verify your solution's robustness.
    - You should run multiple rounds of edge cases. When creating edge cases:
    - Consider complex scenarios beyond the original issue description
    - Test for regressions to ensure existing functionality remains intact

8. Refine if necessary:
    - If edge case testing reveals issues, refine your solution accordingly.
    - Ensure your final implementation handles all identified scenarios correctly.
    - Document any assumptions or limitations of your solution.

9. Submit your solution:
    - Once you have verified your solution, submit your solution using the `finish` tool.

A successful resolution means:
- The specific error/issue described no longer occurs
- Your changes maintain compatibility with existing functionality
- Edge cases are properly handled

"""
        return instruction

    @classmethod
    async def initialize_workspace(
        cls,
        instance: pd.Series,
        dataset: str,
        max_iterations: int,
    ) -> APIRemoteWorkspace:
        """Initialize a workspace for the agent.

        This creates an APIRemoteWorkspace that connects to the SLURM remote
        runtime server. The runtime URL should be set via the
        SANDBOX_REMOTE_RUNTIME_API_URL environment variable.

        Args:
            instance: The SWE-Bench instance.
            dataset: The dataset name.
            max_iterations: Maximum number of iterations.

        Returns:
            The initialized workspace.
        """
        # Get SLURM remote runtime API URL from environment
        # This is set by the SLURM job coordination (see slurm/jobs/verl_training.sbatch)
        runtime_api_url = os.environ.get("SANDBOX_REMOTE_RUNTIME_API_URL")
        logger.info(f"[DEBUG initialize_workspace] SANDBOX_REMOTE_RUNTIME_API_URL = {runtime_api_url}")
        if not runtime_api_url:
            logger.error("[DEBUG] SANDBOX_REMOTE_RUNTIME_API_URL is NOT SET in Ray worker!")
            raise ValueError(
                "SANDBOX_REMOTE_RUNTIME_API_URL environment variable not set. "
                "This should be set by the SLURM job coordination scripts."
            )

        runtime_api_key = os.environ.get("ALLHANDS_API_KEY", "")

        # Get container image for this instance
        server_image = get_instance_docker_image(instance, dataset)

        workspace_dir_name = _get_swebench_workspace_dir_name(instance, dataset)

        logger.info(f"Creating workspace with runtime URL: {runtime_api_url}")
        logger.info(f"Using container image: {server_image}")
        logger.info(f"Working directory: {workspace_dir_name}")

        # Create the workspace using SDK's APIRemoteWorkspace
        # This connects to the slurm-remote-runtime server
        workspace = APIRemoteWorkspace(
            working_dir=workspace_dir_name,
            host="localhost",
            runtime_api_url=runtime_api_url,
            runtime_api_key=runtime_api_key,
            server_image=server_image,
            # Timeouts for workspace initialization and API calls
            init_timeout=300.0,
            startup_wait_timeout=120.0,
            api_timeout=600.0,
            # Keep sandbox alive for potential debugging
            keep_alive=False,
        )

        # Initialize the workspace with required commands
        max_retries = 3
        retry_delay = 2

        for attempt in range(1, max_retries + 1):
            try:
                await initialize_workspace_commands(workspace, instance, dataset)
                break
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        f"[Retry {attempt}/{max_retries}] Workspace initialization failed: {e}. "
                        f"Retrying in {retry_delay} sec..."
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

        return workspace

    @classmethod
    def complete_runtime(
        cls,
        workspace: APIRemoteWorkspace,
        instance: pd.Series,
        dataset: str,
    ) -> dict[str, Any]:
        """Complete the runtime and extract results.

        Args:
            workspace: The SDK workspace.
            instance: The SWE-Bench instance.
            dataset: The dataset name.

        Returns:
            Dictionary containing the git patch and other results.
        """
        logger.info("-" * 30)
        logger.info("BEGIN Runtime Completion")
        logger.info("-" * 30)

        workspace_dir_name = _get_swebench_workspace_dir_name(instance, dataset)

        # Change to workspace directory
        result = workspace.execute_command(f"cd {workspace_dir_name}", timeout=600)

        if result.exit_code == -1:
            # Previous command still running, try to kill it
            logger.info("Previous command still running, trying to cancel...")
            workspace.execute_command("\x03", timeout=10)  # Ctrl+C
            result = workspace.execute_command(f"cd {workspace_dir_name}", timeout=600)

        # Get git diff
        try:
            git_result = workspace.git_diff(workspace_dir_name)
            git_patch = git_result.diff if hasattr(git_result, 'diff') else str(git_result)
        except Exception as e:
            logger.warning(f"Failed to get git diff: {e}")
            # Fallback to command-based diff
            result = workspace.execute_command("git diff", cwd=workspace_dir_name, timeout=600)
            git_patch = result.output if result.exit_code == 0 else ""

        logger.info("-" * 30)
        logger.info("END Runtime Completion")
        logger.info("-" * 30)

        return {
            "git_patch": git_patch,
            "workspace_dir": workspace_dir_name,
        }

    @classmethod
    async def evaluate_result(
        cls,
        instance: pd.Series,
        result: dict,
        instance_id: str,
        trajectory_id: int,
        dataset: str,
    ) -> bool:
        """Evaluate the result of a trajectory.

        Args:
            instance: The SWE-Bench instance.
            result: The result dictionary from complete_runtime.
            instance_id: The instance ID.
            trajectory_id: The trajectory ID.
            dataset: The dataset name.

        Returns:
            True if the solution is correct, False otherwise.
        """
        git_patch = result.get("git_patch", "")

        if not git_patch or git_patch.strip() == "":
            raise Exception("No git patch found in results")

        # For now, return True if we have a non-empty patch
        # Full evaluation would require running the SWE-Bench harness
        logger.info(f"Instance {instance_id}, trajectory {trajectory_id}: Got patch of length {len(git_patch)}")

        # TODO: Implement full SWE-Bench evaluation using the harness
        # This would involve:
        # 1. Applying the patch to a clean copy of the repo
        # 2. Running the test suite
        # 3. Checking if the tests pass

        return True  # Placeholder - actual evaluation needed
