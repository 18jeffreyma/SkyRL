"""CodeAct Trajectory Runner for RL Training.

This module provides the trajectory runner that coordinates agent execution
with runtime environments for RL training with SWE-Bench tasks.
"""

from typing import Any, Dict, Callable, List
import pandas as pd
import traceback

from skyrl_agent.agents.oh_codeact.codeact_agent import OHCodeActAgent
from skyrl_agent.dispatcher.async_utils import call_sync_from_async
from skyrl_agent.config.configuration_utils import TrajectoryConfig
from skyrl_agent.agents.base import (
    BaseTrajectory,
    TrajectoryResult,
    TrajectoryConfig,
    AsyncInferBackend,
    AutoTokenizer
)

# OpenHands SDK imports
from openhands.sdk import LocalConversation, Conversation, get_logger
from openhands.sdk.event import Event, MessageEvent, ActionEvent
from openhands.sdk.llm import Message, TextContent
from openhands.sdk.agent.agent import ConversationExecutionStatus
from openhands.workspace import APIRemoteWorkspace
from openhands.sdk.workspace import LocalWorkspace, RemoteWorkspace

import logging
from dataclasses import dataclass
import os

from skyrl_agent.tasks.swebench.utils import SWEBenchTask

logger = logging.getLogger(__name__)


class LocalWorkspaceAdapter(LocalWorkspace):
    """Adapter that wraps APIRemoteWorkspace to satisfy LocalConversation's type check.

    This adapter makes a remote workspace appear as a LocalWorkspace for the SDK's
    assertion check, while delegating all operations to the underlying remote workspace.
    This is needed because:
    - LocalConversation requires LocalWorkspace (assertion at line 124)
    - RemoteConversation requires sending agent to server (incompatible with custom agents)
    - Our use case is hybrid: local agent execution + remote workspace operations
    """

    _remote_workspace: APIRemoteWorkspace = None

    def __init__(self, remote_workspace: APIRemoteWorkspace):
        """Initialize adapter wrapping a remote workspace.

        Args:
            remote_workspace: The APIRemoteWorkspace to wrap.
        """
        # Use a temporary local directory for LocalConversation's state management
        # The remote workspace's working_dir (/testbed) is only for remote command execution
        import tempfile
        local_temp_dir = tempfile.mkdtemp(prefix="skyrl_workspace_")
        super().__init__(working_dir=local_temp_dir)
        object.__setattr__(self, '_remote_workspace', remote_workspace)

    def execute_command(self, command: str, timeout: int = 120, **kwargs):
        """Execute command via remote workspace."""
        return self._remote_workspace.execute_command(command, timeout=timeout, **kwargs)

    def file_upload(self, local_path: str, remote_path: str, **kwargs):
        """Upload file via remote workspace."""
        return self._remote_workspace.file_upload(local_path, remote_path, **kwargs)

    def file_download(self, remote_path: str, local_path: str, **kwargs):
        """Download file via remote workspace."""
        return self._remote_workspace.file_download(remote_path, local_path, **kwargs)

    def file_read(self, path: str, **kwargs):
        """Read file via remote workspace."""
        return self._remote_workspace.file_read(path, **kwargs)

    def file_write(self, path: str, content: str, **kwargs):
        """Write file via remote workspace."""
        return self._remote_workspace.file_write(path, content, **kwargs)

    def file_delete(self, path: str, **kwargs):
        """Delete file via remote workspace."""
        return self._remote_workspace.file_delete(path, **kwargs)

    def file_exists(self, path: str, **kwargs) -> bool:
        """Check if file exists via remote workspace."""
        return self._remote_workspace.file_exists(path, **kwargs)

    def list_dir(self, path: str, **kwargs):
        """List directory via remote workspace."""
        return self._remote_workspace.list_dir(path, **kwargs)

    def git_diff(self, path: str, **kwargs):
        """Get git diff via remote workspace."""
        return self._remote_workspace.git_diff(path, **kwargs)

    def git_changes(self, path: str, **kwargs):
        """Get git changes via remote workspace."""
        return self._remote_workspace.git_changes(path, **kwargs)

    def pause(self):
        """Pause the remote workspace container."""
        return self._remote_workspace.pause()

    def resume(self):
        """Resume the remote workspace container."""
        return self._remote_workspace.resume()

    @property
    def alive(self) -> bool:
        """Check if remote workspace is alive."""
        return self._remote_workspace.alive

    def close(self):
        """Close the remote workspace."""
        # APIRemoteWorkspace uses cleanup() instead of close()
        if hasattr(self._remote_workspace, 'cleanup'):
            self._remote_workspace.cleanup()
        elif hasattr(self._remote_workspace, 'close'):
            self._remote_workspace.close()


@dataclass
class TaskHandle:
    """Handle for tracking task execution."""
    instance_id: str
    trajectory_id: int
    batch_id: int


def is_fatal_evaluation_error(error: str | None) -> bool:
    """Check if an error is fatal and should stop execution.

    Args:
        error: Error message string.

    Returns:
        True if the error is fatal.
    """
    if not error:
        return False

    FATAL_EXCEPTIONS = [
        "RuntimeError",
        "ConnectionError",
        "TimeoutError",
        "WorkspaceError",
    ]

    if any(exception in error for exception in FATAL_EXCEPTIONS):
        logger.error(f"Fatal evaluation error detected: {error}")
        return True

    return False


class CodeActTrajectory(BaseTrajectory):
    """Trajectory runner for CodeAct agent with SWE-Bench tasks.

    This class coordinates the lifecycle of a single trajectory:
    1. Initialize runtime environment
    2. Generate trajectory (agent-environment interaction loop)
    3. Evaluate results
    """

    def __init__(
        self,
        cfg: TrajectoryConfig,
        data: Dict[str, Any],
        infer_engine: AsyncInferBackend,
        tokenizer: AutoTokenizer,
        task: SWEBenchTask,
        val_mode: bool,
    ) -> None:
        """Initialize trajectory runner.

        Args:
            cfg: Trajectory configuration.
            data: Task data dictionary.
            infer_engine: Async inference backend.
            tokenizer: Tokenizer for encoding.
            task: SWE-Bench task instance.
            val_mode: Whether in validation mode.
        """
        super().__init__(cfg, data, infer_engine, tokenizer, task, val_mode)
        assert isinstance(task, SWEBenchTask)
        self.events: List[Event] = []
        self.last_error: str | None = None

    def _event_callback(self, event: Event) -> None:
        """Callback for conversation events.

        Args:
            event: The event from the conversation.
        """
        self.events.append(event)
        event_type = type(event).__name__
        logger.debug(f"Event received: {event_type}")

    async def initialize_trajectory(self):
        """Initialize the runtime for this trajectory."""
        assert isinstance(self.task, SWEBenchTask)

        batch_id = self.cfg.instance_id
        trajectory_id = self.cfg.trajectory_id

        data = self.data
        instance_id = data["instance_id"] if data["instance_id"] else batch_id
        instance = pd.Series(data["instance"])
        data_source = data["data_source"]

        # Create agent
        self.agent = OHCodeActAgent(
            traj_config=self.cfg,
            infer_engine=self.infer_engine,
            tokenizer=self.tokenizer
        )

        init_successful = False
        try:
            # Initialize workspace using SDK
            workspace = await self.task.initialize_workspace(
                instance, data_source, self.cfg.max_iterations
            )

            # Store workspace in agent (use object.__setattr__ for frozen Pydantic model)
            object.__setattr__(self.agent, '_workspace', workspace)

            # Get the instruction for this task
            instruction = self.task.get_instruction(instance, data_source)
            object.__setattr__(self.agent, 'instruction', instruction)

            init_successful = True
            logger.info(f"Successfully initialized workspace for instance {instance_id}")

        except Exception as e:
            logger.error(f"Failed to initialize workspace for instance {instance_id}: {str(e)}")
            object.__setattr__(self.agent, '_workspace', None)

            return_val = {
                "instance_id": instance_id,
                "trajectory_id": trajectory_id,
                "messages": [],
                "state": None,
                "results": None,
                "error": str(e),
                "finish": False,
                "finish_reason": "error_initialization",
            }

            self.result = return_val

        finally:
            if not init_successful:
                logger.info(
                    f"Init failed. Running cleanup for init agent task for instance {instance_id}, trajectory {trajectory_id}"
                )
                if self.agent._workspace:
                    try:
                        self.agent._workspace.close()
                    except Exception:
                        pass

    async def generate_trajectory(self) -> None:
        """Run the agent-environment interaction loop."""
        assert isinstance(self.task, SWEBenchTask)

        data = self.data
        instance_id = data["instance_id"] if data["instance_id"] else self.cfg.instance_id
        trajectory_id = self.cfg.trajectory_id
        instance = pd.Series(data["instance"])
        data_source = data["data_source"]
        agent = self.agent
        workspace = agent._workspace  # Access private attribute

        try:
            if not workspace:
                raise Exception(
                    f"Workspace not initialized for instance {instance_id}, trajectory {trajectory_id}"
                )

            # Create conversation with SDK
            # For remote workspaces, wrap in LocalWorkspaceAdapter to satisfy
            # LocalConversation's type check while delegating operations to the remote
            self.events = []
            if isinstance(workspace, (APIRemoteWorkspace, RemoteWorkspace)):
                # Wrap remote workspace in adapter for LocalConversation compatibility
                workspace_for_conversation = LocalWorkspaceAdapter(workspace)
            else:
                workspace_for_conversation = workspace

            conversation = LocalConversation(
                agent=agent,
                workspace=workspace_for_conversation,
                callbacks=[self._event_callback],
                max_iteration_per_run=self.cfg.max_iterations,
            )

            # Send initial instruction
            conversation.send_message(agent.instruction)

            # Run the conversation loop
            conversation.run()

            # Check for fatal errors
            if self.last_error and is_fatal_evaluation_error(self.last_error):
                raise Exception("Fatal error: " + self.last_error)

            # Get final messages from agent
            final_messages = agent.get_final_messages()

            # Complete the task (get git patch, etc.)
            result = await call_sync_from_async(
                self.task.complete_runtime, workspace, instance, data_source
            )

            # Determine finish status
            finish = agent.is_finished(conversation)
            finish_reason = self._determine_finish_reason(conversation)

            if "finish_reason" in result:
                finish_reason = result["finish_reason"]

            return_val = TrajectoryResult(
                {
                    "instance_id": instance_id,
                    "trajectory_id": trajectory_id,
                    "messages": final_messages,
                    "state": conversation.state,
                    "results": result,
                    "error": self.last_error,
                    "finish": finish,
                    "finish_reason": finish_reason,
                }
            )

        except Exception as e:
            logger.error(f"Run error {instance_id}: {e}")
            logger.debug(f"Full Traceback: {traceback.format_exc()}")
            final_messages = agent.get_final_messages() if agent else []

            if not final_messages or len(final_messages) == 0:
                logger.debug(
                    f"Final messages are non-existent (or empty) for instance {instance_id}, trajectory {trajectory_id}"
                )

            return_val = TrajectoryResult(
                {
                    "instance_id": instance_id,
                    "trajectory_id": trajectory_id,
                    "messages": final_messages,
                    "state": None,
                    "results": None,
                    "error": str(e),
                    "finish": False,
                    "finish_reason": "error_runtime",
                }
            )

        finally:
            logger.info(
                f"Running cleanup for run agent task for instance {instance_id}, trajectory {trajectory_id}"
            )
            self._cleanup_agent()

        self.result = return_val

    def _determine_finish_reason(self, conversation: LocalConversation) -> str | None:
        """Determine the reason for conversation finish.

        Args:
            conversation: The conversation context.

        Returns:
            The finish reason string.
        """
        state = conversation.state

        # Check events for finish actions
        for event in reversed(self.events):
            if isinstance(event, ActionEvent):
                action = event.action
                if hasattr(action, 'thought') and action.thought:
                    if action.thought in ["CONTEXT_WINDOW_EXCEEDED", "BAD_LLM_RESPONSE"]:
                        return action.thought

        # Check if max iterations reached
        if self.agent.step_count >= self.cfg.max_iterations:
            return "max_iterations_reached"

        return "FINISH_TOOL"

    async def evaluate_trajectory(self) -> None:
        """Evaluate the trajectory results."""
        assert isinstance(self.task, SWEBenchTask)

        batch_id = self.cfg.instance_id
        trajectory_id = self.cfg.trajectory_id
        data = self.data
        instance_id = data["instance_id"] if data["instance_id"] else batch_id
        instance = pd.Series(data["instance"])
        data_source = data["data_source"]

        try:
            results = self.result.get("results", None)
            if not results:
                raise Exception(
                    f"No results found for instance {instance_id}, trajectory {trajectory_id}"
                )

            # Check if reward already computed
            if "reward" in results:
                self.result["reward"] = results["reward"]
                return

            # Evaluate results
            eval_results = await self.task.evaluate_result(
                instance, results, instance_id, trajectory_id, data_source
            )
            self.result["reward"] = eval_results

            logger.info(
                f"Successfully evaluated instance {instance_id}, trajectory {trajectory_id} with reward {eval_results}"
            )

        except Exception as e:
            logger.error(
                f"Failed to evaluate traj {trajectory_id} for instance {instance_id}: {str(e)}"
            )
            self.result["reward"] = False
            self.result["eval_error"] = str(e)
            self.result["finish_reason"] = (
                "error_evaluation" if "No git patch found" not in str(e) else "no_git_patch"
            )

    def _cleanup_agent(self):
        """Clean up agent resources."""
        try:
            self.agent.close()
        except Exception as e:
            logger.warning(
                f"Error closing agent {self.cfg.instance_id}, trajectory {self.cfg.trajectory_id}: {str(e)}"
            )
