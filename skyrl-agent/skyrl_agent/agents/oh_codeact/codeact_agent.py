"""OpenHands CodeAct Agent for RL Training.

This module provides a CodeAct-style agent implementation using the OpenHands SDK
with a custom AsyncInferBackend for RL training with VERL.
"""

from typing import Any, List, Optional, Callable
import json
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import traceback
import copy
from collections import deque

from skyrl_agent.functional.function_calling import convert_str_to_completion_format
from skyrl_agent.functional.chat_template import get_templates_path
from skyrl_agent.config.configuration_utils import TrajectoryConfig
from skyrl_agent.integrations.base import AsyncInferBackend
from skyrl_agent.dispatcher.async_utils import call_async_from_sync

# OpenHands SDK imports
from openhands.sdk import Agent, LLM, LocalConversation, get_logger
from openhands.sdk.agent.agent import (
    prepare_llm_messages,
    ConversationExecutionStatus,
    ConversationState,
    FinishAction,
)
from openhands.sdk.event import MessageEvent, ActionEvent, ObservationEvent, Event
from openhands.sdk.llm import Message, TextContent
from openhands.tools.preset.default import get_default_tools

import logging

logger = logging.getLogger(__name__)


class OHCodeActAgent(Agent):
    """CodeAct Agent for RL training using OpenHands SDK.

    This agent extends the SDK's Agent to:
    - Use AsyncInferBackend instead of the SDK's LLM for model inference
    - Maintain manual message history for trajectory tracking
    - Support context window management for long sequences
    - Integrate with VERL PPO training pipeline
    """

    # Custom fields for RL training (not part of base Agent)
    # Use underscore prefix for Pydantic compatibility (frozen model)
    _infer_engine: Optional[AsyncInferBackend] = None
    _tokenizer: Optional[Any] = None
    _traj_config: Optional[TrajectoryConfig] = None
    _max_prompt_length: int = 32768
    _step_count: int = 0
    _sampling_params: dict = None
    _instance_id: str = None
    _trajectory_id: int = 0
    _qwen3_enable_thinking: bool = False
    _qwen3_acc_thinking: bool = False
    _agent_id: str = None

    # Additional instance state (also private for frozen model)
    _messages: List[dict] = None
    _prompt_token_len: int = 0
    _response_token_len: int = 0
    _pending_actions: deque = None
    _workspace: Any = None
    _conversation: Any = None
    _max_iterations: int = 30

    def __init__(
        self,
        traj_config: TrajectoryConfig,
        infer_engine: Optional[AsyncInferBackend] = None,
        tokenizer: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """Initialize a CodeAct agent instance for RL training.

        Args:
            traj_config: Configuration for the trajectory.
            infer_engine: Async inference backend for RL training.
            tokenizer: Tokenizer for encoding prompts.
        """
        # Create a dummy LLM for SDK compatibility
        # We won't use it for inference - our custom backend handles that
        dummy_llm = LLM(
            model="dummy-model",
            api_key="dummy",
            max_message_chars=128000,
        )

        # Get default tools
        tools = get_default_tools(
            enable_browser=traj_config.tools.get("enable_browsing", False),
        )

        # Initialize parent Agent class
        super().__init__(
            llm=dummy_llm,
            tools=tools,
            **kwargs,
        )

        # Store our custom inference components
        # Use object.__setattr__ to bypass Pydantic frozen model restrictions
        object.__setattr__(self, '_infer_engine', infer_engine)
        object.__setattr__(self, '_tokenizer', tokenizer)
        object.__setattr__(self, '_traj_config', traj_config)
        object.__setattr__(self, '_max_prompt_length', traj_config.max_prompt_length)
        object.__setattr__(self, '_step_count', 0)
        object.__setattr__(self, '_sampling_params', traj_config.sampling_params)

        # Store instance and trajectory IDs
        object.__setattr__(self, '_instance_id', traj_config.instance_id)
        object.__setattr__(self, '_trajectory_id', traj_config.trajectory_id)
        object.__setattr__(self, '_qwen3_enable_thinking', traj_config.qwen3_enable_thinking)
        object.__setattr__(self, '_qwen3_acc_thinking', traj_config.qwen3_acc_thinking)

        # Agent ID for request tracking
        object.__setattr__(self, '_agent_id', uuid4().hex)

        # Manual message tracking for trajectory recording
        object.__setattr__(self, '_messages', [])
        object.__setattr__(self, '_prompt_token_len', 0)
        object.__setattr__(self, '_response_token_len', 0)

        # Pending actions queue
        object.__setattr__(self, '_pending_actions', deque())

        # Workspace and conversation (set during initialization)
        object.__setattr__(self, '_workspace', None)
        object.__setattr__(self, '_conversation', None)
        max_iter = traj_config.max_iterations if hasattr(traj_config, 'max_iterations') else 30
        object.__setattr__(self, '_max_iterations', max_iter)

    def _encode_prompt(self, messages: List[dict]) -> List[int]:
        """Encode messages to token IDs using the tokenizer.

        Args:
            messages: List of message dictionaries.

        Returns:
            List of token IDs.
        """
        if self._qwen3_acc_thinking:
            # Use the Qwen3 thinking mode chat template
            assert self._qwen3_enable_thinking, "Qwen3 thinking mode should for accumulating thinking."
            chat_template = get_templates_path() / "qwen3_acc_thinking.jinja2"
            input_ids = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=self._qwen3_enable_thinking,
                chat_template=chat_template.read_text(),
            )
        else:
            input_ids = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=self._qwen3_enable_thinking
            )
        return input_ids

    def step(
        self,
        conversation: LocalConversation,
        on_event: Callable[[Event], None],
        on_token: Optional[Callable] = None,
    ) -> None:
        """Generate a response using the custom inference backend.

        This overrides the SDK's step() method to use our AsyncInferBackend
        instead of the SDK's LLM, enabling RL training with VERL.

        Args:
            conversation: The conversation context.
            on_event: Callback to emit events.
            on_token: Optional callback for streaming tokens.
        """
        object.__setattr__(self, '_step_count', self._step_count + 1)
        print(f"instance id {self._instance_id}, trajectory {self._trajectory_id}, step {self._step_count}")

        state = conversation.state

        # Check for pending actions and execute them first
        pending_actions = ConversationState.get_unmatched_actions(state.events)
        if pending_actions:
            logger.info(f"Executing {len(pending_actions)} pending action(s)")
            self._execute_actions(conversation, pending_actions, on_event)
            return

        # Check for exit/finish conditions
        for event in reversed(list(state.events)):
            if isinstance(event, MessageEvent) and event.source == "user":
                if hasattr(event, 'llm_message') and event.llm_message:
                    content = ""
                    for c in event.llm_message.content:
                        if isinstance(c, TextContent):
                            content += c.text
                    if content.strip() == "/exit":
                        state.execution_status = ConversationExecutionStatus.FINISHED
                        return
                break

        # Prepare messages for LLM
        _messages_or_condensation = prepare_llm_messages(
            state.events, condenser=self.condenser, llm=self.llm
        )

        # Handle condensation if needed
        if hasattr(_messages_or_condensation, 'type') and _messages_or_condensation.type == 'condensation':
            on_event(_messages_or_condensation)
            return

        llm_messages = _messages_or_condensation

        # Convert SDK messages to dict format for our tokenizer
        messages = self._convert_messages_to_dict(llm_messages)

        # Initialize or update our message history for trajectory tracking
        if len(self._messages) == 0:
            object.__setattr__(self, '_messages', messages)
        else:
            if messages:
                obs = messages[-1]
                if obs.get("role") == "user":
                    remaining_steps = self._max_iterations - self._step_count + 1
                    if remaining_steps > 1:
                        obs["content"] += f"\nSteps remaining: {remaining_steps}."
                    else:
                        obs["content"] += "\nThis is your last step, make sure to use the finish tool to submit your final answer."
                    self._messages.append(obs)
                    print(f"Obs: {obs['content'][:200]}...")

        try:
            # Encode prompt using our tokenizer
            input_ids = self._encode_prompt(self._messages)

            # Track token lengths for context window management
            if len(self._messages) == 2:
                # system + first user message
                object.__setattr__(self, '_prompt_token_len', len(input_ids))
            else:
                object.__setattr__(self, '_response_token_len', len(input_ids) - self._prompt_token_len)

            # Check context window limits
            if self._response_token_len >= self._max_prompt_length - 3000:
                self._messages[-1]["content"] += "\nNote: You are running out of tokens, submit your solution through finish tool now."
                input_ids = self._encode_prompt(self._messages)
                object.__setattr__(self, '_response_token_len', len(input_ids) - self._prompt_token_len)

            if self._response_token_len >= self._max_prompt_length:
                # Emit finish action for context exceeded
                finish_event = ActionEvent(
                    source="agent",
                    action=FinishAction(thought="CONTEXT_WINDOW_EXCEEDED"),
                )
                on_event(finish_event)
                state.execution_status = ConversationExecutionStatus.FINISHED
                return

            # Set up sampling params with remaining context
            sampling_params = copy.deepcopy(self._sampling_params)
            sampling_params["max_tokens"] = self._max_prompt_length - self._response_token_len

            # Generate response using our custom inference backend
            response_str, meta_info = call_async_from_sync(
                self._infer_engine.async_generate_ids,
                input_ids=input_ids,
                sampling_params=sampling_params,
                request_id=self._agent_id,
            )
            stop_reason = meta_info.get("finish_reason")
            print(f"instance id {self._instance_id}, trajectory {self._trajectory_id}, response {response_str[:200] if response_str else 'None'}... stop reason {stop_reason}")

            if not response_str:
                finish_event = ActionEvent(
                    source="agent",
                    action=FinishAction(thought="BAD_LLM_RESPONSE"),
                )
                on_event(finish_event)
                state.execution_status = ConversationExecutionStatus.FINISHED
                return

            # Store response in our message history
            self._messages.append({"role": "assistant", "content": response_str})

            if stop_reason == "length":
                finish_event = ActionEvent(
                    source="agent",
                    action=FinishAction(thought="CONTEXT_WINDOW_EXCEEDED"),
                )
                on_event(finish_event)
                state.execution_status = ConversationExecutionStatus.FINISHED
                return

            # Parse the response to extract tool calls/actions
            # The SDK's Agent has built-in tool parsing
            message = Message(
                role="assistant",
                content=[TextContent(text=response_str)],
            )

            # Create a message event for the response
            response_event = MessageEvent(
                source="agent",
                llm_message=message,
            )
            on_event(response_event)

            # Parse actions from response (tool calls)
            actions = self._parse_actions_from_response(response_str)

            for action in actions:
                action_event = ActionEvent(source="agent", action=action)
                on_event(action_event)

            if not actions:
                # No action detected, prompt user
                error_msg = MessageEvent(
                    source="user",
                    llm_message=Message(
                        role="user",
                        content=[TextContent(
                            text="No function call detected. Please use a tool to continue."
                        )],
                    ),
                )
                on_event(error_msg)

        except Exception as e:
            logger.error(f"Error in agent step: {str(e)}")
            logger.debug(f"{traceback.format_exc()}")
            error_msg = MessageEvent(
                source="user",
                llm_message=Message(
                    role="user",
                    content=[TextContent(
                        text=f"An error: {str(e)} encountered. Please try a different approach."
                    )],
                ),
            )
            on_event(error_msg)

    def _convert_messages_to_dict(self, llm_messages: List[Message]) -> List[dict]:
        """Convert SDK Message objects to dict format for tokenizer.

        Args:
            llm_messages: List of SDK Message objects.

        Returns:
            List of message dictionaries.
        """
        messages = []
        for msg in llm_messages:
            content = ""
            for c in msg.content:
                if isinstance(c, TextContent):
                    content += c.text
            messages.append({
                "role": msg.role,
                "content": content,
            })
        return messages

    def _parse_actions_from_response(self, response: str) -> List:
        """Parse tool calls/actions from the response.

        Args:
            response: The LLM response string.

        Returns:
            List of parsed actions.
        """
        # Use SDK's tool parsing if available
        # For now, check for finish action
        actions = []

        if "<finish>" in response.lower() or "finish(" in response.lower():
            actions.append(FinishAction())

        # TODO: Parse other tool calls from response
        # The SDK should handle this through its tool system

        return actions

    def _execute_actions(
        self,
        conversation: LocalConversation,
        actions: List,
        on_event: Callable[[Event], None],
    ) -> None:
        """Execute pending actions.

        Args:
            conversation: The conversation context.
            actions: List of actions to execute.
            on_event: Callback to emit events.
        """
        for action in actions:
            # The SDK's workspace handles action execution
            try:
                if hasattr(conversation, 'workspace') and conversation.workspace:
                    result = conversation.workspace.execute(action)
                    if result:
                        on_event(result)
            except Exception as e:
                logger.error(f"Error executing action: {e}")

    def get_final_messages(self) -> List[dict]:
        """Get the final messages for this agent trajectory.

        Returns:
            List of message dictionaries for trajectory logging.
        """
        # Log trace for debugging
        try:
            import wandb
            current_step = wandb.run.step if wandb.run else 1
            run_name = wandb.run.name if wandb.run else "no_run"
            logger.info(f"Detected run name: {run_name}")

            if (current_step == 1) or (current_step % 10 == 0):
                instance_dir = (
                    Path(f"./trace/{run_name}/step{current_step}") / str(self._instance_id) / str(self._trajectory_id)
                )
                instance_dir.mkdir(exist_ok=True, parents=True)

                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
                trace_file = instance_dir / f"trace_{timestamp}.json"

                with open(trace_file, "w") as f:
                    result_json = json.dumps(self._messages, default=lambda x: str(x))
                    f.write(result_json)
        except Exception as e:
            logger.warning(f"Failed to save trace: {e}")

        return self._messages

    def is_finished(self, conversation: LocalConversation) -> bool:
        """Check if the agent has finished.

        Args:
            conversation: The conversation context.

        Returns:
            True if finished.
        """
        state = conversation.state
        return state.execution_status == ConversationExecutionStatus.FINISHED

    def close(self) -> None:
        """Close the agent and release resources."""
        if self._workspace:
            try:
                self._workspace.close()
            except Exception as e:
                logger.warning(f"Error closing workspace: {e}")
            object.__setattr__(self, '_workspace', None)
