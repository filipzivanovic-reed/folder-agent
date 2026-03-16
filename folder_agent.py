#!/usr/bin/env python3
"""
Folder Agent - A CLI wrapper around mini-swe-agent with Azure OpenAI support.

Usage:
    python folder_agent.py --folder /path/to/folder --task "your instructions"
    python folder_agent.py --folder /path/to/folder --system-prompt "custom system prompt" --task "instructions"
    python folder_agent.py --folder /path/to/folder --task "instructions" --config config.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import yaml

# Try to import mini-swe-agent components
try:
    from minisweagent.agents import get_agent
    from minisweagent.environments import get_environment
    from minisweagent.models import get_model
except ImportError:
    print("Error: mini-swe-agent not installed. Run: pip install mini-swe-agent")
    sys.exit(1)


DEFAULT_SYSTEM_PROMPT = """You are a coding/file-operations agent working inside a target folder.

Follow these rules:
- Stay inside the configured working directory unless explicitly told
- First inspect the folder and understand the task.
- Make changes carefully.
- Prefer deterministic, non-interactive commands.
- At the end, return:
  1. What you changed
  2. Files touched
  3. Important outputs/results
  4. Any follow-up notes
"""

DEFAULT_INSTANCE_TEMPLATE = """Task:
{{task}}

Important execution rules:
- You are operating in a stateless shell environment.
- Directory changes are not persistent across steps.
- Always run commands in the working directory.
- When finished, output the final report by running exactly:
  echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && \\
  printf "SUMMARY\\n" && \\
  git status --short || true
"""


def load_config(config_path: str) -> dict:
    """Load and parse the YAML config file with environment variable substitution."""
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Warning: Config file {config_path} not found, using defaults")
        return {}
    
    # Read the config and substitute env vars
    content = config_file.read_text()
    
    # Simple env var substitution
    for key, value in os.environ.items():
        content = content.replace(f"${{{key}}}", value)
        content = content.replace(f"${{{key}}}", value)  # Handle ${VAR} format
    
    return yaml.safe_load(content) or {}


def get_azure_config(config: dict) -> dict:
    """Extract Azure-specific configuration."""
    azure_cfg = {}
    
    # Get from config or environment
    azure_cfg["api_key"] = config.get("azure_api_key") or os.getenv("AZURE_API_KEY")
    azure_cfg["api_base"] = config.get("azure_api_base") or os.getenv("AZURE_API_BASE")
    azure_cfg["api_version"] = config.get("azure_api_version", "2024-10-21")
    azure_cfg["deployment"] = config.get("azure_deployment", "gpt-4o")
    
    # Get model config
    model_cfg = config.get("model", {})
    azure_cfg["model_name"] = model_cfg.get("model_name", "azure/gpt-4o")
    azure_cfg["model_kwargs"] = model_cfg.get("model_kwargs", {})
    azure_cfg["cost_tracking"] = model_cfg.get("cost_tracking", "ignore_errors")
    
    # Merge API settings into model_kwargs
    if azure_cfg["api_key"]:
        azure_cfg["model_kwargs"]["api_key"] = azure_cfg["api_key"]
    if azure_cfg["api_base"]:
        azure_cfg["model_kwargs"]["api_base"] = azure_cfg["api_base"]
    if azure_cfg["api_version"]:
        azure_cfg["model_kwargs"]["api_version"] = azure_cfg["api_version"]
    
    return azure_cfg


def get_agent_config(config: dict) -> dict:
    """Extract agent configuration."""
    return config.get("agent", {"mode": "yolo"})


def get_environment_config(config: dict) -> dict:
    """Extract environment configuration."""
    return config.get("environment", {"timeout": 120})


class FolderAgent:
    """A wrapper around mini-swe-agent for working with specific folders."""
    
    def __init__(
        self,
        folder_path: str,
        config_path: str = "config.yaml",
        system_prompt: Optional[str] = None,
        instance_template: Optional[str] = None,
    ):
        self.folder_path = str(Path(folder_path).resolve())
        self.config = load_config(config_path)
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.instance_template = instance_template or DEFAULT_INSTANCE_TEMPLATE
        
        if not Path(self.folder_path).exists():
            raise ValueError(f"Folder does not exist: {self.folder_path}")
        
        # Get configurations
        self.azure_cfg = get_azure_config(self.config)
        self.agent_cfg = get_agent_config(self.config)
        self.env_cfg = get_environment_config(self.config)
    
    def build(self):
        """Build the mini-swe-agent with configured settings."""
        # Build model
        model = get_model(
            input_model_name=self.azure_cfg["model_name"],
            config={
                "cost_tracking": self.azure_cfg["cost_tracking"],
                "model_kwargs": self.azure_cfg["model_kwargs"],
            }
        )
        
        # Build environment
        env = get_environment({
            "cwd": self.folder_path,
            "timeout": self.env_cfg.get("timeout", 120),
        }, default_type="local")
        
        # Build agent
        agent = get_agent(
            model,
            env,
            {
                "system_template": self.system_prompt,
                "instance_template": self.instance_template,
            },
            default_type="default",
        )
        
        return agent
    
    def _display_trajectory(self, trajectory_file: str) -> None:
        """Parse and display trajectory in readable format."""
        try:
            with open(trajectory_file) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return

        print("\n" + "=" * 80)
        print("🔍 AGENT EXECUTION TRACE")
        print("=" * 80)

        # Extract tool calls from messages
        tool_calls = []
        tool_results = {}
        
        for msg in data.get("messages", []):
            if "tool_calls" in msg and msg["tool_calls"]:
                for call in msg["tool_calls"]:
                    func = call.get("function", {})
                    args = func.get("arguments", "{}")
                    try:
                        args_dict = json.loads(args)
                    except:
                        args_dict = {"raw": args}
                    tool_calls.append({
                        "id": call.get("id"),
                        "name": func.get("name"),
                        "args": args_dict
                    })
            elif msg.get("role") == "tool" and "content" in msg:
                tool_results[msg.get("tool_call_id")] = msg["content"]

        if tool_calls:
            print(f"\n📊 Commands Executed: {len(tool_calls)}")
            print("-" * 80)

            for i, call in enumerate(tool_calls, 1):
                print(f"\n📍 Step {i}: {call['name']}")
                
                # Display command
                if call["name"] == "bash":
                    cmd = call["args"].get("command", "")
                    print(f"   Command: {cmd}")
                
                # Display result if available
                result = tool_results.get(call["id"])
                if result:
                    print(f"   Output:")
                    for line in result.split("\n")[:15]:
                        if line.strip():
                            print(f"      {line}")
                    if len(result.split("\n")) > 15:
                        print(f"      ... ({len(result.split(chr(10))) - 15} more lines)")
        
        # Display final submission
        if "info" in data and "submission" in data["info"]:
            print(f"\n\n{'=' * 80}")
            print("📝 FINAL SUBMISSION")
            print("=" * 80)
            submission = data["info"]["submission"]
            print(submission)

    def run(self, task: str, output_file: Optional[str] = None) -> dict:
        """
        Run the agent on the given task.
        
        Args:
            task: The instruction/task for the agent
            output_file: Optional path to save the full run trajectory (defaults to trajectory.json)
            
        Returns:
            dict with keys: submission, exit_status, full_run_file
        """
        print(f"🚀 Starting folder agent on: {self.folder_path}")
        print(f"📋 Task: {task[:100]}{'...' if len(task) > 100 else ''}")
        print("-" * 50 + "\n")
        sys.stdout.flush()
        
        agent = self.build()
        
        try:
            # Track all messages to detect new ones
            last_msg_count = 0
            step_num = 0
            
            original_step = agent.step
            
            def step_wrapper():
                nonlocal last_msg_count, step_num
                original_step()
                
                # Check for new messages
                current_msg_count = len(agent.messages)
                if current_msg_count > last_msg_count:
                    # Look at messages added since last check
                    for msg in agent.messages[last_msg_count:]:
                        # Show tool calls (commands)
                        if "tool_calls" in msg and msg["tool_calls"]:
                            for call in msg["tool_calls"]:
                                step_num += 1
                                func = call.get("function", {})
                                args = func.get("arguments", "{}")
                                try:
                                    args_dict = json.loads(args)
                                except:
                                    args_dict = {}
                                
                                cmd = args_dict.get("command", "")
                                if cmd:
                                    print(f"⚡ Step {step_num}: {cmd[:80]}")
                                    sys.stdout.flush()
                        
                        # Show tool results
                        elif msg.get("role") == "tool":
                            content = msg.get("content", "")
                            if content and "<output>" in content:
                                output = content.split("<output>")[1].split("</output>")[0].strip()
                                lines = output.split("\n")
                                # Show first line or summary
                                if lines:
                                    first_line = lines[0][:70]
                                    if len(lines) > 1:
                                        print(f"   ↳ {first_line} ... ({len(lines)} total lines)")
                                    else:
                                        print(f"   ↳ {first_line}")
                                    sys.stdout.flush()
                    
                    last_msg_count = current_msg_count
            
            agent.step = step_wrapper
            
            # Run the agent
            result = agent.run(task)
            
            # Save trajectory
            traj_file = output_file or "trajectory.json"
            full = agent.serialize()
            output_path = Path(traj_file)
            output_path.write_text(json.dumps(full, indent=2))
            
            print(f"\n✓ Execution complete\n")
            sys.stdout.flush()
            
            # Display clean summary
            self._display_clean_summary(full)
            
            return {
                "submission": result.get("submission"),
                "exit_status": result.get("exit_status"),
                "full_run_file": traj_file,
                "ok": result.get("exit_status", "").lower() == "submitted",
            }
        except Exception as e:
            print(f"❌ Error running agent: {e}")
            import traceback
            traceback.print_exc()
            return {
                "submission": None,
                "exit_status": "error",
                "error": str(e),
                "ok": False,
            }
    
    def _display_clean_summary(self, trajectory_data: dict) -> None:
        """Display a clean, simplified summary of agent execution."""
        print("=" * 80)
        print("📝 EXECUTION SUMMARY")
        print("=" * 80)
        
        # Extract commands and results
        commands = []
        for msg in trajectory_data.get("messages", []):
            if "tool_calls" in msg and msg["tool_calls"]:
                for call in msg["tool_calls"]:
                    func = call.get("function", {})
                    args = func.get("arguments", "{}")
                    try:
                        args_dict = json.loads(args)
                    except:
                        args_dict = {}
                    
                    commands.append({
                        "id": call.get("id"),
                        "command": args_dict.get("command", ""),
                    })
        
        # Collect results
        results = {}
        for msg in trajectory_data.get("messages", []):
            if msg.get("role") == "tool" and "content" in msg:
                call_id = msg.get("tool_call_id")
                results[call_id] = msg["content"]
        
        # Display commands with results
        if commands:
            print(f"\nCommands executed: {len(commands)}\n")
            for i, cmd in enumerate(commands, 1):
                print(f"{i}. $ {cmd['command']}")
                result = results.get(cmd["id"], "")
                if result:
                    # Parse XML-like result format
                    if "<output>" in result:
                        output = result.split("<output>")[1].split("</output>")[0].strip()
                    else:
                        output = result.strip()
                    
                    if output:
                        # Show first 3 lines of output
                        lines = output.split("\n")
                        for line in lines[:3]:
                            if line.strip():
                                print(f"   {line}")
                        if len(lines) > 3:
                            print(f"   ... ({len(lines) - 3} more lines)")
                print()
        
        # Show final result
        info = trajectory_data.get("info", {})
        if "submission" in info:
            print("=" * 80)
            print("✅ RESULT")
            print("=" * 80)
            print(info["submission"])


def main():
    parser = argparse.ArgumentParser(
        description="Folder Agent - Run mini-swe-agent on a specific folder with Azure OpenAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --folder ./my-project --task "Read the README and summarize"
  %(prog)s --folder /path/to/code --task "Fix all TODO comments" --system-prompt "You are a code reviewer"
  %(prog)s --folder ./project --task "Run tests" --config custom-config.yaml

Environment variables:
  AZURE_API_KEY      Your Azure OpenAI API key
  AZURE_API_BASE     Your Azure OpenAI endpoint (https://xxx.openai.azure.com)
  AZURE_API_VERSION  API version (default: 2024-10-21)
        """
    )
    
    parser.add_argument(
        "--folder", "-f",
        default=".",
        help="Path to the target folder to work on (default: current directory)"
    )
    
    parser.add_argument(
        "--task", "-t",
        required=True,
        help="Task/instructions for the agent"
    )
    
    parser.add_argument(
        "--system-prompt", "-s",
        default=None,
        help="Custom system prompt (default: built-in coding agent prompt)"
    )
    
    parser.add_argument(
        "--instance-template", "-i",
        default=None,
        help="Custom instance template with {{task}} placeholder"
    )
    
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save full trajectory JSON (optional)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress startup messages"
    )
    
    args = parser.parse_args()
    
    # Validate folder
    if not Path(args.folder).exists():
        print(f"Error: Folder not found: {args.folder}")
        sys.exit(1)
    
    if not Path(args.folder).is_dir():
        print(f"Error: Not a directory: {args.folder}")
        sys.exit(1)
    
    # Create agent
    try:
        agent = FolderAgent(
            folder_path=args.folder,
            config_path=args.config,
            system_prompt=args.system_prompt,
            instance_template=args.instance_template,
        )
    except Exception as e:
        print(f"Error initializing agent: {e}")
        sys.exit(1)
    
    # Run agent
    result = agent.run(args.task, output_file=args.output)
    
    # Output results
    print("\n" + "=" * 50)
    print("📊 RESULT")
    print("=" * 50)
    print(f"Status: {'✅ Success' if result['ok'] else '❌ Failed'}")
    print(f"Exit status: {result['exit_status']}")
    
    if result.get("full_run_file"):
        print(f"Trajectory: {result['full_run_file']}")
    
    if result.get("submission"):
        print("\n📝 Submission:")
        print("-" * 30)
        print(result["submission"])
    
    if result.get("error"):
        print(f"\n❌ Error: {result['error']}")
    
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
