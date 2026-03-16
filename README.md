# Folder Agent

A CLI wrapper around [mini-swe-agent](https://github.com/klieret/mini-swe-agent) with Azure OpenAI support. Allows you to give a folder and instructions, then the agent runs autonomously and returns results.

## Setup

1. **Install dependencies:**
   ```bash
   pip install mini-swe-agent pyyaml
   ```

2. **Configure Azure OpenAI:**
   
   Either set environment variables:
   ```bash
   export AZURE_API_KEY="your-key"
   export AZURE_API_BASE="https://your-resource.openai.azure.com"
   export AZURE_API_VERSION="2024-10-21"
   ```
   
   Or edit `config.yaml` and reference them:
   ```yaml
   azure_api_key: "${AZURE_API_KEY}"
   azure_api_base: "${AZURE_API_BASE}"
   azure_deployment: "gpt-4o"  # your deployment name
   ```

## Usage

### Basic usage:
```bash
# Run a task on a folder
./folder-agent --folder ./my-project --task "Read the README and summarize what this project does"

# Short form
./folder-agent -f ./my-project -t "List all Python files in this directory"
```

### With custom system prompt:
```bash
./folder-agent -f ./code --system-prompt "You are a code reviewer. Focus on security issues." -t "Review the auth.py file"
```

### Save trajectory for debugging:
```bash
./folder-agent -f ./project -t "Fix the bug" -o run_trajectory.json
```

### Using a custom config:
```bash
./folder-agent -f ./project -t "Your task" -c my-config.yaml
```

## Python API

You can also use it as a Python library:

```python
from folder_agent import FolderAgent

agent = FolderAgent(
    folder_path="./my-project",
    config_path="config.yaml",
)

result = agent.run("Read the README and summarize")
print(result["submission"])
```

## Files

- `folder_agent.py` - Main Python wrapper with CLI
- `folder-agent` - Bash wrapper script
- `config.yaml` - Azure OpenAI configuration
- `.env.example` - Environment variable template
