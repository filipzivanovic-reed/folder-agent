#!/bin/bash

# Folder Agent Installation Script
# This script installs or updates Folder Agent as a global uv tool

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Folder Agent Installation Script${NC}"
echo -e "${BLUE}================================${NC}\n"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}✗ Error: uv is not installed${NC}"
    echo -e "${YELLOW}Please install uv first: https://docs.astral.sh/uv/getting-started/installation/${NC}"
    exit 1
fi

echo -e "${GREEN}✓ uv found: $(uv --version)${NC}\n"

# Check if we're in the folder-agent repository
if [ ! -f "$SCRIPT_DIR/pyproject.toml" ]; then
    echo -e "${RED}✗ Error: pyproject.toml not found in $SCRIPT_DIR${NC}"
    echo -e "${YELLOW}Please run this script from the Folder Agent repository root${NC}"
    exit 1
fi

echo -e "${BLUE}Installing Folder Agent from: $SCRIPT_DIR${NC}\n"

# Install or update the tool using uv tool install
echo -e "${YELLOW}Installing Folder Agent as global command...${NC}"
uv tool install --editable "$SCRIPT_DIR" --force
echo -e "${GREEN}✓ Folder Agent installed successfully${NC}"

echo ""

# Verify installation
if command -v folder-agent &> /dev/null; then
    echo -e "${GREEN}✓ Verification successful!${NC}"
    echo -e "${BLUE}Location: $(which folder-agent)${NC}"
    echo ""
    echo -e "${BLUE}You can now use Folder Agent from any directory:${NC}"
    echo ""
    echo -e "${GREEN}  folder-agent -t \"your task\"${NC} (or ${GREEN}fa -t \"your task\"${NC})"
    echo -e "${YELLOW}    Run task on current directory${NC}"
    echo ""
    echo -e "${GREEN}  folder-agent -f /path/to/folder -t \"your task\"${NC}"
    echo -e "${YELLOW}    Run task on specific folder${NC}"
    echo ""
    echo -e "${GREEN}  folder-agent -t \"task\" -s \"You are a code reviewer\"${NC}"
    echo -e "${YELLOW}    Run with custom system prompt${NC}"
    echo ""
    echo -e "${GREEN}  folder-agent -t \"task\" -c config.yaml${NC}"
    echo -e "${YELLOW}    Run with custom config file${NC}"
    echo ""
    echo -e "${GREEN}  folder-agent --help${NC}"
    echo -e "${YELLOW}    Show all available options${NC}"
else
    echo -e "${RED}✗ Verification failed: folder-agent command not found${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}================================${NC}"
echo -e "${GREEN}Installation complete!${NC}"
echo -e "${BLUE}================================${NC}"
