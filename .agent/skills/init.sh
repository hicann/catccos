#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
set -e

# --- Color & output helpers ---
if [ -t 1 ]; then
    GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'
    CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'
else
    GREEN=''; YELLOW=''; RED=''; CYAN=''; BOLD=''; DIM=''; NC=''
fi

ok()   { echo -e "  ${GREEN}✓${NC} $*"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $*"; }
err()  { echo -e "  ${RED}✗${NC} $*"; }
info() { echo -e "  ${CYAN}→${NC} $*"; }
step() { echo -e "${DIM}$*${NC}"; }

# --- Constants ---
BRAND="catccos-agent"
VERSION="1.0.0"

# All skills to install (directory basenames under skills/)
INCLUDED_SKILLS="orchestrator requirement-analyzer architecture-designer kernel-generator example-scaffolder torch-binding verifier test-integrator"

# Knowledge base (always installed alongside skills)
KNOWLEDGE_BASE="knowledge-base"

show_banner() {
    echo ""
    echo -e "${CYAN}"
    cat << 'BANNER'
   ___   _  _____ ___ ___ ___  ___     _                _
  / __| /_\|_   _/ __/ __/ _ \/ __|   /_\  __ _ ___ _ _| |_
 | (__ / _ \ | || (__| (_| (_) \__ \  / _ \/ _` / -_) ' \  _|
  \___/_/ \_\|_| \___\___\___/|___/ /_/ \_\__, \___|_||_\__|
                                          |___/
BANNER
    echo -e "${NC}"
    echo -e "  ${BOLD}CATCCOS Operator Agentic Development Framework${NC}"
    echo ""
}

show_help() {
    cat << 'EOF'
CATCCOS Agent - Skills Installer

Usage: init.sh [level] [tool] [install_path]

Arguments:
  level        - "project" (default) or "global"
  tool         - "gemini" (default), "claude", "cursor", "trae", or "copilot"
  install_path - Target project directory (default: repository root)

Options:
  --help       - Show this help message

Examples:
  init.sh                              # Project-level, Gemini CLI
  init.sh project gemini               # Project-level, Gemini CLI
  init.sh project claude               # Project-level, Claude Code
  init.sh project cursor               # Project-level, Cursor
  init.sh project trae                 # Project-level, Trae
  init.sh project copilot              # Project-level, GitHub Copilot
  init.sh global claude                # Global-level, Claude Code
  init.sh project gemini /path/to/proj # Custom install path

Installation paths:
  Gemini CLI:  .gemini/skills/          + GEMINI.md
  Claude:      .claude/skills/          + CLAUDE.md
  Cursor:      .cursor/skills/          + .cursorrules
  Trae:        .trae/skills/            + AGENTS.md
  Copilot:     .github/skills/          + AGENTS.md

Each skill is symlinked individually so your existing config is preserved.
EOF
}

# --- Argument parsing ---
LEVEL="project"
TOOL="gemini"
INSTALL_PATH=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$SCRIPT_DIR"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

for arg in "$@"; do
    case "$arg" in
        --help)           show_help; exit 0 ;;
        global|project)   LEVEL="$arg" ;;
        gemini|claude|trae|cursor|copilot) TOOL="$arg" ;;
    esac
done

# Last argument as install_path if it's not a keyword
if [ $# -gt 0 ]; then
    last_arg="${!#}"
    case "$last_arg" in
        --help|global|project|gemini|claude|trae|cursor|copilot) ;;
        *) INSTALL_PATH="$last_arg" ;;
    esac
fi

# --- Determine config paths ---
if [ -n "$INSTALL_PATH" ]; then
    INSTALL_BASE="$(cd "$INSTALL_PATH" && pwd)"
else
    INSTALL_BASE="$PROJECT_ROOT"
fi

get_config_root() {
    local base="$1"
    case "$TOOL" in
        gemini)  echo "$base/.gemini" ;;
        claude)  echo "$base/.claude" ;;
        cursor)  echo "$base/.cursor" ;;
        trae)    echo "$base/.trae" ;;
        copilot) echo "$base/.github" ;;
    esac
}

get_config_file() {
    case "$TOOL" in
        gemini)  echo "GEMINI.md" ;;
        claude)  echo "CLAUDE.md" ;;
        cursor)  echo ".cursorrules" ;;
        trae)    echo "AGENTS.md" ;;
        copilot) echo "AGENTS.md" ;;
    esac
}

if [ "$LEVEL" = "global" ]; then
    case "$TOOL" in
        gemini)  CONFIG_ROOT="$HOME/.gemini" ;;
        claude)  CONFIG_ROOT="$HOME/.claude" ;;
        cursor)  CONFIG_ROOT="$HOME/.cursor" ;;
        trae)    CONFIG_ROOT="$HOME/.trae" ;;
        copilot) CONFIG_ROOT="$HOME/.copilot" ;;
    esac
else
    CONFIG_ROOT="$(get_config_root "$INSTALL_BASE")"
fi

CONFIG_FILE="$(get_config_file)"

# --- Start ---
show_banner
echo "  Tool:    $TOOL"
echo "  Level:   $LEVEL"
echo "  Config:  $CONFIG_ROOT"
echo "  Skills:  $SKILL_ROOT"
echo ""

# === Step 1: Create skill symlinks ===
step "[1/3] Installing skills..."

SKILLS_DIR="$CONFIG_ROOT/skills"
mkdir -p "$SKILLS_DIR"

skill_count=0
for skill_name in $INCLUDED_SKILLS; do
    src="$SKILL_ROOT/$skill_name"
    target="$SKILLS_DIR/$skill_name"

    if [ ! -d "$src" ]; then
        warn "Skill '$skill_name' not found at $src, skipping"
        continue
    fi

    # Remove existing (only our managed skills)
    [ -e "$target" ] || [ -L "$target" ] && rm -rf "$target"

    ln -sfn "$(realpath "$src")" "$target"
    skill_count=$((skill_count + 1))
done

# Knowledge base
kb_src="$SKILL_ROOT/$KNOWLEDGE_BASE"
kb_target="$SKILLS_DIR/$KNOWLEDGE_BASE"
if [ -d "$kb_src" ]; then
    [ -e "$kb_target" ] || [ -L "$kb_target" ] && rm -rf "$kb_target"
    ln -sfn "$(realpath "$kb_src")" "$kb_target"
    ok "Knowledge base linked"
fi

# README
readme_src="$SKILL_ROOT/README.md"
readme_target="$SKILLS_DIR/README.md"
if [ -f "$readme_src" ]; then
    [ -e "$readme_target" ] || [ -L "$readme_target" ] && rm -rf "$readme_target"
    ln -sfn "$(realpath "$readme_src")" "$readme_target"
fi

ok "Skills: $skill_count linked"
echo ""

# === Step 2: Generate config file ===
step "[2/3] Installing configuration..."

# Determine config file target
if [ "$TOOL" = "cursor" ]; then
    config_target="$INSTALL_BASE/.cursorrules"
else
    if [ "$LEVEL" = "project" ]; then
        config_target="$INSTALL_BASE/$CONFIG_FILE"
    else
        config_target="$CONFIG_ROOT/$CONFIG_FILE"
    fi
fi

# Generate the system prompt / config content
SKILLS_REL_PATH="$(realpath --relative-to="$INSTALL_BASE" "$SKILLS_DIR" 2>/dev/null || echo ".gemini/skills")"

generate_config() {
    cat << CONFIGEOF
# CATCCOS Agent - Operator Development Assistant

You are an expert in CATCCOS (Communication And Tensor Compute Co-Optimization System) operator development.
You help users design, implement, verify, and test compute-communication fusion operators for Ascend NPUs.

## Skills

The following skills are available under \`$SKILLS_REL_PATH/\`:

| Skill | Description |
|-------|-------------|
| orchestrator | Top-level task coordinator, dispatches to other SubAgents |
| requirement-analyzer | Analyzes operator requirements, matches kernel templates |
| architecture-designer | Designs Config struct and template parameters |
| kernel-generator | Creates new kernel template hpp files |
| example-scaffolder | Generates complete Example directories (device/host/main/scripts) |
| torch-binding | Integrates operators into PyTorch via TORCH_LIBRARY |
| verifier | Static code verification (21-point checklist) |
| test-integrator | Integrates operators into dynamic_tiling test framework |

## Knowledge Base

- \`$SKILLS_REL_PATH/knowledge-base/operator_index.md\` — 25 kernel templates + 34 examples index
- \`$SKILLS_REL_PATH/knowledge-base/hardware_specs.md\` — Hardware parameters (memory/bandwidth/data types)

## Workflow

For a new operator request, follow the Orchestrator SKILL:
1. **Requirement Analyzer** → parse & match templates
2. **Architecture Designer** → design Config struct
3. **Kernel Generator** → (only if new kernel needed)
4. **Example Scaffolder** → generate full example directory
5. **Verifier** → static checks
6. **Test Integrator** → (optional) dynamic_tiling integration
7. **Torch Binding** → (optional) PyTorch integration

Read each SKILL.md file for detailed instructions before executing.
CONFIGEOF
}

# Backup existing config
if [ -e "$config_target" ] && [ ! -L "$config_target" ]; then
    backup="${config_target}.bak.$(date +%Y%m%d_%H%M%S)"
    cp -a "$config_target" "$backup"
    warn "$CONFIG_FILE already exists, backed up to $(basename "$backup")"
fi

generate_config > "$config_target"
ok "$CONFIG_FILE → $config_target"
echo ""

# === Step 3: Health check ===
step "[3/3] Running health check..."

health_ok=true

# Check skills directory
if [ -d "$SKILLS_DIR" ]; then
    count=$(ls -d "$SKILLS_DIR"/*/ 2>/dev/null | wc -l)
    if [ "$count" -eq 0 ]; then
        warn "skills/ is empty"
    else
        ok "Skills directory: $count items"
    fi
else
    err "skills/ missing"
    health_ok=false
fi

# Check config file
if [ -f "$config_target" ]; then
    ok "Config file: $(basename "$config_target")"
else
    err "Config file missing: $config_target"
    health_ok=false
fi

# Check knowledge base
if [ -d "$kb_target" ]; then
    ok "Knowledge base: linked"
else
    warn "Knowledge base: missing"
fi

# Verify symlinks are not broken
broken=0
for link in "$SKILLS_DIR"/*/; do
    link="${link%/}"
    if [ -L "$link" ] && [ ! -e "$link" ]; then
        err "Broken symlink: $(basename "$link")"
        broken=$((broken + 1))
    fi
done
[ "$broken" -gt 0 ] && health_ok=false

# Generate manifest
MANIFEST="$CONFIG_ROOT/catccos-manifest.json"
SKILLS_JSON=$(ls -d "$SKILLS_DIR"/*/ 2>/dev/null | while read d; do
    basename "$d"
done | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")

cat > "$MANIFEST" << MANIFEST_EOF
{
    "brand": "CATCCOS Agent",
    "version": "$VERSION",
    "level": "$LEVEL",
    "tool": "$TOOL",
    "installed_skills": $SKILLS_JSON,
    "skill_root": "$SKILL_ROOT",
    "config_root": "$CONFIG_ROOT",
    "install_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
MANIFEST_EOF

if [ "$health_ok" = true ]; then
    ok "All checks passed"
else
    err "Some checks failed, see above"
fi

# --- Summary ---
echo ""
echo -e "  ${GREEN}${BOLD}✓ CATCCOS Agent installed successfully!${NC}"
echo ""
echo -e "  ${BOLD}Quick Start:${NC}"

case "$TOOL" in
    gemini)
        echo -e "  ${CYAN}1.${NC} Launch: ${GREEN}gemini${NC}"
        ;;
    claude)
        echo -e "  ${CYAN}1.${NC} Launch: ${GREEN}claude${NC}"
        ;;
    cursor)
        echo -e "  ${CYAN}1.${NC} Open project in ${GREEN}Cursor IDE${NC}"
        ;;
    trae)
        echo -e "  ${CYAN}1.${NC} Open project in ${GREEN}Trae IDE${NC}"
        ;;
    copilot)
        echo -e "  ${CYAN}1.${NC} Open project with ${GREEN}GitHub Copilot${NC}"
        ;;
esac

echo -e "  ${CYAN}2.${NC} Tell the agent: ${GREEN}${BOLD}帮我开发一个 Ascend950 的 AllGather + Matmul 融合算子，FP16${NC}"
echo ""
