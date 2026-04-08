#!/bin/bash
# Quick setup to create a global 'semtalk' command
# Run this once: bash scripts/setup_global_command.sh

echo "🔧 Setting up global 'semtalk' command..."

ALIAS_LINE='alias semtalk="/home/ferpaa/SemTalk/scripts/agent_connect.sh"'

# Detect shell
if [ -n "$BASH_VERSION" ]; then
    SHELL_RC="$HOME/.bashrc"
elif [ -n "$ZSH_VERSION" ]; then
    SHELL_RC="$HOME/.zshrc"
else
    SHELL_RC="$HOME/.bashrc"
fi

# Check if alias already exists
if grep -q "alias semtalk=" "$SHELL_RC" 2>/dev/null; then
    echo "✅ Alias already exists in $SHELL_RC"
else
    echo "" >> "$SHELL_RC"
    echo "# SemTalk quick connect" >> "$SHELL_RC"
    echo "$ALIAS_LINE" >> "$SHELL_RC"
    echo "✅ Added alias to $SHELL_RC"
fi

echo ""
echo "📋 To activate now, run:"
echo "   source $SHELL_RC"
echo ""
echo "📋 After that, from anywhere you can run:"
echo "   semtalk                                  # Interactive shell"
echo "   semtalk \"nvidia-smi\"                    # Run command"
echo "   semtalk \"python train.py --config ...\"  # Run training"
