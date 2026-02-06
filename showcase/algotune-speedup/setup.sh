#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== AlgoTune Speedup Showcase Setup ==="

# 1. Clone AlgoTune if not present
if [ ! -d "algotune" ]; then
    echo "Cloning AlgoTune repository..."
    git clone --depth 1 https://github.com/oripress/AlgoTune.git algotune
else
    echo "AlgoTune already cloned."
fi

# 2. Create venv if not present
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3.12 -m venv .venv
else
    echo "Virtual environment already exists."
fi

source .venv/bin/activate

# 3. Install AlgoTune dependencies
echo "Installing AlgoTune dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r algotune/requirements.txt 2>/dev/null || \
    pip install --quiet -e "algotune[dev]" 2>/dev/null || \
    echo "Warning: Some AlgoTune deps may have failed. Check manually."

# 4. Install evolve-sdk and dependencies
echo "Installing evolve-sdk..."
pip install --quiet -e "../../sdk"
pip install --quiet claude-agent-sdk

# 5. Add AlgoTune to PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR/algotune:${PYTHONPATH:-}"

# 6. Verify installation
echo "Verifying AlgoTune installation..."
python -c "
import sys
sys.path.insert(0, 'algotune')
from AlgoTuneTasks import factory
print('AlgoTune TaskFactory: OK')
print(f'Python: {sys.version}')
" || {
    echo "ERROR: AlgoTune import failed. Check dependencies."
    exit 1
}

echo ""
echo "=== Setup Complete ==="
echo "Activate with: source .venv/bin/activate"
echo "Set PYTHONPATH: export PYTHONPATH=$SCRIPT_DIR/algotune:\$PYTHONPATH"
