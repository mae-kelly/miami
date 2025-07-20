#!/bin/bash

# DeFi Momentum Trading System - Repository Cleanup & Compression
# Removes unnecessary files and compresses the production-ready system

echo "🧹 DeFi Trading System - Repository Cleanup & Compression"
echo "=========================================================="
echo "Removing unnecessary files and compressing production system..."
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get current directory name for archive
REPO_NAME=$(basename "$PWD")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE_NAME="${REPO_NAME}_production_${TIMESTAMP}.tar.gz"

# Function to log actions
log_action() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Calculate initial size
INITIAL_SIZE=$(du -sh . 2>/dev/null | cut -f1)
log_info "Initial repository size: $INITIAL_SIZE"

echo ""
echo "🗑️  REMOVING UNNECESSARY FILES"
echo "─────────────────────────────────"

# Remove common unnecessary files and directories
CLEANUP_ITEMS=(
    # Version control artifacts
    ".git"
    ".gitignore"
    ".gitattributes"
    
    # IDE and editor files
    ".vscode"
    ".idea"
    "*.swp"
    "*.swo"
    "*~"
    ".DS_Store"
    "Thumbs.db"
    
    # Python cache and temporary files
    "__pycache__"
    "*.pyc"
    "*.pyo"
    "*.pyd"
    ".Python"
    "*.so"
    ".pytest_cache"
    ".coverage"
    ".tox"
    "*.egg-info"
    "dist/"
    "build/"
    
    # Node.js (if any)
    "node_modules"
    "package-lock.json"
    "yarn.lock"
    
    # Logs and temporary files
    "*.log"
    "*.tmp"
    "*.temp"
    "*.bak"
    "*.backup"
    "logs/"
    "tmp/"
    "temp/"
    
    # Documentation that might be auto-generated
    "docs/_build"
    "site/"
    
    # OS specific files
    ".AppleDouble"
    ".LSOverride"
    "Icon"
    "._*"
    
    # Large media files that might be examples
    "*.mp4"
    "*.avi"
    "*.mov"
    "*.wmv"
    "*.flv"
    "*.webm"
    
    # Large data files (keep essential ones)
    "*.dump"
    "*.sql"
    "*.csv"
    "*.xlsx"
    "*.xls"
    
    # Jupyter notebook checkpoints
    ".ipynb_checkpoints"
    
    # Virtual environments
    "venv/"
    "env/"
    ".env/"
    "virtualenv/"
    
    # Docker files if not needed
    "Dockerfile"
    "docker-compose.yml"
    ".dockerignore"
)

# Remove items
removed_count=0
for item in "${CLEANUP_ITEMS[@]}"; do
    if [[ "$item" == *"*"* ]]; then
        # Handle wildcards
        if ls $item 1> /dev/null 2>&1; then
            rm -rf $item 2>/dev/null && log_action "Removed: $item" && ((removed_count++))
        fi
    else
        # Handle direct files/directories
        if [ -e "$item" ] || [ -d "$item" ]; then
            rm -rf "$item" 2>/dev/null && log_action "Removed: $item" && ((removed_count++))
        fi
    fi
done

echo ""
echo "📁 PRESERVING ESSENTIAL FILES"
echo "──────────────────────────────"

# List of essential files to keep (production system)
ESSENTIAL_FILES=(
    # Core Python modules
    "scanner_v3.py"
    "executor_v3.py"
    "model_inference.py"
    "model_trainer.py"
    "optimizer.py"
    "honeypot_detector.py"
    "anti_rug_analyzer.py"
    "mempool_watcher.py"
    "token_profiler.py"
    "feedback_loop.py"
    "inference_server.py"
    "token_graph.py"
    "train_initial_model.py"
    "production_model_trainer.py"
    
    # Configuration and orchestration
    "run_pipeline.ipynb"
    "settings.yaml"
    "requirements.txt"
    ".env.example"
    
    # Scripts
    "quick.sh"
    
    # Data directory (if exists)
    "data/"
    
    # Models directory (if exists)
    "models/"
    
    # Documentation
    "README.md"
    "LICENSE"
)

# Check which essential files exist
existing_files=0
missing_files=0

for file in "${ESSENTIAL_FILES[@]}"; do
    if [ -e "$file" ]; then
        log_action "Preserved: $file"
        ((existing_files++))
    else
        log_warning "Missing: $file"
        ((missing_files++))
    fi
done

echo ""
echo "🗜️  COMPRESSING REPOSITORY"
echo "──────────────────────────"

# Create compressed archive
log_info "Creating compressed archive: $ARCHIVE_NAME"

# First, let's check what we're actually compressing
echo ""
log_info "Files to be compressed:"
find . -type f -name "*.py" -o -name "*.ipynb" -o -name "*.yaml" -o -name "*.yml" -o -name "*.txt" -o -name "*.md" -o -name "*.sh" -o -name "*.json" | head -20

# Create the archive
if tar -czf "../$ARCHIVE_NAME" \
    --exclude='.git*' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.log' \
    --exclude='.pytest_cache' \
    --exclude='node_modules' \
    --exclude='.DS_Store' \
    --exclude='Thumbs.db' \
    --exclude='.ipynb_checkpoints' \
    --exclude='venv' \
    --exclude='env' \
    --exclude='.vscode' \
    --exclude='.idea' \
    .; then
    
    ARCHIVE_SIZE=$(du -sh "../$ARCHIVE_NAME" 2>/dev/null | cut -f1)
    log_action "Archive created successfully: ../$ARCHIVE_NAME ($ARCHIVE_SIZE)"
else
    log_error "Failed to create archive"
    exit 1
fi

# Calculate final size
FINAL_SIZE=$(du -sh . 2>/dev/null | cut -f1)

echo ""
echo "📊 CLEANUP SUMMARY"
echo "──────────────────"
echo -e "${CYAN}Initial Size:${NC} $INITIAL_SIZE"
echo -e "${CYAN}Final Size:${NC} $FINAL_SIZE"
echo -e "${CYAN}Archive Size:${NC} $ARCHIVE_SIZE"
echo -e "${CYAN}Files Removed:${NC} $removed_count items"
echo -e "${CYAN}Essential Files:${NC} $existing_files preserved"
if [ $missing_files -gt 0 ]; then
    echo -e "${YELLOW}Missing Files:${NC} $missing_files (see warnings above)"
fi

echo ""
echo "📦 ARCHIVE CONTENTS"
echo "───────────────────"
echo "Archive location: ../$ARCHIVE_NAME"
echo ""
echo "Contents preview:"
tar -tzf "../$ARCHIVE_NAME" | head -15
if [ $(tar -tzf "../$ARCHIVE_NAME" | wc -l) -gt 15 ]; then
    echo "... and $(( $(tar -tzf "../$ARCHIVE_NAME" | wc -l) - 15 )) more files"
fi

echo ""
echo "🎯 PRODUCTION READINESS CHECK"
echo "──────────────────────────────"

# Check for critical production files
CRITICAL_FILES=("run_pipeline.ipynb" "scanner_v3.py" "executor_v3.py" "model_inference.py" "settings.yaml")
critical_missing=0

for file in "${CRITICAL_FILES[@]}"; do
    if [ ! -e "$file" ]; then
        log_error "CRITICAL: Missing $file"
        ((critical_missing++))
    fi
done

if [ $critical_missing -eq 0 ]; then
    log_action "All critical production files present"
else
    log_error "$critical_missing critical files missing - system may not function"
fi

# Check archive integrity
log_info "Verifying archive integrity..."
if tar -tzf "../$ARCHIVE_NAME" >/dev/null 2>&1; then
    log_action "Archive integrity verified"
else
    log_error "Archive integrity check failed"
fi

echo ""
echo "🚀 DEPLOYMENT INSTRUCTIONS"
echo "───────────────────────────"
echo "1. Extract archive: tar -xzf $ARCHIVE_NAME"
echo "2. Install dependencies: pip install -r requirements.txt"
echo "3. Configure environment: cp .env.example .env && edit .env"
echo "4. Run system health check: bash quick.sh"
echo "5. Launch trading system: jupyter notebook run_pipeline.ipynb"

echo ""
echo "⚠️  IMPORTANT SECURITY NOTES"
echo "─────────────────────────────"
echo "• Never commit real private keys to version control"
echo "• Use .env.example as template, create your own .env"
echo "• Test with small amounts before full deployment"
echo "• Monitor system performance and logs continuously"

echo ""
if [ $critical_missing -eq 0 ]; then
    echo -e "${GREEN}✅ CLEANUP COMPLETE - PRODUCTION SYSTEM READY${NC}"
    echo -e "${GREEN}📦 Archive: ../$ARCHIVE_NAME${NC}"
else
    echo -e "${RED}⚠️  CLEANUP COMPLETE - MISSING CRITICAL FILES${NC}"
    echo -e "${YELLOW}📦 Archive: ../$ARCHIVE_NAME (may be incomplete)${NC}"
fi

echo ""
echo "🎉 Repository cleanup and compression completed!"