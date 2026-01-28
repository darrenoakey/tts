#!/bin/bash
# Memory-monitored TTS runner
# Monitors memory usage and kills process if it exceeds limit

MEMORY_LIMIT_GB=10
MEMORY_LIMIT_KB=$((MEMORY_LIMIT_GB * 1024 * 1024))
CHECK_INTERVAL=2

INPUT_FILE="${1:-test_large.txt}"
OUTPUT_FILE="${2:-test_large.mp3}"

cd "$(dirname "$0")"

echo "Starting TTS with memory monitoring (limit: ${MEMORY_LIMIT_GB}GB)"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo ""

# Start TTS in background
./run tts "$INPUT_FILE" -o "$OUTPUT_FILE" &
TTS_PID=$!

echo "TTS process started with PID: $TTS_PID"
echo "Monitoring memory usage..."
echo ""

# Function to get memory usage of process tree in KB
get_memory_kb() {
    local pid=$1
    # Get RSS of main process and all children (Python spawns subprocesses)
    ps -o rss= -p "$pid" 2>/dev/null | awk '{sum += $1} END {print sum+0}'
}

# Function to format KB as human readable
format_memory() {
    local kb=$1
    if [ "$kb" -ge 1048576 ]; then
        echo "$(echo "scale=2; $kb / 1048576" | bc)GB"
    elif [ "$kb" -ge 1024 ]; then
        echo "$(echo "scale=2; $kb / 1024" | bc)MB"
    else
        echo "${kb}KB"
    fi
}

MAX_MEMORY_KB=0
START_TIME=$(date +%s)

while kill -0 "$TTS_PID" 2>/dev/null; do
    # Get memory of main process
    MAIN_MEM=$(get_memory_kb "$TTS_PID")

    # Also check for python processes that might be children
    PYTHON_MEM=$(pgrep -P "$TTS_PID" 2>/dev/null | xargs -I{} ps -o rss= -p {} 2>/dev/null | awk '{sum += $1} END {print sum+0}')

    TOTAL_MEM=$((MAIN_MEM + PYTHON_MEM))

    if [ "$TOTAL_MEM" -gt "$MAX_MEMORY_KB" ]; then
        MAX_MEMORY_KB=$TOTAL_MEM
    fi

    ELAPSED=$(($(date +%s) - START_TIME))

    echo "[${ELAPSED}s] Memory: $(format_memory $TOTAL_MEM) (max: $(format_memory $MAX_MEMORY_KB))"

    if [ "$TOTAL_MEM" -gt "$MEMORY_LIMIT_KB" ]; then
        echo ""
        echo "MEMORY LIMIT EXCEEDED! Current: $(format_memory $TOTAL_MEM) > Limit: ${MEMORY_LIMIT_GB}GB"
        echo "Killing TTS process..."
        kill -9 "$TTS_PID" 2>/dev/null
        # Kill any child processes too
        pkill -9 -P "$TTS_PID" 2>/dev/null
        echo "Process killed."
        exit 1
    fi

    sleep "$CHECK_INTERVAL"
done

# Wait for process to finish and get exit code
wait "$TTS_PID"
EXIT_CODE=$?

ELAPSED=$(($(date +%s) - START_TIME))

echo ""
echo "=========================================="
echo "TTS completed with exit code: $EXIT_CODE"
echo "Total time: ${ELAPSED}s"
echo "Peak memory: $(format_memory $MAX_MEMORY_KB)"
echo "=========================================="

if [ -f "$OUTPUT_FILE" ]; then
    echo "Output file created: $OUTPUT_FILE ($(ls -lh "$OUTPUT_FILE" | awk '{print $5}'))"
fi

exit $EXIT_CODE
