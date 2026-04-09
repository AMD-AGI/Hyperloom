#!/bin/bash
# PRISM Optimization Dashboards — local HTTP server
#
# Usage:
#   bash /shared_nfs/nehaprakriya/PRISM/dashboards/serve.sh           # start
#   bash /shared_nfs/nehaprakriya/PRISM/dashboards/serve.sh stop      # stop
#   bash /shared_nfs/nehaprakriya/PRISM/dashboards/serve.sh status    # check
#   bash /shared_nfs/nehaprakriya/PRISM/dashboards/serve.sh restart   # restart

PORT="${PORT:-8765}"
DIR="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$DIR/.server.pid"

case "${1:-start}" in
  stop)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      kill "$(cat "$PIDFILE")"
      rm -f "$PIDFILE"
      echo "Stopped."
    else
      echo "Not running."
    fi
    ;;
  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "Running (PID $(cat "$PIDFILE")) → http://localhost:$PORT/"
    else
      echo "Not running. Launch with: bash $DIR/serve.sh"
    fi
    ;;
  restart)
    "$0" stop
    sleep 1
    "$0" start
    ;;
  start|"")
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "Already running (PID $(cat "$PIDFILE"))."
    else
      cd "$DIR"
      nohup python3 -m http.server "$PORT" > /dev/null 2>&1 &
      echo $! > "$PIDFILE"
      echo "Started (PID $!)."
    fi
    echo ""
    echo "=== PRISM Optimization Dashboards ==="
    echo ""
    echo "  Combined (all models + trees):  http://localhost:$PORT/prism-optimization-dashboard.html"
    echo ""
    echo "  Individual:"
    echo "    GLM-5 Timeline:               http://localhost:$PORT/glm5-optimization-timeline.html"
    echo "    Qwen3.5 Timeline:             http://localhost:$PORT/qwen35-optimization-timeline.html"
    echo "    DFS Search Trees:             http://localhost:$PORT/optimization-search-tree.html"
    echo ""
    echo "  Docs:"
    echo "    DFS Scoring Walkthrough:      $DIR/PRISM_DFS_SCORING_WALKTHROUGH.md"
    echo ""
    ;;
esac
