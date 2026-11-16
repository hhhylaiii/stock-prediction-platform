#!/bin/sh
set -u
umask 077

repo=$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd -P)
backup_dir=${PG_BACKUP_DIR:-"$HOME/Backups/stock-prediction-platform"}
keep=${PG_BACKUP_KEEP-7}
dump_limit=${PG_BACKUP_TIMEOUT-5400}
log_file="$backup_dir/pg_backup.log"
started=$(date +%s)
partial=''

alert_code=$(cat <<'PYTHON'
import os

from apps.etl.task_alerts import report_failure


class PgBackupFailed(RuntimeError):
    """The host's nightly pg_dump of the database did not finish."""


report_failure("ops.pg_backup", PgBackupFailed(), os.environ.get("PG_BACKUP_CONTEXT", ""))
PYTHON
)

log() {
    line="$(date '+%Y-%m-%dT%H:%M:%S%z') $*"
    printf '%s\n' "$line"
    printf '%s\n' "$line" >> "$log_file"
}

elapsed() {
    echo $(($(date +%s) - started))
}

alert() {
    if docker compose exec -T -e "PG_BACKUP_CONTEXT=$1" worker \
        python manage.py shell -c "$alert_code" >> "$log_file" 2>&1; then
        log "alert handed to the worker; a webhook miss is logged above"
    else
        log "alert not sent: the worker did not run the report"
    fi
}

fail() {
    trap '' HUP INT TERM
    context="step=$1 status=$2 seconds=$(elapsed)"
    log "failed $context"
    alert "$context"
    exit 1
}

positive() {
    case $1 in
        '' | *[!0-9]*) return 1 ;;
    esac
    [ "$1" -gt 0 ]
}

cleanup() {
    if [ -n "$partial" ]; then
        rm -f -- "$partial"
    fi
}

prune() {
    ls -1 -- "$backup_dir" | grep -E '^pg-[0-9]{8}-[0-9]{6}\.dump$' | sort -r |
        awk -v keep="$keep" 'NR > keep' | while IFS= read -r old; do
            if rm -f -- "$backup_dir/$old"; then
                log "removed $old"
            else
                log "could not remove $old"
            fi
        done
}

cd -- "$repo" || fail repo 1
if ! mkdir -p -- "$backup_dir"; then
    log_file=/dev/null
    fail folder 1
fi
positive "$keep" || fail keep_count 1
positive "$dump_limit" || fail time_limit 1

name="pg-$(date +%Y%m%d-%H%M%S).dump"
partial="$backup_dir/.$name.partial"
trap cleanup EXIT
trap 'fail signal 1' HUP INT TERM
log "start $name keep=$keep time_limit=$dump_limit"

docker compose exec -T -e "PG_BACKUP_TIMEOUT=$dump_limit" db sh -c \
    'exec timeout "$PG_BACKUP_TIMEOUT" pg_dump --lock-wait-timeout=60s -Fc -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
    > "$partial" 2>> "$log_file"
status=$?
if [ "$status" -eq 124 ]; then
    fail dump_timeout 124
fi
if [ "$status" -ne 0 ]; then
    fail pg_dump "$status"
fi
bytes=$(wc -c < "$partial" | tr -d ' ')
if [ "$bytes" -eq 0 ]; then
    fail empty_dump 0
fi
if ! mv -- "$partial" "$backup_dir/$name"; then
    fail rename 1
fi
partial=''
trap - HUP INT TERM
prune
log "end ok file=$name bytes=$bytes seconds=$(elapsed)"
