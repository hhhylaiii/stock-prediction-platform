#!/bin/sh
set -u

repo=$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd -P)
backup_dir=${PG_BACKUP_DIR:-"$HOME/Backups/stock-prediction-platform"}
keep=${PG_BACKUP_KEEP-7}
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
    context="step=$1 status=$2 seconds=$(elapsed)"
    log "failed $context"
    alert "$context"
    exit 1
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
case $keep in
    '' | *[!0-9]*) fail keep_count 1 ;;
esac
if [ "$keep" -lt 1 ]; then
    fail keep_count 1
fi

name="pg-$(date +%Y%m%d-%H%M%S).dump"
partial="$backup_dir/.$name.partial"
trap cleanup EXIT
trap 'fail signal 1' HUP INT TERM
log "start $name keep=$keep"

docker compose exec -T db sh -c 'exec pg_dump -Fc -U "$POSTGRES_USER" -d "$POSTGRES_DB"' \
    > "$partial" 2>> "$log_file"
status=$?
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
prune
log "end ok file=$name bytes=$bytes seconds=$(elapsed)"
