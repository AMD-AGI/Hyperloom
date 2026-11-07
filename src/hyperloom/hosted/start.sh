#!/bin/sh
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Start a hosted Hyperloom run on any framework image, from the layer directory this script sits in
# (built by scripts/hosted_app_image/build_layer.sh, shipped as the small image of Dockerfile.layer), instead
# of from an image that was built for Hyperloom. Run it as the application's command:
#
#   sh <layer dir>/start.sh
#
# It runs as the container's root and
#   1. refuses, naming the tool, when the image has no git or bash (nothing is installed here),
#   2. chooses how the specialists are kept from the application's credentials:
#      - as the account HYPERLOOM_SPECIALIST_USER names, which it makes sure exists with uid/gid 2000, where a process
#        here can become that account and stop it: this start runs as root, the image has setpriv, this container's
#        user namespace maps uid and gid 2000, and root here holds CAP_KILL, CAP_SETGID and CAP_SETUID;
#      - else under the layer's bubblewrap (bin/bwrap): it makes no account, unsets HYPERLOOM_SPECIALIST_USER and says
#        why in one line; the application's start-up probes that bubblewrap and ends the run with a final exit.json
#        where it cannot confine an agent,
#      and refuses when neither can: in a user namespace that maps only its root no process can become the account
#      (setuid fails with EINVAL, or under a fakeroot preload seems to succeed while the specialist stays root),
#   3. names its own directory in HYPERLOOM_HOST_LAYER (the application launches agents with that directory's
#      bin/bwrap, never one found on PATH) and puts the layer's claude CLI on PATH,
#   4. trusts the layer's extra CA certificate (if any) through a bundle written to ${TMPDIR:-/tmp}/hyperloom-host
#      (local to the container: not the layer, not the state dir, not /etc),
#   5. makes the container's root mount private once, so bwrap can bind it (a failure only logs a line),
#   6. execs the layer's app_bootstrap.sh, which does the rest.
# Every refusal here is a plain failure (exit 1) of an image that cannot host a run, not an exit.json.

set -eu

layer="$(cd "$(dirname "$0")" && pwd)"
user="${HYPERLOOM_SPECIALIST_USER:-hl-specialist}"
uid=2000

die() {
  echo "hosted layer: $1" >&2
  exit 1
}

# Whether $1 is a decimal number as the kernel writes one (no sign, no leading zero).
decimal() {
  case "$1" in
    "" | *[!0-9]* | 0?*) return 1 ;;
  esac
}

# Whether the id $1 is in the user namespace map $2 ("inside outside count" per line: the ids inside to
# inside+count-1): 0 when it is, 1 when it is not, 2 when the map cannot be read or a line is not three decimal ids.
# Sets map_text to the map in short, or map_error to why it cannot be read.
in_id_map() {
  map_text="" map_error="" map_found=1
  if [ ! -r "$2" ]; then
    map_error="it cannot be read"
    return 2
  fi
  while read -r inside outside count extra || [ -n "${inside:-}" ]; do
    if ! decimal "$inside" || ! decimal "$outside" || ! decimal "$count" || [ -n "$extra" ]; then
      map_error="the line '$inside${outside:+ $outside}${count:+ $count}${extra:+ $extra}' is not 'inside outside count'"
      return 2
    fi
    map_text="${map_text:+$map_text; }$inside $outside $count"
    if [ "$inside" -le "$1" ] && [ "$1" -lt $((inside + count)) ]; then
      map_found=0
    fi
    inside=""
  done <"$2"
  return "$map_found"
}

# Whether the user namespace maps the id $1 as a uid (in the uid map $2) and as a gid (in the gid map $3): 0 when it
# does, 1 when it does not, with no_account saying which is not. Refuses (exit 1, naming the map) when a map cannot be
# read or has a line that is not three decimal ids: whether the specialist account can be switched to is then unknown.
id_mapped() {
  wanted="$1"
  shift
  for kind in uid gid; do
    rc=0
    in_id_map "$wanted" "$1" || rc=$?
    case "$rc" in
      0) ;;
      1)
        no_account="$kind $wanted is not mapped in this container's user namespace ($1: '${map_text:-empty}'): no process here can become the specialist account (setuid fails with EINVAL, or under a fakeroot preload seems to succeed while the specialist stays root and can read the application's credentials)"
        return 1
        ;;
      *) die "cannot read $1 as a user namespace map ($map_error), so whether the specialist account can be switched to is unknown" ;;
    esac
    shift
  done
}

# Whether root here holds CAP_KILL, CAP_SETGID and CAP_SETUID, which start a specialist as another account and later
# stop it, as the capability bounding set in the process status file $1 says: root's effective set once it execs the
# application is that set. 0 when it does, 1 when one is missing, with no_account naming them. A file that cannot be
# read, or whose CapBnd is not hexadecimal, counts as holding them, as root has them unless they were dropped (the
# application checks its own again before it starts a specialist as an account).
launch_capabilities() {
  bounding=""
  if [ -r "$1" ]; then
    while read -r field value rest || [ -n "${field:-}" ]; do
      [ "$field" != "CapBnd:" ] || bounding="$value"
      field=""
    done <"$1"
  fi
  case "$bounding" in
    "" | ? | *[!0-9a-f]*) return 0 ;;
  esac
  # CAP_KILL, CAP_SETGID and CAP_SETUID are bits 5, 6 and 7: all in the last two hexadecimal digits.
  low=$((0x${bounding#"${bounding%??}"}))
  missing=""
  [ $((low & 0x20)) -ne 0 ] || missing="CAP_KILL"
  [ $((low & 0x40)) -ne 0 ] || missing="${missing:+$missing, }CAP_SETGID"
  [ $((low & 0x80)) -ne 0 ] || missing="${missing:+$missing, }CAP_SETUID"
  [ -n "$missing" ] || return 0
  no_account="root here lacks $missing ($1: CapBnd $bounding), so it cannot start a specialist as another account and later stop it"
  return 1
}

# Whether the specialists can run as the account: this start runs as root, the image has setpriv, the user namespace
# maps uid and gid $uid, and root here holds the capabilities that start and stop them. Sets no_account to why not.
account_possible() {
  no_account=""
  me="$(id -u)"
  if [ "$me" != 0 ]; then
    no_account="this start runs as uid $me, not root, so it cannot make the specialist account"
  elif ! command -v setpriv >/dev/null 2>&1; then
    no_account="the image has no setpriv (util-linux), which starts the specialists as their account"
  elif id_mapped "$uid" /proc/self/uid_map /proc/self/gid_map; then
    launch_capabilities /proc/self/status || true
  fi
  [ -z "$no_account" ]
}

for tool in git bash; do
  command -v "$tool" >/dev/null 2>&1 || die "the image has no $tool; use a framework image that has git and bash"
done
[ -x "$layer/app_bootstrap.sh" ] || die "$layer has no app_bootstrap.sh"

# The account only where a process here can become it: its uid and gid mapped in this container's user namespace (the
# maps are kernel files; a fakeroot preload, which makes setuid seem to succeed, does not change what they say) and
# the capabilities to switch to it held. Else the layer's bubblewrap confines the specialists, as the app's own uid.
bwrap="$layer/bin/bwrap"
if account_possible; then
  # The account: its own uid and primary group, no supplementary group, a home it can write.
  by_uid="$(getent passwd "$uid" | cut -d: -f1 || true)"
  if [ -n "$by_uid" ] && [ "$by_uid" != "$user" ]; then
    die "uid $uid is taken by the account $by_uid, not $user"
  fi
  if [ -z "$by_uid" ]; then
    getent passwd "$user" >/dev/null 2>&1 && die "the account $user exists with a uid other than $uid"
    if ! getent group "$uid" >/dev/null 2>&1; then
      getent group "$user" >/dev/null 2>&1 && die "the group $user exists with a gid other than $uid"
      groupadd --gid "$uid" "$user"
    fi
    useradd --create-home --uid "$uid" --gid "$uid" --shell /bin/bash "$user"
  fi
elif [ -x "$bwrap" ]; then
  # The application would otherwise try the account it names (the manifest may set it) before bubblewrap.
  unset HYPERLOOM_SPECIALIST_USER
  echo "hosted layer: $no_account; no account is made, and the specialists run under the layer's bubblewrap ($bwrap)" >&2
else
  die "$no_account; and the layer has no bubblewrap ($bwrap) to confine them instead, so the specialists could not be kept from the application's credentials: start from a layer that has bin/bwrap (the layer image's does), or where a process can become uid and gid $uid"
fi

# The one bubblewrap the application launches agents with is this directory's: the image may have its own on PATH.
HYPERLOOM_HOST_LAYER="$layer"
export HYPERLOOM_HOST_LAYER

# The CLI the specialists spawn from PATH; a copy of it for every account, not a path in a home.
PATH="$layer/bin:$PATH"
export PATH
command -v claude >/dev/null 2>&1 || die "the layer has no claude CLI ($layer/bin/claude)"

# Trust the layer's CA on top of the image's own. A bundle of its own, local to this container (the layer
# directory may be read-only and the state dir is shared storage, possibly private to root), readable by every account.
ca="$layer/ca.pem"
if [ -f "$ca" ]; then
  system=""
  for f in /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/cert.pem; do
    if [ -f "$f" ]; then
      system="$f"
      break
    fi
  done
  dir="${TMPDIR:-/tmp}/hyperloom-host"
  bundle="$dir/ca-bundle.pem"
  mkdir -p "$dir"
  chmod 0755 "$dir"
  {
    [ -z "$system" ] || cat "$system"
    cat "$ca"
  } >"$bundle.tmp"
  chmod 0644 "$bundle.tmp"
  mv -f "$bundle.tmp" "$bundle"
  SSL_CERT_FILE="$bundle"
  REQUESTS_CA_BUNDLE="$bundle"
  NODE_EXTRA_CA_CERTS="$bundle"
  GIT_SSL_CAINFO="$bundle"
  export SSL_CERT_FILE REQUESTS_CA_BUNDLE NODE_EXTRA_CA_CERTS GIT_SSL_CAINFO
fi

# apptainer marks the root mount unbindable, and bwrap's bind of / then fails with EINVAL: make it private once. Not
# fatal: without CAP_SYS_ADMIN (plain docker) the caller wraps each bwrap call in `unshare --mount` instead.
if ! command -v mount >/dev/null 2>&1; then
  echo "hosted layer: the image has no mount (util-linux), so the root mount stays as it is; wrap each bwrap call in 'unshare --mount'" >&2
elif ! why="$(mount --make-rprivate / 2>&1)"; then
  echo "hosted layer: mount --make-rprivate / failed (${why:-no message}), so the root mount stays as it is; wrap each bwrap call in 'unshare --mount'" >&2
fi

exec "$layer/app_bootstrap.sh"
