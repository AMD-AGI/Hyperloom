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
#   1. refuses, naming the tool, when the image has no git or setpriv (nothing is installed here),
#   2. makes sure the account HYPERLOOM_SPECIALIST_USER names exists with uid/gid 2000, refusing when this container's
#      user namespace does not map uid or gid 2000 (one that maps only its root): no process can become that account
#      there (setuid fails with EINVAL, or under a fakeroot preload seems to succeed while the specialist stays root),
#      so the specialists could not be kept from the application's credentials,
#   3. puts the layer's claude CLI on PATH,
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

# Refuses (exit 1, naming the map) unless the user namespace maps the id $1 as a uid (in the uid map $2) and as a gid
# (in the gid map $3). A map that cannot be read, or has a line that is not three decimal ids, never counts as mapped.
require_mapped() {
  wanted="$1"
  shift
  for kind in uid gid; do
    rc=0
    in_id_map "$wanted" "$1" || rc=$?
    case "$rc" in
      0) ;;
      1) die "$kind $wanted is not mapped in this container's user namespace ($1: '${map_text:-empty}'): no process here can become the specialist account (setuid fails with EINVAL, or under a fakeroot preload seems to succeed while the specialist stays root and can read the application's credentials); run on a node whose user namespace maps uid and gid $wanted" ;;
      *) die "cannot read $1 as a user namespace map ($map_error), so whether the specialist account can be switched to is unknown" ;;
    esac
    shift
  done
}

for tool in git setpriv bash; do
  command -v "$tool" >/dev/null 2>&1 || die "the image has no $tool; use a framework image that has git, setpriv (util-linux) and bash"
done
[ "$(id -u)" = 0 ] || die "must run as root (uid 0) to create the specialist account"
[ -x "$layer/app_bootstrap.sh" ] || die "$layer has no app_bootstrap.sh"

# The account must be one a process here can become: its uid and gid mapped in this container's user namespace. The
# maps are kernel files; a fakeroot preload, which makes setuid seem to succeed, does not change what they say.
require_mapped "$uid" /proc/self/uid_map /proc/self/gid_map

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
