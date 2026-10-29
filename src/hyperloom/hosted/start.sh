#!/bin/sh
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Start a hosted Hyperloom run on any framework image, from the layer directory this script sits in
# (built by scripts/hosted_app_image/build_layer.sh and published on a shared filesystem), instead
# of from an image that was built for Hyperloom. Run it as the application's command:
#
#   sh <layer dir>/start.sh
#
# It runs as the container's root and
#   1. refuses, naming the tool, when the image has no git or setpriv (nothing is installed here),
#   2. makes sure the account HYPERLOOM_SPECIALIST_USER names exists with uid/gid 2000,
#   3. puts the layer's claude CLI on PATH,
#   4. trusts the layer's extra CA certificate (if any) through a bundle written under the run's
#      state dir, without touching /etc,
#   5. execs the layer's app_bootstrap.sh, which does the rest.
# Every refusal here is a plain failure (exit 1) of an image that cannot host a run, not an exit.json.

set -eu

layer="$(cd "$(dirname "$0")" && pwd)"
user="${HYPERLOOM_SPECIALIST_USER:-hl-specialist}"
uid=2000

die() {
  echo "hosted layer: $1" >&2
  exit 1
}

for tool in git setpriv bash; do
  command -v "$tool" >/dev/null 2>&1 || die "the image has no $tool; use a framework image that has git, setpriv (util-linux) and bash"
done
[ "$(id -u)" = 0 ] || die "must run as root (uid 0) to create the specialist account"
[ -x "$layer/app_bootstrap.sh" ] || die "$layer has no app_bootstrap.sh"

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

# Trust the layer's CA on top of the image's own. A bundle of its own, in the state dir.
ca="$layer/ca.pem"
state="${HYPERLOOM_STATE_DIR:-}"
if [ -f "$ca" ] && [ -d "$state" ]; then
  system=""
  for f in /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/cert.pem; do
    [ -f "$f" ] && { system="$f"; break; }
  done
  bundle="$state/ca-bundle.pem"
  { [ -z "$system" ] || cat "$system"; cat "$ca"; } >"$bundle.tmp"
  mv -f "$bundle.tmp" "$bundle"
  chmod 0644 "$bundle"
  SSL_CERT_FILE="$bundle"
  REQUESTS_CA_BUNDLE="$bundle"
  NODE_EXTRA_CA_CERTS="$bundle"
  GIT_SSL_CAINFO="$bundle"
  export SSL_CERT_FILE REQUESTS_CA_BUNDLE NODE_EXTRA_CA_CERTS GIT_SSL_CAINFO
fi

exec "$layer/app_bootstrap.sh"
