#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Install the AMD Corporate Root CA into the system trust store, so that every
# TLS client on the machine can verify the KB and Pulse endpoints without a
# per-process --ca-bundle.
#
# The certificate is deliberately NOT in this repository; its fingerprint is.
# --bootstrap downloads the root from AMD's PKI and accepts it only if it
# matches that pin. Otherwise pass the path to a copy you obtained yourself.
# Either way the pin decides, and this script puts the file where each distro
# family expects it. No host or URL is committed; they come from the environment:
#
#   KB_STORE_URL           the service to probe (or KBMINE_TLS_PROBE_HOST=host)
#   AMD_ROOT_CA_URL        --bootstrap only: the root's caIssuers URL(s), space-separated
#
#   ./scripts/provision-amd-ca.sh --check
#   ./scripts/provision-amd-ca.sh --bootstrap
#   ./scripts/provision-amd-ca.sh /path/to/amd_root.pem
#   ./scripts/provision-amd-ca.sh --from-bundle /path/to/amd_bundle.pem

set -euo pipefail

# Pinned identity of the root. A fingerprint is a hash, so it is safe to publish
# and it is the one property an attacker supplying a substitute root cannot
# reproduce. Verify it out of band once; after that this pin does the checking.
readonly WANT_SHA256="74:7D:15:A5:63:89:04:B0:98:EE:1C:8D:04:30:AB:37:E7:A9:F2:D2:D7:A4:C2:C9:CB:CD:7E:89:6E:05:32:C4"
readonly WANT_SUBJECT="CN = AMD Corporate Root CA"

# The host both services live behind, used to prove the install actually worked.
# Taken from KBMINE_TLS_PROBE_HOST, else from the host part of KB_STORE_URL.
probe_host() {
  local host=${KBMINE_TLS_PROBE_HOST:-}
  if [ -z "$host" ] && [ -n "${KB_STORE_URL:-}" ]; then
    host=${KB_STORE_URL#*://}; host=${host%%/*}; host=${host%%:*}
  fi
  [ -n "$host" ] || die "set KB_STORE_URL (or KBMINE_TLS_PROBE_HOST) to the service host to probe"
  printf '%s' "$host"
}

# AMD's PKI publishes the root at the caIssuers URL in the issuing CA's
# Authority Information Access extension; read it with
#   openssl x509 -in issuing_ca.pem -noout -ext authorityInfoAccess
# and export it as AMD_ROOT_CA_URL. Plain HTTP is the norm for this and is not a
# weakness here: the transport is unauthenticated, but WANT_SHA256 above arrived
# through a verified channel (this repository over GitHub TLS), so a substituted
# certificate cannot match the pin. Fetching without checking the pin is the
# unsafe act, not the HTTP itself.

die() { printf 'error: %s\n' "$*" >&2; exit 1; }
note() { printf '  %s\n' "$*"; }

need() { command -v "$1" >/dev/null || die "$1 is required but not installed"; }

# Anchor directory and refresh command differ by distro family. Debian
# additionally requires a .crt extension: update-ca-certificates silently
# ignores a file named .pem, which looks like success and does nothing.
detect_trust_dir() {
  if command -v update-ca-certificates >/dev/null \
     && [ -d /usr/local/share/ca-certificates ]; then
    TRUST_DIR=/usr/local/share/ca-certificates
    TRUST_NAME=AMD_Corporate_Root_CA.crt
    TRUST_CMD=update-ca-certificates
    SYSTEM_BUNDLE=/etc/ssl/certs/ca-certificates.crt
  elif command -v update-ca-trust >/dev/null \
       && [ -d /etc/pki/ca-trust/source/anchors ]; then
    TRUST_DIR=/etc/pki/ca-trust/source/anchors
    TRUST_NAME=AMD_Corporate_Root_CA.crt
    TRUST_CMD=update-ca-trust
    SYSTEM_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
  else
    die "unrecognised trust tooling; expected update-ca-certificates (Debian/Ubuntu) or update-ca-trust (RHEL/Fedora)"
  fi
}

fingerprint() {
  openssl x509 -in "$1" -noout -fingerprint -sha256 2>/dev/null | cut -d= -f2
}

# Pull the AMD root out of a concatenated bundle, so an existing amd_bundle.pem
# can seed the install without hunting for the standalone file.
extract_from_bundle() {
  local bundle=$1 out=$2 dir n f
  dir=$(mktemp -d)
  awk -v d="$dir" 'BEGIN{n=0} /BEGIN CERT/{n++} {print > (d "/c" n ".pem")}' "$bundle"
  for f in "$dir"/c*.pem; do
    if [ "$(fingerprint "$f")" = "$WANT_SHA256" ]; then
      cp "$f" "$out"; rm -rf "$dir"; return 0
    fi
  done
  rm -rf "$dir"
  die "no certificate in $bundle matches the pinned fingerprint"
}

# Download the root from AMD's PKI. Deliberately does not install anything: the
# caller runs it through verify_cert first, so a wrong or tampered download is
# rejected before it can reach the trust store.
bootstrap_fetch() {
  local out=$1 url
  need curl
  [ -n "${AMD_ROOT_CA_URL:-}" ] || die "set AMD_ROOT_CA_URL to the root's caIssuers URL (see the comment above bootstrap_fetch)"
  for url in ${AMD_ROOT_CA_URL}; do
    if curl -fsS --max-time 30 "$url" -o "$out" 2>/dev/null && [ -s "$out" ]; then
      note "fetched the root from ${url%%/CertEnroll*}"
      return 0
    fi
    note "unreachable: ${url%%/CertEnroll*}"
  done
  die "could not reach AMD PKI; obtain the root another way and pass its path"
}

verify_cert() {
  local cert=$1 got subject issuer
  [ -s "$cert" ] || die "$cert is missing or empty"
  openssl x509 -in "$cert" -noout >/dev/null 2>&1 \
    || die "$cert is not a certificate openssl can parse"

  got=$(fingerprint "$cert")
  [ "$got" = "$WANT_SHA256" ] || {
    printf 'error: fingerprint mismatch, refusing to install\n' >&2
    printf '  expected %s\n  got      %s\n' "$WANT_SHA256" "$got" >&2
    exit 1
  }

  # A root is self-signed. If subject and issuer differ we were handed an
  # intermediate, which will not complete the chain on its own.
  subject=$(openssl x509 -in "$cert" -noout -subject | sed 's/^subject=//')
  issuer=$(openssl x509 -in "$cert" -noout -issuer | sed 's/^issuer=//')
  [ "$subject" = "$issuer" ] || die "not a self-signed root (subject=$subject issuer=$issuer)"
  [ "$subject" = "$WANT_SUBJECT" ] || die "unexpected subject: $subject"

  openssl x509 -in "$cert" -noout -checkend 0 >/dev/null \
    || die "certificate has expired"
  openssl x509 -in "$cert" -noout -checkend 7776000 >/dev/null \
    || note "WARNING: root expires within 90 days; a fleet-wide renewal is due"
}

# Success means the default trust store alone verifies the endpoint: no
# --ca-bundle, no SSL_CERT_FILE. Run with those variables stripped so an
# ambient setting cannot make a failed install look fine.
probe_default_trust() {
  env -u SSL_CERT_FILE -u SSL_CERT_DIR -u REQUESTS_CA_BUNDLE -u CURL_CA_BUNDLE \
    openssl s_client -connect "$PROBE_HOST:443" -servername "$PROBE_HOST" \
    </dev/null 2>/dev/null | grep -q 'Verify return code: 0 (ok)'
}

main() {
  need openssl
  case "${1:-}" in
    "" | -h | --help) sed -n '5,21p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  esac
  PROBE_HOST=$(probe_host)

  if [ "${1:-}" = "--check" ]; then
    if probe_default_trust; then
      note "$PROBE_HOST verifies against the default trust store; --ca-bundle is not needed"
      exit 0
    fi
    note "$PROBE_HOST does NOT verify against the default trust store"
    note "fix it with: $0 --bootstrap"
    exit 1
  fi

  local cert
  case "${1:-}" in
    --bootstrap)
      if probe_default_trust; then
        note "already trusted; nothing to do"
        exit 0
      fi
      cert=$(mktemp); bootstrap_fetch "$cert" ;;
    --from-bundle)
      [ -n "${2:-}" ] || die "--from-bundle needs a path"
      cert=$(mktemp); extract_from_bundle "$2" "$cert"
      note "extracted the pinned root from $2" ;;
    *)
      cert=$1 ;;
  esac

  verify_cert "$cert"
  note "fingerprint matches the pin; subject is $WANT_SUBJECT"

  # Re-encode to PEM unconditionally. openssl reads DER happily, so a DER input
  # passes every check above and then lands in the anchor directory, where
  # update-ca-certificates skips it without an error: the install reports
  # success and does nothing. Normalising here removes that trap.
  local pem; pem=$(mktemp)
  openssl x509 -in "$cert" -outform pem -out "$pem"
  grep -q 'BEGIN CERTIFICATE' "$pem" || die "failed to normalise $cert to PEM"
  cert=$pem

  detect_trust_dir
  if [ "$(id -u)" -ne 0 ] && ! command -v sudo >/dev/null; then
    die "need root to write $TRUST_DIR, and sudo is unavailable"
  fi
  local sudo=""; [ "$(id -u)" -ne 0 ] && sudo=sudo

  $sudo install -m 644 "$cert" "$TRUST_DIR/$TRUST_NAME"
  note "installed $TRUST_DIR/$TRUST_NAME"
  # Debian's rehash step always complains about the multi-cert bundle it just
  # generated. Drop that one line; keep every other diagnostic.
  $sudo "$TRUST_CMD" 2>&1 >/dev/null | grep -v 'skipping ca-certificates.crt' >&2 || true
  note "refreshed the trust store with $TRUST_CMD"

  if probe_default_trust; then
    note "verified: $PROBE_HOST now trusts with no flags and no environment"
  else
    die "install completed but $PROBE_HOST still fails to verify"
  fi

  check_certifi "$cert"
}

# certifi ships its own root list and ignores the system store, so pip and
# requests can stay broken after a successful system install. Debian patches
# certifi to point back at the system bundle, in which case there is nothing to
# report; a virtualenv's own cacert.pem is the case that actually bites.
check_certifi() {
  local cert=$1 ca py
  for py in python3 "${VIRTUAL_ENV:-}/bin/python"; do
    command -v "$py" >/dev/null 2>&1 || continue
    ca=$("$py" -m certifi 2>/dev/null) || continue
    [ -n "$ca" ] && [ -f "$ca" ] || continue
    [ "$ca" = "$SYSTEM_BUNDLE" ] && continue   # Debian-patched, already covered
    if ! bundle_has_root "$ca" "$cert"; then
      note "note: certifi at $ca lacks this root."
      note "      stdlib urllib (what this tool uses) is fine, but pip and"
      note "      requests need: export REQUESTS_CA_BUNDLE=$SYSTEM_BUNDLE"
    fi
  done
}

# Compare by fingerprint rather than by text: PEM line wrapping and trailing
# newlines vary between producers, so a substring match is unreliable.
bundle_has_root() {
  local bundle=$1 dir f found=1
  dir=$(mktemp -d)
  awk -v d="$dir" 'BEGIN{n=0} /BEGIN CERT/{n++} n>0{print > (d "/c" n ".pem")}' "$bundle"
  for f in "$dir"/c*.pem; do
    [ -f "$f" ] || continue
    [ "$(fingerprint "$f")" = "$WANT_SHA256" ] && { found=0; break; }
  done
  rm -rf "$dir"
  return $found
}

main "$@"
