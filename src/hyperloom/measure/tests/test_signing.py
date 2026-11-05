# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest

from hyperloom.measure import signing

DATA = b'{"request_id": "r1"}'
# Assembled so the file holds no literal key header for secret scanners to flag.
_PEM_BEGIN = "-----BEGIN " + "PRIVATE KEY-----"
_PEM_END = "-----END " + "PRIVATE KEY-----"


@pytest.fixture
def key():
    return signing.generate_private_key()


def test_a_signature_round_trips(key) -> None:
    signature = signing.sign(DATA, key)
    assert signing.verify(DATA, signature, signing.public_key_pem(key)) is True


def test_one_flipped_data_byte_fails_verification(key) -> None:
    signature = signing.sign(DATA, key)
    for index in range(len(DATA)):
        changed = bytearray(DATA)
        changed[index] ^= 0x01
        assert signing.verify(bytes(changed), signature, signing.public_key_pem(key)) is False, index


def test_one_flipped_signature_byte_fails_verification(key) -> None:
    raw = bytearray(base64.b64decode(signing.sign(DATA, key)))
    raw[10] ^= 0x01
    assert signing.verify(DATA, base64.b64encode(bytes(raw)).decode(), signing.public_key_pem(key)) is False


def test_a_signature_made_with_the_callers_own_key_does_not_verify_against_the_sandbox_key(key) -> None:
    own = signing.generate_private_key()
    forged = signing.sign(DATA, own)
    assert signing.verify(DATA, forged, signing.public_key_pem(key)) is False
    assert signing.verify(DATA, forged, signing.public_key_pem(own)) is True  # the check is the key, not the format


@pytest.mark.parametrize("signature", ["", "not base64!!", "AAAA", base64.b64encode(b"x" * 65).decode()])
def test_a_malformed_signature_is_false_not_an_error(key, signature: str) -> None:
    assert signing.verify(DATA, signature, signing.public_key_pem(key)) is False


def test_an_unusable_public_key_is_an_error_not_a_false(key) -> None:
    with pytest.raises(signing.SigningError, match="public key"):
        signing.verify(DATA, signing.sign(DATA, key), "-----BEGIN PUBLIC KEY-----\nAAAA\n-----END PUBLIC KEY-----\n")


def test_a_public_key_of_another_algorithm_is_refused(key) -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    pem = (
        ec.generate_private_key(ec.SECP256R1())
        .public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    with pytest.raises(signing.SigningError, match="not an Ed25519"):
        signing.verify(DATA, signing.sign(DATA, key), pem)


def test_the_private_key_file_round_trips(tmp_path: Path, key) -> None:
    path = tmp_path / "key.pem"
    path.write_bytes(signing.private_key_pem(key))
    loaded = signing.load_private_key(path)
    assert signing.verify(DATA, signing.sign(DATA, loaded), signing.public_key_pem(key))


def test_loading_a_bad_private_key_names_no_key_material(tmp_path: Path) -> None:
    path = tmp_path / "key.pem"
    path.write_text(_PEM_BEGIN + "\nSECRETSECRETSECRET\n" + _PEM_END + "\n")
    with pytest.raises(signing.SigningError) as raised:
        signing.load_private_key(path)
    assert "SECRETSECRET" not in str(raised.value)
    with pytest.raises(signing.SigningError, match="cannot read"):
        signing.load_private_key(tmp_path / "missing.pem")


def test_the_fingerprint_is_the_sha256_of_the_der_public_key(key) -> None:
    import hashlib

    from cryptography.hazmat.primitives import serialization

    der = key.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    assert signing.public_key_fingerprint(signing.public_key_pem(key)) == "sha256:" + hashlib.sha256(der).hexdigest()


def test_a_missing_cryptography_package_says_how_to_install_it(monkeypatch) -> None:
    for name in [n for n in sys.modules if n == "cryptography" or n.startswith("cryptography.")]:
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(signing.SigningError, match=r"hyperloom\[hosted\]"):
        signing.generate_private_key()
