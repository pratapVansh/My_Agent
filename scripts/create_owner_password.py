"""
Generate the owner's bcrypt password hash and JWT signing secrets.

The plaintext password is never written anywhere — only the hash goes into
your .env, and only the hash is ever compared against at login.

Usage:
    python scripts/create_owner_password.py
    python scripts/create_owner_password.py --secrets-only
"""
import argparse
import getpass
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.auth.password import hash_password  # noqa: E402

MIN_PASSWORD_LENGTH = 12


def prompt_for_password() -> str:
    while True:
        password = getpass.getpass("Choose a password: ")
        if len(password) < MIN_PASSWORD_LENGTH:
            print(f"  Too short — use at least {MIN_PASSWORD_LENGTH} characters.\n")
            continue
        if len(password.encode("utf-8")) > 72:
            print("  Too long — bcrypt supports at most 72 bytes.\n")
            continue

        confirmation = getpass.getpass("Confirm password: ")
        if password != confirmation:
            print("  Passwords did not match.\n")
            continue
        return password


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", default="", help="Owner username (prompted if omitted)")
    parser.add_argument(
        "--secrets-only",
        action="store_true",
        help="Only generate JWT signing secrets, skip the password",
    )
    args = parser.parse_args()

    print("\n=== My_Agent authentication setup ===\n")

    lines: list[str] = []

    if not args.secrets_only:
        username = args.username or input("Owner username [vansh]: ").strip() or "vansh"
        password = prompt_for_password()
        password_hash = hash_password(password)
        # Drop the plaintext as soon as the hash exists.
        del password

        lines.append(f"OWNER_USERNAME={username}")
        lines.append(f"OWNER_PASSWORD_HASH={password_hash}")

    # 64 hex chars = 256 bits of entropy per secret. Distinct values matter:
    # sharing one secret would let an access token be replayed as a refresh token.
    lines.append(f"JWT_ACCESS_SECRET={secrets.token_hex(32)}")
    lines.append(f"JWT_REFRESH_SECRET={secrets.token_hex(32)}")

    print("\nAdd these lines to your .env file:\n")
    print("-" * 68)
    for line in lines:
        print(line)
    print("-" * 68)
    print(
        "\nKeep these secret. Rotating JWT_ACCESS_SECRET or JWT_REFRESH_SECRET\n"
        "immediately invalidates every existing session.\n"
    )


if __name__ == "__main__":
    main()
