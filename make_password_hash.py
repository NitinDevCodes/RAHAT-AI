"""Creates the password hash for the RAHAT AI authority login.

Run this in your project folder:   python make_password_hash.py
Then paste the two printed lines into your .env file. The password itself is never saved anywhere.
"""
import getpass
import hashlib
import os
import sys

ITERATIONS = 200_000


def main():
    password = getpass.getpass("Choose a password (at least 10 characters): ")
    if len(password) < 10:
        print("That password is too short. Please use at least 10 characters.")
        sys.exit(1)
    if password != getpass.getpass("Repeat the password: "):
        print("The passwords do not match.")
        sys.exit(1)
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    print("\nAdd these lines to your .env file (you can change the username):\n")
    print("AUTHORITY_USERNAME=control_room")
    print(f"AUTHORITY_PASSWORD_HASH=pbkdf2_sha256:{ITERATIONS}:{salt.hex()}:{digest.hex()}")


if __name__ == "__main__":
    main()