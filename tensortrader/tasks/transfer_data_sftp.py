import argparse
import os

import paramiko


def transfer_file(remote_path, local_path, action):
    # Server credentials
    hostname = os.getenv("SERVER_HOSTNAME")
    port = 22  # Default SSH port
    username = os.getenv("SERVER_USERNAME")
    password = os.getenv("SERVER_PASSWORD")

    # Create an SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(hostname, port, username, password)

        # Use SFTP to transfer the file
        sftp = ssh.open_sftp()
        if action == "download":
            sftp.get(remote_path, local_path)  # Download file
            print(f"File downloaded successfully to {local_path}")
        elif action == "upload":
            sftp.put(local_path, remote_path)  # Upload file
            print(f"File uploaded successfully to {remote_path}")
        else:
            print("Invalid action. Use 'download' or 'upload'.")
        sftp.close()
    finally:
        ssh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload or download a file via SFTP.")
    parser.add_argument("remote_path", type=str, help="The remote file path.")
    parser.add_argument("local_path", type=str, help="The local file path.")
    parser.add_argument(
        "action",
        type=str,
        choices=["upload", "download"],
        help="The action to perform: upload or download.",
    )

    args = parser.parse_args()
    transfer_file(args.remote_path, args.local_path, args.action)
