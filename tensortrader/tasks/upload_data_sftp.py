import argparse
import os
import paramiko

def transfer_files(local_folder, target_folder, overwrite=False):
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

        # Use SFTP to transfer the files
        sftp = ssh.open_sftp()
        for filename in os.listdir(local_folder):
            local_path = os.path.join(local_folder, filename)
            remote_path = os.path.join(target_folder, filename)

            if os.path.isfile(local_path):
                if overwrite or not sftp.exists(remote_path):
                    print(f"Uploading File {filename} to {remote_path} ...")
                    sftp.put(local_path, remote_path)  # Upload file
                    print(f"File {filename} uploaded successfully to {remote_path}")
                else:
                    print(f"File {filename} already exists at {remote_path}. Skipping.")
        sftp.close()
    finally:
        ssh.close()

if __name__ == "__main__":
    
    SYMBOLS = [
        #"BTCUSDT",
        "ETHUSDT",
        "LTCUSDT",
        "ADAUSDT",
        "BNBUSDT",
        "BNBBTC",
        "EOSUSDT",
        "ETCUSDT",
        "TRXUSDT",
        "IOTAUSDT",
        "MKRUSDT",
        "DOGEUSDT",
    ]

    local_folder = "/mnt/c/Tensor/Database/Cryptos/"
    target_folder = "/home/optimlops/Documents/JOHN/mlops/datalake/cryptos/"
    
    for symbol in SYMBOLS:
        local_path = os.path.join(local_folder, symbol)
        target_path = os.path.join(target_folder, symbol)
        transfer_files(local_path, target_path, overwrite=True)
        
        
    # parser = argparse.ArgumentParser(description="Upload all files from a local folder to a target folder via SFTP.")
    # parser.add_argument("local_folder", type=str, help="The local folder path.")
    # parser.add_argument("target_folder", type=str, help="The target folder path on the server.")
    # parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files on the server.")
    # args = parser.parse_args()
    #transfer_files(args.local_folder, args.target_folder, args.overwrite)