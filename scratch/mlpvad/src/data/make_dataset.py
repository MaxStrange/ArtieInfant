"""
When run as a script, this will mount the given hard drive at data/raw/
and then download all the data into it and create a test split.
"""
import os
import subprocess
import sys

# TODO: More and especially, you still need to get voiced stuff
playlists_by_name = [
    "animals",
    "vehicles",,
    "construction",
    "orchestra",
    "whistling",
    "relaxing",
    "city",
    ]
playlists_by_url = [
    "https://www.youtube.com/playlist?list=PLUdgyJns5kk4CqxE2m_U0jmUhLVluyAVg",
    "https://www.youtube.com/playlist?list=PLNHQwiFqj9P85rlcaEsYe-cE-uK-n1VKa",
    "https://www.youtube.com/playlist?list=PLGUX2N7uSFwJsful0ONVBEcbISzD7knK-",
    "https://www.youtube.com/playlist?list=PLrYyvZOoKKiT_AmF0LLes4qDEEpNHxyes",
    "https://www.youtube.com/playlist?list=PLZV1ZbskgOJebWy0JQ27zy3FYQda75n-D",
    "https://www.youtube.com/playlist?list=PLpGECAvGnKuezB-u_yVFn0pRA2kUHpB5M",
    "https://www.youtube.com/playlist?list=PLY0sW_63wdrJmFieOFQHX9Bs5DKi1rkh-",
    ]

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: sudo python3", sys.argv[0], os.sep.join("path to hard drive".split(' ')),
                                                  os.sep.join("path to raw data folder".split(' ')))
        exit(1)

    dev_path = sys.argv[1]
    raw_data_path = sys.argv[2]

    # umount device first and ignore any errors
    subprocess.run(["sync"], stdout=subprocess.PIPE)
    umount_command = "umount " + dev_path
    subprocess.run(umount_command.split(' '), stdout=subprocess.PIPE)

    # mount the device here instead
    mount_command = "mount " + dev_path + " " + raw_data_path
    res = subprocess.run(mount_command.split(' '), stdout=subprocess.PIPE)
    res.check_returncode()

    # Download each playlist
    for pl_name, pl in zip(playlists_by_name, playlists_by_url):
        # Make a directory on the device for this playlist
        path = os.sep.join([raw_data_path, "mlpvad_raw_data", pl_name])
        mkdir_command = "mkdir -p " + path
        res = subprocess.run(mkdir_command.split(' '), stdout=subprocess.PIPE)
        res.check_returncode()

        # Download the playlist to that directory
        dl_command = ""# TODO : Only download a file if it is under one hour
        res = subprocess.run(dl_command.split(' '), stdout=subprocess.PIPE)
        res.check_returncode()

    # Get ~10% of each playlist and stick it in a test folder
    for pl_name in playlists_by_name:
        path = os.sep.join([raw_data_path, "mlpvad_raw_data", pl_name])
        files = os.listdir(path)

        # Make this playlist's test dir
        path_test = os.sep.join([raw_data_path, "mlpvad_raw_data", pl_name, "test_split"])
        mkdir_command = "mkdir -p " + path_test
        res = subprocess.run(mkdir_command.split(' '), stdout=subprocess.PIPE)
        res.check_returncode()

        files = files[::10]
        file_paths = [os.sep.join([path, f]) for f in files]
        # Move the files into the other dir
        for f in file_paths:
            mv_command = "mv " + f + " " + path_test + os.sep
            res = subprocess.run(mv_command.split(' '), stdout=subprocess.PIPE)
            res.check_returncode()
