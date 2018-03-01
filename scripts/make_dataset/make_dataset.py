"""
TODO: Give this script a list of playlists and have it make a directory of raw
and test data. Don't do any preprocessing on the data - you can
have the octopod system do that.

When run as a script, this will mount the given hard drive at data/raw/
and then download all the data into it and create a test split.
"""
import audiosegment
import os
import shutil
import subprocess
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: python3", sys.argv[0], "<path to conf file> <target path>")
        exit(1)

    _, config_path, target_path = sys.argv
    with open(config_path) as configfile:
        lines = [line.strip() for line in configfile if not line.strip().startswith('#')]
        names_and_urls = [map(lambda x: x.strip(), line.split(',')) for line in lines]

    for name, url in names_and_urls:
        print("Working on playlist:", name)

        path = target_path + "/" + name
        os.makedirs(path, exist_ok=True)

        # Download the playlist to that directory
        print("  |-> Executing youtube-dl on the playlist...")
        dl_command = "youtube-dl --extract-audio --audio-format wav --yes-playlist --ignore-errors --max-filesize 3G "\
                     + url + " -o " + path + "/%(title)s-%(id)s.%(ext)s"
        subprocess.run(dl_command.split(' '))
        # Don't check result, who knows what youtube-dl returns

        # Cut each file into 10 minute pieces
        processed_path = path + "/processed"
        os.makedirs(processed_path, exist_ok=True)
        for dpath, _, fnames in os.walk(path):
            for fname in fnames:
                raw_file_path = dpath + "/" + fname
                processed_fname = "".join([i if ord(i) < 128 else 'x' for i in text.replace(' ', '_')])
                processed_file_path = processed_path + "/" + processed_fname
                try:
                    segment = audiosegment.from_file(raw_file_path)
                    new_segments = segment.dice(seconds=10 * 60)
                    for i, new in enumerate(new_segments):
                        new = new.resample(sample_rate_Hz=48000, channels=1, sample_width=2)
                        new_name, _ext = os.path.splitext(processed_file_path)
                        new_name = new_name + "_seg" + str(i) + ".wav"
                        new.export(new_name, format="wav")
                except OSError:
                    pass  # Probably not enough RAM to fit the whole thing into memory. Just skip it.
                except MemoryError:
                    pass
                os.remove(raw_file_path)

    # Get ~10% of each playlist and stick it in a test folder
    print("|-> Making test split for each playlist...")
    for pl_name, _ in names_and_urls:
        pl_processed_data_dir_path = target_path + "/" + pl_name + "/processed"
        print("  |-> Working on playlist:", pl_name, "in directory:", pl_processed_data_dir_path)
        files = os.listdir(pl_processed_data_dir_path)
        files = [f for f in files if not os.path.isdir(f)]

        # Make this playlist's test dir
        pl_processed_data_test_path = pl_processed_data_dir_path + "/test_split/" + pl_name
        os.makedirs(pl_processed_data_test_path, exist_ok=True)

        print("  |-> Collecting every tenth file in", pl_processed_data_dir_path)
        files = files[::10]
        file_paths = [pl_processed_data_dir_path + "/" + f for f in files]

        # Move the files into the other dir
        print("  |-> Moving the files into the appropriate test directory...")
        for f in file_paths:
            fname = os.path.basename(f)
            new = pl_processed_data_test_path + "/" + fname
            os.rename(f, new)

