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

playlists_by_name = [
#    "debug_list_NO",
#    "debug_list_VO",
#    "animals_NO",
#    "vehicles_NO",
#    "construction_NO",
#    "orchestra_NO",
#    "relaxing_NO",
#    "city_NO",
#   ------------
#    "singing_VO",
#    "dateline_VO",
#    "jeopardy_VO",
    "arguments_VO",
    "crazies_VO",
    "letterman_VO",
    "girlcode_VO",
    "chinese_VO",
    "arabic_VO",
    "whispering_VO",
    ]
playlists_by_url = [
#    "https://www.youtube.com/playlist?list=PLi9gcAlLQJoJg6gqp-MiT8Mm3v7OOFarv",
#    "https://www.youtube.com/playlist?list=PLAYFVhxsaqDuOh4Ic5mRu5CiZVKCMVv66",
#    "https://www.youtube.com/playlist?list=PLUdgyJns5kk4CqxE2m_U0jmUhLVluyAVg",
#    "https://www.youtube.com/playlist?list=PLNHQwiFqj9P85rlcaEsYe-cE-uK-n1VKa",
#    "https://www.youtube.com/playlist?list=PLGUX2N7uSFwJsful0ONVBEcbISzD7knK-",
#    "https://www.youtube.com/playlist?list=PLrYyvZOoKKiT_AmF0LLes4qDEEpNHxyes",
#    "https://www.youtube.com/playlist?list=PLpGECAvGnKuezB-u_yVFn0pRA2kUHpB5M",
#    "https://www.youtube.com/playlist?list=PLY0sW_63wdrJmFieOFQHX9Bs5DKi1rkh-",
#   ----------------------
#    "https://www.youtube.com/playlist?list=PL25JVhBBh1BUJpyguS0WrFR-xxl8rKMVC",
#    "https://www.youtube.com/playlist?list=PLA4fUiA62LMgsBhBA5H9eTJ_MxqYaJKsD",
#    "https://www.youtube.com/playlist?list=PL-C_fu0ZbbDLdc3ZW5-eqkr68_dKR2OWN",
    "https://www.youtube.com/playlist?list=PLMATWUx3t7L9AgEifZjTVT4iiOKxzfNO6",
    "https://www.youtube.com/playlist?list=PLtbSrjOWdnG6LQUKjSMA00OUVv8n4-iV3",
    "https://www.youtube.com/playlist?list=PLQErULyJFhO9tNT2gxOCk6Xm7LxGq5lI8",
    "https://www.youtube.com/playlist?list=PLG-PeTcq2CV3ttMW6LcaWa3jfJo7NoztJ",
    "https://www.youtube.com/playlist?list=PLfAyWdGHnLdFJnwmW_Yb4AJj5eGDlWIH4",
    "https://www.youtube.com/playlist?list=PLWKPdTtKr85pjFqOkp6k2s_8KvZKAAXXH",
    "https://www.youtube.com/playlist?list=PL0178B60ED927A8AD",
    ]

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
        os.mkdir(processed_path, exist_ok=True)
        for dpath, _, fnames in os.walk(path):
            for fname in fnames:
                raw_file_path = dpath + "/" + fname
                processed_file_path = processed_path + "/" + fname.replace(' ', '_')
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

