"""
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
#    "arguments_VO",
#    "crazies_VO",
#    "letterman_VO",
#    "girlcode_VO",
#    "chinese_VO",
#    "arabic_VO",
#    "whispering_VO",
    "english",
    "chinese",
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
#    "https://www.youtube.com/playlist?list=PLMATWUx3t7L9AgEifZjTVT4iiOKxzfNO6",
#    "https://www.youtube.com/playlist?list=PLtbSrjOWdnG6LQUKjSMA00OUVv8n4-iV3",
#    "https://www.youtube.com/playlist?list=PLQErULyJFhO9tNT2gxOCk6Xm7LxGq5lI8",
#    "https://www.youtube.com/playlist?list=PLG-PeTcq2CV3ttMW6LcaWa3jfJo7NoztJ",
#    "https://www.youtube.com/playlist?list=PLfAyWdGHnLdFJnwmW_Yb4AJj5eGDlWIH4",
#    "https://www.youtube.com/playlist?list=PLWKPdTtKr85pjFqOkp6k2s_8KvZKAAXXH",
#    "https://www.youtube.com/playlist?list=PL0178B60ED927A8AD",
    "",
    "",
    ]

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: sudo python3", sys.argv[0], os.sep.join("path to hard drive".split(' ')),
                                                  os.sep.join("path to data folder".split(' ')))
        exit(1)

    dev_path = sys.argv[1]
    raw_data_path = os.path.join(sys.argv[2], "raw")
    processed_data_path = os.path.join(sys.argv[2], "processed")

    # Download each playlist
    for pl_name, pl_url in zip(playlists_by_name, playlists_by_url):
        print("Working on playlist:", pl_name)

        # Make a directory on the device for this playlist
        pl_raw_data_dir_path = os.sep.join([raw_data_path, pl_name])
#        print("  |-> Making directory:", pl_raw_data_dir_path)
#        mkdir_command = "mkdir -p " + pl_raw_data_dir_path
#        res = subprocess.run(mkdir_command.split(' '), stdout=subprocess.PIPE)
#        # Don't check result, in case this directory exists
#
#        # Download the playlist to that directory
#        print("  |-> Executing youtube-dl on the playlist...")
#        dl_command = "youtube-dl --extract-audio --audio-format wav --yes-playlist --ignore-errors --max-filesize 3G "\
#                     + pl_url + " -o " + pl_raw_data_dir_path + "/%(title)s-%(id)s.%(ext)s"
#        res = subprocess.run(dl_command.split(' '))
#        # Don't check result, who knows what youtube-dl returns
#
#        # Cut each file into 10 second pieces
#        pl_processed_data_dir_path = os.sep.join([processed_data_path, pl_name])
#        print("  |-> Making directory:", pl_processed_data_dir_path)
#        try:
#            os.mkdir(pl_processed_data_dir_path)
#        except FileExistsError:
#            pass
#        for dpath, __, fnames in os.walk(pl_raw_data_dir_path):
#            print("  |-> Walking path:", pl_raw_data_dir_path)
#            for fname in fnames:
#                try:
#                    print("    |-> Dicing and exporting segments for:", fname)
#                    raw_file_path = os.sep.join([dpath, fname])
#                    processed_file_path = os.sep.join([pl_processed_data_dir_path, fname.replace(' ', '_')])
#                    segment = audiosegment.from_file(raw_file_path)
#                    new_segments = segment.dice(seconds=10)
#                    for i, new in enumerate(new_segments):
#                        new = new.resample(sample_rate_Hz=32000, channels=1, sample_width=2)
#                        new_name, _ext = os.path.splitext(processed_file_path)
#                        new_name = new_name + "_seg" + str(i) + ".wav"
#                        new.export(new_name, format="wav")
#                    os.remove(raw_file_path)
#                    subprocess.run(["rm", "/tmp/*"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#                except OSError:
#                    try:
#                        for dpath, __, fnames in os.walk("/tmp"):
#                            for fname in fnames:
#                                os.remove(fname)
#                    except Exception:
#                        pass  # probably not enough ram to fit the whole thing into memory. just skip it.
#                except memoryError:
#                    pass
#                except Exception:
#                    print("There was an unknown error processing", raw_file_path)
#
#        # Now since the playlists are gigantic, remove the raw data
#        shutil.rmtree(pl_raw_data_dir_path)

    # Get ~10% of each playlist and stick it in a test folder
    print("|-> Making test split for each playlist...")
    for pl_name in playlists_by_name:
        pl_processed_data_dir_path = os.sep.join([processed_data_path, pl_name])
        print("  |-> Working on playlist:", pl_name, "in directory:", pl_processed_data_dir_path)
        files = os.listdir(pl_processed_data_dir_path)
        files = [f for f in files if not os.path.isdir(f)]

        # Make this playlist's test dir
        pl_processed_data_test_path = os.sep.join([pl_processed_data_dir_path, "test_split", pl_name])
        print("  |-> Making directory:", pl_processed_data_test_path)
        mkdir_command = "mkdir -p " + pl_processed_data_test_path
        res = subprocess.run(mkdir_command.split(' '), stdout=subprocess.PIPE)

        print("  |-> Collecting every tenth file in", pl_processed_data_dir_path)
        files = files[::10]
        file_paths = [os.sep.join([pl_processed_data_dir_path, f]) for f in files]
        # Move the files into the other dir
        print("  |-> Moving the files into the appropriate test directory...")
        for f in file_paths:
            fname = os.path.basename(f)
            new = os.sep.join([pl_processed_data_test_path, fname])
            os.rename(f, new)

