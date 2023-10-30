"Removes silences from multitrack video files"
from argparse import ArgumentParser
from subprocess import run, PIPE, Popen, STDOUT
from pathlib import Path
from pydub import AudioSegment, silence
from ffprobe import FFProbe
from logging import basicConfig, info, INFO
from tharospytools.list_tools import grouper
from tharospytools.console_utilities import progress_bar
from os import path
from json import dump, load
ERROR_FORMAT = (
    "%(asctime)s | %(levelname)s in %(funcName)s in %(filename)s "
    "at line %(lineno)d: %(message)s"
)
basicConfig(
    format=ERROR_FORMAT, datefmt="%d-%b-%y %H:%M:%S", level=INFO
)


def extract_audio(input_file: str, ignore_tracks: [list[int]]) -> list[str]:
    """Given an imput file, extracts all audio tracks if they aren't already been extracted

    Args:
        input_file (str): a video file path
        ignore_tracks ([list[int]]): a list of audio ID tracks to skip

    Returns:
        list[str]: list of paths to audio tracks
    """
    metadata = FFProbe(input_file)
    fluxes: list[str] = list()
    print("Extracting audio from container")
    if metadata.audio:
        for i in range(0, len(metadata.audio)):
            if i not in ignore_tracks:
                progress_bar(i+1, len(metadata.audio), bar_length=50)
                if not path.exists(target_audio_file := path.join(Path(input_file).parent, f"audio{i:02}.wav")):
                    cmd = (
                        f'ffmpeg -i "{input_file}" '
                        f"-map 0:a:{i} -acodec pcm_s16le "
                        f'-ar 16000 "{target_audio_file}" -y'
                        '-loglevel quiet'
                    )
                    p = Popen(
                        cmd,
                        shell=True,
                        stdout=PIPE,
                        stderr=STDOUT,
                        bufsize=1,
                        universal_newlines=True,
                    )
                    with p:
                        for out_line in p.stdout:
                            info(out_line)
                fluxes.append(target_audio_file)
    return fluxes


def concat(list_to_merge: list[str], output: str = 'output.mp4') -> None:
    """Given a series of video files, concat those in a single file

    Args:
        list_to_merge (list[str]): list of all files
        output (str, optional): output path. Defaults to 'output.mp4'.
    """
    pipeline_file: str = "inputs.txt"
    with open(pipeline_file, 'w', encoding='utf-8') as writer:
        for file in list_to_merge:
            writer.write(f"file\t{file}\n")
    run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', pipeline_file, "-map", "0", "-c:a", "copy", output])


def cut_video_file(input_file: str, output_file: str, start_timecode: float, end_timecode: float, avoid_freezes: bool = True, delta: float = 2.0, margins: float = 0) -> str:
    """Extracts section from video, based on timecodes, in seconds.

    Args:
        input_file (str): the video input file
        output_file (str): the name where to extract
        start_timecode (float): start
        end_timecode (float): end
        delta (float): default security to avoid glitches
        margins (float): room to breathe

    Raises:
        ValueError: if end is before start

    Retuns:
        str: path of the output file
    """
    if delta > start_timecode:
        delta = start_timecode
    if end_timecode > start_timecode:
        if avoid_freezes:
            run(["ffmpeg", '-y', "-ss", str(start_timecode-delta-margins), "-i", input_file, "-ss",
                str(delta-margins), "-t", str((end_timecode-start_timecode)+margins+delta), "-map", "0", "-c:a", "copy", output_file, '-loglevel', 'quiet'])
        else:
            run(["ffmpeg", '-y', "-ss", str(start_timecode-delta-margins), "-i", input_file, "-ss",
                str(delta-margins), "-t", str((end_timecode-start_timecode)+margins+delta), "-map", "0", "-c:a", "copy", output_file, '-loglevel', 'quiet'])
        return output_file
    raise ValueError("Specified timestamps are not valid.")


def filter_silences(list_of_silences: list[set]) -> list[tuple]:
    """Output bounds of shared silences in seconds.
    Seeks for the smallest subsets between sequences, seeking for minimum overlaps to keep all spoken parts.

    Args:
        list_of_silences (list[set]): a list of sets of tuples of silences in miliseconds

    Returns:
        list[tuple]: a list of tuples of silences in seconds
    """
    silences = sorted(list(set.intersection(*list_of_silences)))
    bound_low: int = silences[0]
    bound_high: int
    blanks: list[tuple[float, float]] = list()
    for i, timecode in enumerate(silences[1:]):
        if timecode != silences[i]+1:
            bound_high = silences[i]
            blanks.append((bound_low/1000, bound_high/1000))
            bound_low = timecode
    return blanks


def detect_silences(input_file: str, audio_to_skip: list[int]) -> list:
    """Extracts the silences in miliseconds from a audio file

    Args:
        input_file (str): the audio file

    Returns:
        list: list of tuples of silences positions
    """
    audio_tracks: list[str] = extract_audio(input_file, audio_to_skip)
    silences: list[set] = list()
    # Extracting audio chans to perform silence analysis
    print("Processing audio tracks analysis")
    for i, audio in enumerate(audio_tracks):
        progress_bar(i+1, len(audio_tracks), bar_length=50)
        myaudio = AudioSegment.from_wav(audio)
        blanks = silence.detect_silence(
            myaudio,
            min_silence_len=1000,
            silence_thresh=myaudio.dBFS-16
        )
        silences.append(
            {i for start, stop in blanks for i in range(start, stop)}
        )
    return filter_silences(silences)


def get_duration(input_file: str) -> float:
    """Returns the duration of the media

    Args:
        input_file (str): a path to a video file

    Returns:
        float: length in seconds
    """
    return float(run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_file], stdout=PIPE).stdout.decode(encoding='utf-8'))


def parts_to_keep(start_timecode: float, end_timecode: float, list_of_silences: list[tuple[float, float]], duration_threshold: float = 0.5) -> list[list[float, float]]:
    """Filters out unwanted parts and returns the timecodes to keep.

    Args:
        start_timecode (float): start timecode of the file
        end_timecode (float): end of the file
        list_of_silences (list[tuple[float,float]]): list of intersection of silences
        duration_threshold (float, optional): limit to consider a silence. Defaults to 0.5.

    Returns:
        list[tuple[float,float]]: a purged list of silences
    """
    remove_start: bool = False
    remove_end: bool = False
    endpoints: list = list()
    if start_timecode != list_of_silences[0][0]:
        endpoints.append(start_timecode)
    else:
        remove_start = True
    for start_silence, end_silence in list_of_silences:
        if end_silence-start_silence >= duration_threshold:
            # We remove this silence
            endpoints.extend([start_silence, end_silence])
    if end_timecode != list_of_silences[-1][-1]:
        endpoints.append(end_timecode)
    else:
        remove_end = True
    # Checking if we need to remove start or end
    if remove_start:
        endpoints = endpoints[1:]
    elif remove_end:
        endpoints = endpoints[:-1]
    return grouper(endpoints, n=2, m=0)


def extract_parts(input_file: str, endpoints: list[list[float, float]]) -> list[str]:
    """extracts parts given a cleansed list of intervals to extract

    Args:
        input_file (str): the video file path
        endpoints (list[list[float, float]]): the purged list in miliseconds

    Returns:
        list[str]: list of files names
    """
    base_name, extension = Path(input_file).stem, Path(input_file).suffix
    outputs: list[str] = list()
    print(f"Extracting {len(endpoints)} video sequences")
    for i, (start, end) in enumerate(endpoints):
        progress_bar(i+1, len(endpoints), bar_length=50)
        if not path.exists(output := path.join(Path(input_file).parent, f"{Path(input_file).stem}_CUT", f"{base_name}_{i}{extension}")):
            cut_video_file(
                input_file=input_file,
                output_file=output,
                start_timecode=start,
                end_timecode=end,
                avoid_freezes=True,
                delta=1.0
            )
        outputs.append(output)
    return outputs


def extract(input_file: str, audio_to_skip: list[int], recombine: bool = False) -> None:
    """Main function, calls the pipeline

    Args:
        input_file (str): video input file
        recombine (bool, optional): if output should be merged or not. Defaults to False.
    """

    # We detect the silences and we dump those in a json file
    if not path.exists(json_file := path.join(Path(input_file).parent, f"silences_detected_{Path(input_file).stem}.json")):
        dump(detect_silences(input_file, audio_to_skip), open(
            json_file, 'w', encoding='utf-8'), indent=3)
    # We use the json file to do the actual extraction
    cut_files: list = extract_parts(input_file, parts_to_keep(0.0, get_duration(
        input_file), load(open(json_file, "r", encoding='utf-8'))))
    # If we ask for, we merge the files in a single one
    if recombine:
        concat(cut_files, f"output{Path(input_file).suffix}")


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("input", type=str, help="Path to input video file.")
    parser.add_argument("-s", "--skipaudiotracks", type=int, nargs='*',
                        help="Intcodes of audio tracks to skip during analysis", default=[])
    args = parser.parse_args()

    extract(args.input, [int(track) for track in args.skipaudiotracks])
    print("Job ended sucessfully!")
