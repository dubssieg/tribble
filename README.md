# tribble: Silence-exclusion based video editing script

:warn: Requires `ffmpeg` and `ffprobe` executables!

tribble is a command-line tool to edit videos by removing silence parts in it, to speed up the editing process of videos.
It handles multi-track video formats, analysing each track and computing the intersection of silences, to keep the maximum of parts with sound.
