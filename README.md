[![](https://img.shields.io/badge/python->=3.10-blue.svg)]()
[![](https://img.shields.io/badge/documentation-unfinished-orange.svg)]()

# tribble: Silence-exclusion based video editing script

tribble is a command-line tool to edit videos by removing silence parts in it, to speed up the editing process of videos.
It handles multi-track video formats, analysing each track and computing the intersection of silences, to keep the maximum of parts with sound. You may select the audio tracks you want to listen to, and even directly merge all files to an output one!

> [!WARNING]\
> Requires `ffmpeg` and `ffprobe` executables!

![](https://media.discordapp.net/attachments/874430800802754623/1193189586638217357/image.png?ex=65abcf65&is=65995a65&hm=0fafb4cdca2ce085f0dd92159c8e5b0c57c91c1357b924d2fe106795b15bec36&=&format=webp&quality=lossless&width=1440&height=137)

## Requirements

+ Python >= 3.10
+ [ffmpeg and ffprobe](https://ffmpeg.org/) (directly at script hierarchy level)
+ pip packages : tharos-pytools, ffprobe-python, pydub

> [!NOTE]\
> Want to contribute? Feel free to open a PR on an issue about a missing, buggy or incomplete feature!

## Tasks

- [x] Proof-of-concept
- [x] Default parameters
- [ ] Documentation
- [ ] Command-line command
- [ ] Build files and release