# Augmented Reality
Augmented Reality with OpenCV

![demo](docs/demo.gif)

![compare](docs/compare.gif)

### How to Use
For demo please run:
```
python augmented.py \
    --pattern_path ./demo/pattern_dm.png \
    --overlay_path ./demo/overlay_dm.jpg \
    --video_path ./demo/video_dm.mp4 \
    --output_path ./demo/output100.avi \
    --viz
```
For helper, see:
```
usage: augmented.py [-h] --pattern_path PATTERN_PATH --overlay_path
                    OVERLAY_PATH --video_path VIDEO_PATH
                    [--output_path OUTPUT_PATH] [--viz_matches] [--viz]

Augmented Reality

optional arguments:
  -h, --help            show this help message and exit
  --pattern_path PATTERN_PATH
                        image directory path
  --overlay_path OVERLAY_PATH
                        image format, png/jpg
  --video_path VIDEO_PATH
                        video path
  --output_path OUTPUT_PATH
                        output video path if required
  --viz_matches         matches will be draw, but cannot visualize different
  --viz                 visualize different, but cannot draw matches
```

### Easy Way With Docker Container
You can use docker to run the code, please follow:
1. Pull Docker Image
```
docker image pull ruhyadi/augmented-reality:latest
```
2. Run Docker
```
./runDocker.sh
```
if you have trouble with permission denied, please run:
```
chmod 666 ./runDocker.sh
```
3. Run Augmented Reality
Run this code inside docker terminal
```
python augmented.py \
    --pattern_path ./demo/pattern_dm.png \
    --overlay_path ./demo/overlay_dm.jpg \
    --video_path ./demo/video_dm.mp4 \
    --viz
```

### Easiest Way with Docker Compose
Just run code below:
```
./runDockerCompose.sh
```
or with terimal, run:
```
docker-compose up
```

### Reference
- [juangallostra/augmented-reality](https://github.com/juangallostra/augmented-reality)