# Video Automation Pipeline

This project automatically **creates and uploads full YouTube videos** using AI. You can choose the **topic**, **length**, **voice**, and **style**, and the pipeline will generate everything needed for a complete video.

Think of it as a factory line for YouTube videos: each step produces a piece of the final video, and you can re-run any step without starting over.

## What it does (in plain English)

- **Plans the video** by creating an outline and a script.
- **Creates a voiceover** using text-to-speech.
- **Plans and generates visuals** (images and short clips).
- **Builds the final video**, including a thumbnail.
- **Generates metadata** like titles and descriptions.
- **Uploads to YouTube**.

Each module saves its output to the channel folder, allowing partial reprocessing and manual editing.

## Pipeline modules (high-level overview)

```
[ Module 1 ] Outline Generator
        |
        v
[ Module 2 ] Script Builder
        |
        v
[ Module 3 ] Voice Generator (TTS)
        |
        v
[ Module 4 ] Media Planner
        |
        v
[ Module 5 ] Image Generator
        |
        v
[ Module 6 ] Short Video Generator
        |
        v
[ Module 7 ] Video Composer
        |
        v
[ Module 8 ] Thumbnail Creator
        |
        v
[ Module 9 ] Metadata Generator
        |
        v
[ Module 10 ] Uploader
```

## What you can configure

- **Topic**: what the video is about.
- **Length**: how long the final video should be.
- **Voice**: the text-to-speech voice used for narration.
- **Style**: the visual and scripting tone (e.g., educational, upbeat, cinematic).

## Repo layout

- `modules/`: Individual pipeline modules.
- `assets/`: Supporting assets used by the pipeline.
- `channels.json`: Channel configuration.
- `run.py`: Entry point for running the pipeline.

## Getting started

1. Review `channels.json` to configure your channel(s).
2. Explore `modules/` to understand or customize each step.
3. Run the pipeline via `run.py` and iterate on the module outputs in the channel folder.
