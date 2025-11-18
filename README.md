# Video Pipeline Overview

This document provides a high level visualization of the modular video generation pipeline.

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

Each module saves its output to the channel folder, allowing partial reprocessing and manual editing.
