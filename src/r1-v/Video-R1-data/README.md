---
license: apache-2.0
task_categories:
- video-text-to-text
language:
- en
size_categories:
- 100K<n<1M
---

This repository contains the data presented in [Video-R1: Reinforcing Video Reasoning in MLLMs](https://huggingface.co/papers/2503.21776).

Code: https://github.com/tulerfeng/Video-R1

Video data folder: CLEVRER, LLaVA-Video-178K, NeXT-QA, PerceptionTest, STAR

Image data folder: Chart, General, Knowledge, Math, OCR, Spatial

Video-R1-COT-165k.json is for SFT cold start, and Video-R1-260k.json is for RL training.

Data Format in Video-R1-COT-165k:

```
  {
      "problem_id": 2,
      "problem": "What appears on the screen in Russian during the missile's ascent?",
      "data_type": "video",
      "problem_type": "multiple choice",
      "options": [
          "A. A YouTube subscription notification",
          "B. A military command",
          "C. A warning message",
          "D. A weather update"
      ],
      "process": "<think>Let me think about this for a moment. The video shows a missile being launched, and there's a pop-up that appears in Russian during the missile's ascent. Hmm, what could it be? Wait, it looks familiar. It resembles something I've seen before online.\n\nOh, I see! It's a YouTube pop-up, specifically a notification that says \"Вы подписаны,\" which translates to \"You are subscribed.\" This is a common feature on YouTube when someone subscribes to a channel. Let's break it down: it's not a military command, a warning message, or a weather update—it's clearly related to YouTube.\n\nSo, after some self-reflection and verification, I'm confident that the pop-up is indeed a YouTube subscription notification.</think>",
      "solution": "<answer>A</answer>",
      "path": "./LLaVA-Video-178K/liwei_youtube_videos/videos/youtube_video_2024/ytb_7nRmsEw7nsE.mp4",
      "data_source": "LLaVA-Video-178K/30_60_s_youtube_v0_1"
  }
```