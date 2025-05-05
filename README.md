# VideoHallu: Evaluating and Mitigating Multi-modal Hallucinations for Synthetic Videos

[Zongxia Li*](https://zli12321.github.io/), [Xiyang Wu*](https://wuxiyang1996.github.io/), [Yubin Qin](https://www.linkedin.com/in/yubin-qin/), [Hongyang Du](https://github.com/SmashedPython), [Guangyao Shi](https://guangyaoshi.github.io/), [Dinesh Manocha](https://www.cs.umd.edu/people/dmanocha), [Tianyi Zhou](https://tianyizhou.github.io/), [Jordan Lee Boyd-Graber](https://users.umiacs.umd.edu/~ying/)

[[📖 Paper](https://github.com/zli12321/VideoHallu/blob/main/paper.pdf)] [[🤗 Dataset](https://huggingface.co/datasets/zli12321/VideoHalluB)] [[🌍Website](https://smashedpython.github.io/videohallu.github.io/)]

<img src="./images/teaser.png" style="zoom:20%;" />

## 👀 About VideoHallu

Synthetic video generation using foundation models has gained significant attention due to its realism and broad applications. However, while these models excel at generating visually coherent and high-quality video frames, they often overlook commonsense reasoning and physical law violations, leading to abnormal content. Existing score-based evaluations like [VideoScore](https://arxiv.org/abs/2406.15252) mainly focus on general video quality and do not take these abnormalities into account, and offer no explanations of the evaluation results. A more promising evaluation approach is to leverage multi-modal large language models (MLLMs) as interpretable video evaluators, following the approach of [FActScore](https://arxiv.org/abs/2305.14251). However, how well MLLMs can detect these abnormalities in synthetic videos is underexplored. 

Motivated by a more interpretable video generation evaluation, we introduce VideoHallu, a benchmark built from synthetic videos produced by popular models like [Sora](https://openai.com/sora/), [Veo2](https://veo2.ai), [Kling](https://www.klingai.com/global/), paired with expert-crafted question-answering pair examples easily solvable with human-level perception and reasoning across multiple categories. We evaluate several State-of-the-Art (SoTA) MLLMs with our benchmark, including [GPT-4o](https://openai.com/index/hello-gpt-4o/), [Gemini-2.5-Pro](https://deepmind.google/technologies/gemini/pro/), [Qwen-2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), and forefront models like [Video-R1](https://github.com/tulerfeng/Video-R1) and [VideoChat-R1](https://github.com/OpenGVLab/VideoChat-R1). Despite the strong performance of R1 MLLMs on real-world video benchmarks like [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench) and [MovieChat](https://github.com/rese1f/MovieChat), these models still struggle and hallucinate on basic commonsense and physics reasoning tasks in synthetic videos, highlighting synthetic video hallucination as an underexplored challenge. 

Moreover, we post-train current SoTA MLLMs, [Qwen-2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), with [Group Relative Policy Optimization (GRPO)](https://arxiv.org/abs/2501.12948) using both real-world and synthetic commonsense/physics datasets. Our results show improved overall accuracy compared to the base model, achieving the highest performance among all models, highlighting the importance of integrating high-quality counterexamples to enhance commonsense and physics reasoning in MLLMs' language priors.

## 🔥 News
- [2025/05/02] We expand our dataset with more QA pairs🤗.
- [2025/05/02] We release our [datasets](https://huggingface.co/datasets/IntelligenceLab/VideoHallu)🤗.
- [2025/05/02] We release our GRPO free-form [RewardModel](https://huggingface.co/IntelligenceLab/RewardPreferenceBert/settings)🤗.


## Table of Contents
* 1. [Dataset](#dataset)
* 2. [Reward Model](#rb)
* 2. [Training](#training)

## 🔍 <a name='dataset'></a>Dataset

To facilitate GRPO training, we also randomly sample 1,000 videos from [PhysBench](https://huggingface.co/datasets/WeiChow/PhysBench-train) training data to first improve model' reasoning abilities in real-world videos, then train the model on part of our synthetic videos.

Our data spans the following categories:

<img src="./images/fig1.png" style="zoom:20%;" />


## Getting Started

```
# Download the synthetic dataset
pip install huggingface_hub

# Download data to your local dir
huggingface-cli download IntelligenceLab/VideoHallu --repo-type dataset --local-dir ./new_video_folders --local-dir-use-symlinks False

# Download and unzip the physben training data videos
curl -L -o video.part1.rar https://huggingface.co/datasets/WeiChow/PhysBench-train/resolve/main/video.part1.rar

# Unzip data (linux system)
unrar x video.part1.rar
```




## The Dawn of MLLMs in Synthetic Videos 🧠 

<div style="border: 2px solid #ddd; border-radius: 10px; padding: 16px; background-color: #f9f9f9; box-shadow: 1px 1px 5px rgba(0,0,0,0.05);">

<details open>
<summary><strong>🎬 Video:</strong> Quail Transforming into rooster</summary>

<p align="center">
  Prompt (Sora): Generate a quail and a rooster celebrating New Year.
  <img src="images/rooster.gif" width="700"/>
  <img src="./images/131021746146018_.pic.jpg" style="zoom:20%;" />
  
</p>
</details>

<div style="border: 2px solid #ddd; border-radius: 10px; padding: 16px; background-color: #f9f9f9; box-shadow: 1px 1px 5px rgba(0,0,0,0.05);">

<details open>
<summary><strong>🎬 Video:</strong> Object Falling and Law of Physics</summary>

<p align="center">
  Prompt (Veo2): A feather and a heavy rock are released at the same height and begin to fall to the ground on Earth.
  <img src="images/feather_veo2.gif" width="700"/>
  <img src="./images/130281746130630_.pic.jpg" style="zoom:20%;" />
  
</p>
</details>

<div style="border: 2px solid #ddd; border-radius: 10px; padding: 16px; background-color: #f9f9f9; box-shadow: 1px 1px 5px rgba(0,0,0,0.05);">

<details open>
<summary><strong>🎬 Video:</strong> Object contact obnormalities</summary>

<p align="center">
  Prompt (Sora): Generate a man drinking up a cup of wine. 
  <img src="images/man_drinking_wine.gif" width="700"/>
  <img src="./images/130291746131015_.pic.jpg" style="zoom:20%;" />
  
</p>
</details>

<div style="border: 2px solid #ddd; border-radius: 10px; padding: 16px; background-color: #f9f9f9; box-shadow: 1px 1px 5px rgba(0,0,0,0.05);">

<details open>
<summary><strong>🎬 Video:</strong> Breaking process</summary>

<p align="center">
  Prompt (Sora): Generate the sequence showing a bullet being shot into a watermelon. 
  <img src="images/watermelon_explode-ezgif.com-video-to-gif-converter.gif" width="700"/>
  <img src="./images/133151746288503_.pic.jpg" style="zoom:20%;" />
  
</p>
</details>


## 🚀 <a name='rb'></a>Reward Model
We use [ModernBERT](https://huggingface.co/docs/transformers/en/model_doc/modernbert) as the base model to finetune on [MOCHA](https://arxiv.org/abs/2010.03636), [Prometheus-preference](https://huggingface.co/datasets/prometheus-eval/Preference-Collection), [Pedants](https://arxiv.org/abs/2402.11161) to evaluate free-form text generations. We use RewardBert as the reward in GRPO finetuning.

#### Method: `compute_score`
**Parameters**
- `reference_answer` (list of str): A list of gold (correct) answers to the question
- `candidate_answer` (str): The answer provided by a candidate that needs to be evaluated

**Returns**
- `tuple`: A tuple of normalized and raw scores.

```python
from qa_metrics.RewardBert import RewardBert

rb = RewardBert(device='cuda')
reference_answer = "The Frog Prince"
candidate_answer = "The movie \"The Princess and the Frog\" is loosely based off the Brother Grimm's \"Iron Henry\""
rb.compute_score(reference_answer, candidate_answer)
# (0.29113227128982544, 2.1645290851593018)
```

## 🚀 <a name='training'></a>Training Set up

We adopt [Video-R1](https://github.com/tulerfeng/Video-R1) training code to finetune model.

Use our formatted json file (synthetic_data.json and physbench_data.json) and follow their setup to train a model.

## Acknowledgements

We sincerely appreciate the contributions of the open-source community. The related projects are as follows: [R1-V](https://github.com/Deep-Agent/R1-V) , [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) , [Video-R1](https://github.com/tulerfeng/Video-R1), [Qwen-2.5-VL](https://arxiv.org/abs/2502.13923)

## Citations

If you find our work helpful for your research, please consider citing our work.   

```
@article{li2025video,
  title={VideoHallu: Evaluating and Mitigating Multi-modal Hallucinations for Synthetic Videos},
  author={{Zongxia Li and Xiyang Wu and Yubin Qin and Guangyao Shi and Hongyang Du and Dinesh Manocha and Tianyi Zhou and Jordan Lee Boyd-Graber}},
  journal={},
  year={2025}
}


@misc{li2025surveystateartlarge,
      title={A Survey of State of the Art Large Vision Language Models: Alignment, Benchmark, Evaluations and Challenges}, 
      author={Zongxia Li and Xiyang Wu and Hongyang Du and Fuxiao Liu and Huy Nghiem and Guangyao Shi},
      year={2025},
      eprint={2501.02189},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.02189}, 
}

@misc{guan2024hallusionbenchadvanceddiagnosticsuite,
      title={HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models}, 
      author={Tianrui Guan and Fuxiao Liu and Xiyang Wu and Ruiqi Xian and Zongxia Li and Xiaoyu Liu and Xijun Wang and Lichang Chen and Furong Huang and Yaser Yacoob and Dinesh Manocha and Tianyi Zhou},
      year={2024},
      eprint={2310.14566},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2310.14566}, 
}

@misc{wu2024autohallusionautomaticgenerationhallucination,
      title={AutoHallusion: Automatic Generation of Hallucination Benchmarks for Vision-Language Models}, 
      author={Xiyang Wu and Tianrui Guan and Dianqi Li and Shuaiyi Huang and Xiaoyu Liu and Xijun Wang and Ruiqi Xian and Abhinav Shrivastava and Furong Huang and Jordan Lee Boyd-Graber and Tianyi Zhou and Dinesh Manocha},
      year={2024},
      eprint={2406.10900},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.10900}, 
}
```
