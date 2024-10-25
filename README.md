# Google Technology Equivalents for Ex-Googlers for AI/ML

This [repo](https://github.com/jhuangtw/xg2xg) covers a lot of the technologies and services that Google uses internally to help ex-googlers survive the *real* world.

This repo further extend a handy lookup table of similar technology and services to help ex-googlers to build AI/ML :)

pull-requests are very welcomed. __Please do not list any confidential projects!__

## AI/ML Stacks

### Data Curation Pipeline

| Google Internal | Google External                          | Open Source / Real-World                 |
| --------------- | ---------------------------------------- | ---------------------------------------- |
| [Flume](https://ai.google/research/pubs/pub35650)   | [DataFlow](https://cloud.google.com/dataflow)  | [Apache Beam](https://beam.apache.org/) |

### Data loader

| Google Internal | Google External                          | Open Source / Real-World                 |
| --------------- | ---------------------------------------- | ---------------------------------------- |
| [tfds](https://www.tensorflow.org/datasets), [grain](https://github.com/google/grain)   | [tfds](https://www.tensorflow.org/datasets), [grain](https://github.com/google/grain)  | [torchdata](https://github.com/pytorch/data), [datasets](https://github.com/huggingface/datasets) |

### Experiment Management and hyperparameter tuning

| Google Internal | Google External                          | Open Source / Real-World                 |
| --------------- | ---------------------------------------- | ---------------------------------------- |
| [xmanager](https://github.com/google-deepmind/xmanager)   | [xmanager](https://github.com/google-deepmind/xmanager)  | [wandb](https://github.com/wandb/wandb), [mlflow](https://github.com/mlflow/mlflow), [tensorboard](https://github.com/tensorflow/tensorboard), [slurm](https://github.com/SchedMD/slurm), [ray](https://github.com/ray-project/ray), [skypilot](https://github.com/skypilot-org/skypilot) |
| [vizier](https://github.com/google/vizier) | [vizier](https://github.com/google/vizier) | [autogluon](https://github.com/autogluon/autogluon) |


### Training

| Google Internal | Google External                          | Open Source / Real-World                 |
| --------------- | ---------------------------------------- | ---------------------------------------- |
| Gemax scale / prod pre-training   |  | [transformers](https://github.com/huggingface/transformers), [composer](https://github.com/mosaicml/composer), [ms-swift](https://github.com/modelscope/ms-swift), [unsloth](https://github.com/unslothai/unsloth) |
| Gemax post-training    |  | [trl](https://github.com/huggingface/trl) |


### Inference/Serving

| Google Internal | Google External                          | Open Source / Real-World                 |
| --------------- | ---------------------------------------- | ---------------------------------------- |
| Sax / Evergreen  |   | [vllm](https://github.com/vllm-project/vllm), [sglang](https://github.com/sgl-project/sglang), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) |


### Evaluation

| Google Internal | Google External                          | Open Source / Real-World                 |
| --------------- | ---------------------------------------- | ---------------------------------------- |
| []() | []() | [evals](https://github.com/openai/evals) |

### Tools

| Google Internal | Google External                          | Open Source / Real-World                 |
| --------------- | ---------------------------------------- | ---------------------------------------- |
| Evergreen Agent   | | [swarm](https://github.com/openai/swarm), [crewai](https://github.com/crewAIInc/crewAI), [autogen](https://github.com/microsoft/autogen) |
| SILM (Self-Improved Language Model)   |   | [langgraph](https://github.com/langchain-ai/langgraph) |
| DASM (Data Assembly)   |   | [dspy](https://github.com/stanfordnlp/dspy) |
| [OneTwo](https://github.com/google-deepmind/onetwo) | | [langchain](https://github.com/langchain-ai/langchain) |

### Visualization

| Google Internal | Google External                          | Open Source / Real-World                 |
| --------------- | ---------------------------------------- | ---------------------------------------- |
| Loupt   |  | [streamlit](https://github.com/streamlit/streamlit), [gradio](https://github.com/gradio-app/gradio) |


*disclaimer: I'm not affiliated with any of the technologies/products mentioned above.*

*disclaimer: I recently left Google so some of the naming might be dated.*
