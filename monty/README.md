# Monty Experiments

This directory contains all experiments from the paper that use the Monty framework. In this file, you can find instructions for [setting up an appropriate python environment](#environment-installation), [running experiments](#running-experiments), and an [exhaustive summary of all experiments](#summary-of-experiments) defined under the `configs` directory.

## Environment Installation

The environment for this project is managed with [conda](https://www.anaconda.com/download/success).

To create the environment, run the following command, tailored to your system:

### ARM64 (Apple Silicon) (zsh shell)
```
conda env create -f environment.yml --subdir=osx-64
conda init zsh
conda activate tbs_sensorimotor_intelligence
conda config --env --set subdir osx-64
```

### ARM64 (Apple Silicon) (bash shell)
```
conda env create -f environment.yml --subdir=osx-64
conda init
conda activate tbs_sensorimotor_intelligence
conda config --env --set subdir osx-64
```

### Intel (zsh shell)
```
conda env create -f environment.yml
conda init zsh
conda activate tbs_sensorimotor_intelligence
```

### Intel (bash shell)
```
conda env create -f environment.yml
conda init
conda activate tbs_sensorimotor_intelligence
```

## Further Setup

At a minimum, you'll need to download the YCB object dataset before you can run experiments, though you can also download our pretrained models if you only wish to run evaluation experiments. Please see [Downloading Datasets](../README.md#downloading-datasets) for download instructions and [Filesystem Setup](../README.md#filesystem-setup) to optionally configure the directories used to store downloaded or generated datasets.

## Running Experiments

Experiments are defined in the `configs` directory, including for experiments for pre-training models from scratch. Experiments can be run using the `run.py` or `run_parallel.py` scripts located in this directory with

```bash
python run.py -e <experiment_name>
```
or
```bash
python run_parallel.py -e <experiment_name> -n <num_parallel>
```
The `-n` argument is optional, and it will be set automatically if not provided. 

**Note**: Not all experiments can be run in parallel. Use [this table](#run-settings) to determine whether to use `run.py` or `run_parallel.py`.

## Summary of Experiments

Below is an exhaustive list of the experiment configs used in this paper, along with descriptions motivating the choice of parameters. This summary is largely organized by figure, the exceptions being configs used to generate [pretrained models](#pretraining-experiments), [data used for visualizations](#visualization-experiments), and [images used by ViT model](#other-experiments).

**Important**: Figures 5 and 6 in this repository were swapped in the final draft of our paper. `configs/fig5.py` contains experiments used for figure 6 in the paper, and `configs/fig6.py` contain experiments used for figure 5.

### Figure 3: Robust Sensorimotor Inference

This figure presents results from five inference experiments testing Monty's robustness under different conditions. Monty was pre-trained on 14 standard rotations derived from cube face and corner views (see full configuration details in `configs/pretraining_experiments/dist_agent_1lm`).

Consists of 5 experiments:
- `dist_agent_1lm`: Standard inference with no sensor noise or random rotations
- `dist_agent_1lm_noise_all`: Tests robustness to heavy sensor noise
- `dist_agent_1lm_randrot_14`: Tests performance across 14 random rotations, not seen during training
- `dist_agent_1lm_randrot_14_noise_all`: Tests performance with both random rotations and heavy sensor noise
- `dist_agent_1lm_randrot_14_noise_all_color_clamped`: Tests performance with random rotations, heavy sensor noise, and with the color feature for each observation clamped
to blue.
  
Here we are showing the performance of the "standard" version of Monty, using:
- 77 objects
- 14 rotations
- Goal-state-driven/hypothesis-testing policy active
- A single LM (no voting)

The main output measures are accuracy and rotation error (degrees) for each condition.

### Figure 4: Structured Object Representations

Consists of 1 experiment:
- `surf_agent_1lm_randrot_noise_10simobj`

This means performance is evaluated with:
- 10 morphologically similar objects
- 5 random rotations
- Sensor noise
- Hypothesis-testing policy active
- No voting

The main output measure is a dendrogram showing evidence score clustering for the 10 objects.

**Notes:**
- Although evaluating on 10 objects, the model is trained on 77 objects.
  
### Default Parameters for Figures 5+
Unless specified otherwise, the following figures/experiments use:
- 77 objects
- 5 predefined "random" rotations. These rotations were randomly generated but are kept constant across experiments. 
- Sensor noise

This captures core model performance in a realistic setting.

### Figure 5: Rapid Inference with Voting

Note in our final paper, these results appear in the "Figure 6", after "Rapid Inference
with Model-Free and Model-Based Policies"

Consists of 5 experiments:
- `dist_agent_1lm_randrot_noise`
- `dist_agent_2lm_randrot_noise`
- `dist_agent_4lm_randrot_noise`
- `dist_agent_8lm_randrot_noise`
- `dist_agent_16lm_randrot_noise`

For single-LM experiments, an episode terminates when the LM has converged onto a
object/pose estimate. For the multi-LM experiments in this paper, a minimum of two LMs
must converge before termination regardless of the number of LMs. (Note that episodes
time-out after 500 steps for all experiments if the convergence criteria is not met.)

Performance is evaluated on:
- 77 objects
- Goal-state-driven/hypothesis-testing policy active
- Sensor noise and 5 random rotations
- Voting over 2, 4, 8, or 16 LMs

The main output measure is accuracy and rotation error as a function of the number of LMs.

### Figure 6: Rapid Inference with Model-Free and Model-Based Policies

Note in our final paper, these results appear in the "Figure 5", before "Rapid Inference
with Voting"

Consists of 3 experiments:
- `dist_agent_1lm_randrot_noise_nohyp` - No hypothesis-testing, and random-walk policy
- `surf_agent_1lm_randrot_noise_nohyp` - Model-free policy to explore surface
- `surf_agent_1lm_randrot_noise` - Default, i.e. model-free and model-based policies

This means performance is evaluated with:
- 77 objects
- Sensor noise and 5 random rotations
- No voting
- Varying policies; the surface agent (i.e. with color etc) gets the same kind of sensory information as the distant agent, and so differs only in its model-free policy that encourages rapid exploration of the surface of the object. We can make it clear in the paper that there is nothing preventing the distant agent from also having model-free and model-based policies.

The main output measure is accuracy and rotation error as a function of the policy used.

### Figure 7a: Rapid Learning

Configs defined 
Consists of 7 experiments:
- `pretrain_dist_agent_1lm_checkpoints`
- `dist_agent_1lm_randrot_nohyp_1rot_trained`
- `dist_agent_1lm_randrot_nohyp_2rot_trained`
- `dist_agent_1lm_randrot_nohyp_4rot_trained`
- `dist_agent_1lm_randrot_nohyp_8rot_trained`
- `dist_agent_1lm_randrot_nohyp_16rot_trained`
- `dist_agent_1lm_randrot_nohyp_32rot_trained`

This means performance is evaluated with:
- 77 objects
- 5 random rotations
- NO sensor noise*
- NO hypothesis-testing*
- No voting
- Varying numbers of rotations trained on (evaluations use different baseline models)

*No hypothesis-testing as the ViT model comparison only receives one view and cannot move around object, and no noise since Sensor-Module noise does not have a clear analogue for the ViT model.

The main output measure is accuracy and rotation error as a function of training rotations.

**Notes:**
- Training rotations are ordered as:
  1. First 6 rotations = cube faces
  2. Next 8 rotations = cube corners
  3. Remaining = random rotations (as otherwise introduces redundancy)

### Figure 7B: Continual Learning

Consists of 78 experiments:
- `pretrain_continual_learning_dist_agent_1lm_checkpoints`
- `continual_learning_dist_agent_1lm_task0`
- `continual_learning_dist_agent_1lm_task1`
- `continual_learning_dist_agent_1lm_task2`
- ...
- `continual_learning_dist_agent_1lm_task76`

As with [Figure 7: Rapid Learning](#figure-7a-rapid-learning), performance is evaluated with:
- N objects seen in pretraining
- 5 random rotations
- NO sensor noise*
- NO hypothesis-testing*
- No voting

*No hypothesis-testing as the ViT model comparison only receives one view and cannot move around object, and no noise since Sensor-Module noise does not have a clear analogue for the ViT model.

The main output measure is accuracy as a function of number of objects seen so far.

### Figure 8: Computationally Efficient Learning and Inference

There are two experiments, one using hypothesis testing and another using no hypothesis testing.

- `dist_agent_1lm_randrot_nohyp`
- `dist_agent_1lm_randrot`

**Notes:**

This performance is evaluated with:

- 77 objects
- 5 random rotations
- No sensor noise*
- No voting

*Due to ViT model comparison.

The main output measure is accuracy and FLOPs as a function of whether hypothesis testing was used or not.

### Pretraining Experiments
`pretraining_experiments.py` defined pretraining experiments that generate models
used throughout this repository. They are required for running eval experiments,
visualization experiments, and for many of the figures generated in the `scripts`
directory. The following is a list of pretraining experiments and the models they produce:
 - `pretrain_dist_agent_1lm` -> `dist_agent_1lm`
 - `pretrain_surf_agent_1lm` -> `surf_agent_1lm`
 - `pretrain_dist_agent_2lm` -> `dist_agent_2lm`
 - `pretrain_dist_agent_4lm` -> `dist_agent_4lm`
 - `pretrain_dist_agent_8lm` -> `dist_agent_8lm`
 - `pretrain_dist_agent_16lm` -> `dist_agent_16lm`

All of these models are trained on 77 YCB objects with 14 rotations each (cube face
and corners).

### Visualization Experiments

`visualizations.py` contains configs defined solely for making visualizations that go into
paper figures. The configs defined are:
- `fig2_object_views`: A one-object experiment that saves high-resolution images from
  the view-finder. Used to create images of the `potted_meat_can` in figure 2.
- `fig2_pretrain_surf_agent_1lm_checkpoints`: A pretraining experiment that saves
  checkpoints for the 14 training rotations. The output is read and plotted by
  functions in `scripts/fig2.py`.
- `fig3_evidence_run`: A one-episode distant agent experiment used to collect evidence
   and sensor data for every step. The output is read and plotted by functions in
    `scripts/fig3.py`.
- `fig4_symmetry_run`: Runs `dist_agent_1lm_randrot_noise` with storage of
   evidence and symmetry including symmetry data for the MLH object only, and only
   for the terminal step of each episode. The output is read and plotted by
   functions in `scripts/fig4.py`.
- `fig5_visualize_8lm_patches`: A one-episode, one-step experiment that is used to
  collect one set of observations for the 8-LM model. The output is read and plotted
  by functions in `scripts/fig5.py` to show how the sensors patches fall on the object.
- `fig6_curvature_guided_policy`: A one-episode surface agent experiment with
  no hypothesis-testing policy active. The output is read and plotted by
  functions in `scripts/fig6.py`.
- `fig6_hypothesis_driven_policy`: A one-episode surface agent experiment with
  hypothesis-testing policy active. The output is read and plotted by
  functions in `scripts/fig6.py`.

All of these experiments should be run in serial due to the memory needs of
detailed logging (or checkpoint-saving in the case of
`fig2_pretrain_surf_agent_1lm_checkpoints`).

All experiments save their results to subdirectories of `DMC_ROOT` / `visualizations`.

### Other Experiments
`view_finder_experiments.py` defines three experiments:
- view_finder_base: 14 standard training rotations
- view_finder_randrot: 5 pre-defined "random" rotations
- view_finder_32: 32 training rotations for rapid learning experiments
  
These experiments are not used for object recognition in Monty. Rather, they use Monty to capture and store images of objects in the YCB dataset. Arrays stored during these experiments can be rendered by `scripts/render_view_finder_images.py`.

## Run Settings

Use the following table to determine whether to run an experiment with `run.py` or `run_parallel.py`.

| Experiment | parallel |
|------------|:---:|
| pretrain_dist_agent_1lm | yes |
| pretrain_surf_agent_1lm | yes |
| pretrain_dist_agent_2lm | yes |
| pretrain_dist_agent_4lm | yes |
| pretrain_dist_agent_8lm | yes |
| pretrain_dist_agent_16lm | yes |
| dist_agent_1lm | yes |
| dist_agent_1lm_noise_all | yes |
| dist_agent_1lm_randrot_14 | yes |
| dist_agent_1lm_randrot_14_noise_all | yes |
| dist_agent_1lm_randrot_14_noise_all_color_clamped | yes |
| dist_agent_1lm | yes |
| surf_agent_1lm_randrot_noise_10simobj | no |
| dist_agent_1lm_randrot_noise | yes |
| dist_agent_2lm_randrot_noise | yes |
| dist_agent_4lm_randrot_noise | yes |
| dist_agent_8lm_randrot_noise | yes |
| dist_agent_16lm_randrot_noise | yes |
| dist_agent_1lm_randrot_noise_nohyp | yes |
| surf_agent_1lm_randrot_noise_nohyp | yes |
| surf_agent_1lm_randrot_noise | yes |
| pretrain_dist_agent_1lm_checkpoints | no |
| dist_agent_1lm_randrot_nohyp_[n]rot_trained | yes |
| pretrain_continual_learning_dist_agent_1lm_checkpoints | no |
| continual_learning_dist_agent_1lm_task[n] | yes |
| dist_agent_1lm_randrot_nohyp | no |
| dist_agent_1lm_randrot | no |
| view_finder_base | no |
| view_finder_randrot | no |
| view_finder_32 | no |
| fig2_object_views | no |
| fig2_pretrain_surf_agent_1lm_checkpoints | no |
| fig3_evidence_run | no |
| fig4_symmetry_run | no |
| fig5_visualize_8lm_patches | no |
| fig6_curvature_guided_policy | no |
| fig6_hypothesis_driven_policy | no |

Note that `dist_agent_1lm_randrot_nohyp_[n]rot_trained` stands in for all variations of this experiment, where `n` is one of 1, 2, 4, 8, or 16. Similarly, `continual_learning_dist_agent_1lm_task[n]` stands in for all variations of the continual learning group, where `n` is one of 0, 1, 2, 3, ..., 76.
