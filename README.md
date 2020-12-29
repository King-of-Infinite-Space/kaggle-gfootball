Attempted  Kaggle competition *Google Research Football with Manchester City F.C.* The goal is to develop an agent in a football game (similar to FIFA/PES but much more crude). The controlled player chooses an action based on observation. 

I don't know why MCFC is interested in reinfocement learning for football game -- without a very realistic physics engine it is just a video game and will not be relevant to actual football.

This project was cloned from Google Research Football repo and works in Ubuntu 18.04. Original readme attached below. 

My agent is rule-based. When in possession, it tries to maximize a "value function" by running/passing towards goal. However, since everything was hand-written, it quickly became too complicated to tune, and wasn't able to exploit the environment. It performs much worse than some simpler rule-based bots. As shown in an example below, combining this approach with ML can lead to much better results.

#### Other participants' agents
- [Yegor Biryukov](https://www.kaggle.com/yegorbiryukov/gfootball-with-memory-patterns) -- a public rule based agent, simple but effective
- [HuiUnited](https://www.kaggle.com/huiunited/hui-gfootball-with-memory-patterns-from-yegor) -- impressive 31st place for a rule based agent, based on previous one
- [LiamKirwin](https://www.kaggle.com/c/google-football/discussion/201202) -- 22nd place, hybrid approach. Started with rules using heuristic value functions, then the value function was replaced by a learned model.
- Unsurprisingly, the top teams used a variety of RL approaches. ML by imitating top teams also worked quite well.

#### Examples

- Run game in Kaggle environment
  ```
  python3 play_kaggle.py left_agent right_agent
  ```
- Run game step by step (for debugging purpose, no rendering)
  ```
  python3 test_kaggle.py left_agent right_agent
  ```
- Run game in Google Football environment (I couldn't make the Kaggle agents work in this environment. Manual control is not handy at all, perhaps partly due to low FPS.)
  ```
  python3 -m gfootball.play_game --players "gamepad:left_players=1;builtin_ai:right_players=1" --action_set=full --level='11_vs_11_kaggle'
  ```
  

---

## Google Research Football

This repository contains an RL environment based on open-source game Gameplay
Football. <br> It was created by the Google Brain team for research purposes.

Useful links:

* __(NEW!)__ [GRF Kaggle competition](https://www.kaggle.com/c/google-football) - take part in the competition playing games against others, win prizes and become the GRF Champion!
* [GRF Game Server](https://research-football.dev/) - challenge other researchers (replaced by GRF Kaggle competition, frozen ranking available).
* [Run in Colab](https://colab.research.google.com/github/google-research/football/blob/master/gfootball/colabs/gfootball_example_from_prebuild.ipynb) - start training in less that 2 minutes.
* [Google Research Football Paper](https://arxiv.org/abs/1907.11180)
* [GoogleAI blog post](https://ai.googleblog.com/2019/06/introducing-google-research-football.html)
* [Google Research Football on Cloud](https://towardsdatascience.com/reproducing-google-research-football-rl-results-ac75cf17190e)
* [Mailing List](https://groups.google.com/forum/#!forum/google-research-football) - please use it for communication with us (comments / suggestions / feature ideas)


For non-public matters that you'd like to discuss directly with the GRF team,
please use google-research-football@google.com.

We'd like to thank Bastiaan Konings Schuiling, who authored and open-sourced the original version of this game.


## Quick Start

### In colab

Open our example [Colab](https://colab.research.google.com/github/google-research/football/blob/master/gfootball/colabs/gfootball_example_from_prebuild.ipynb), that will allow you to start training your model in less than 2 minutes.

This method doesn't support game rendering on screen - if you want to see the game running, please use the method below.

### Using Docker

This is the recommended way to avoid incompatible package versions.
Instructions are available [here](gfootball/doc/docker.md).

### On your computer

#### 1. Install required packages
#### Linux
```
sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc libsdl-sge-dev python3-pip
```

#### Mac OS X
First install [brew](https://brew.sh/). It should automatically install Command Line Tools.
Next install required packages:

```
brew install git python3 cmake sdl2 sdl2_image sdl2_ttf sdl2_gfx boost boost-python3
```
To set up `pygame`, it is also required to install older versions of SDL:

```
brew install sdl sdl_image sdl_mixer sdl_ttf portmidi
```

#### 2a. From PyPi package
```
pip3 install gfootball
```

#### 2b. Installing from sources using GitHub repository

```
git clone https://github.com/google-research/football.git
cd football
```

Optionally you can use [virtual environment](https://docs.python.org/3/tutorial/venv.html):

```
python3 -m venv football-env
source football-env/bin/activate
```

The last step is to build the environment:

```
pip3 install .
```
This command can run for a couple of minutes, as it compiles the C++ environment in the background.

#### 3. Time to play!
```
python3 -m gfootball.play_game --action_set=full
```
Make sure to check out the [keyboard mappings](#keyboard-mappings).
To quit the game press Ctrl+C in the terminal.

# Contents #

* [Running training](#training-agents-to-play-GRF)
* [Playing the game](#playing-the-game)
    * [Keyboard mappings](#keyboard-mappings)
    * [Play vs built-in AI](#play-vs-built-in-AI)
    * [Play vs pre-trained agent](#play-vs-pre-trained-agent)
    * [Trained checkpoints](#trained-checkpoints)
* [Environment API](gfootball/doc/api.md)
* [Observations & Actions](gfootball/doc/observation.md)
* [Scenarios](gfootball/doc/scenarios.md)
* [Multi-agent support](gfootball/doc/multi_agent.md)
* [Running in docker](gfootball/doc/docker.md)
* [Saving replays, logs, traces](gfootball/doc/saving_replays.md)

## Training agents to play GRF

### Run training
In order to run TF training, install additional dependencies
(or alternatively use provided [Docker image](gfootball/doc/docker.md)):

- Update PIP, so that tensorflow 1.15 is available: `python3 -m pip install --upgrade pip setuptools`
- TensorFlow: `pip3 install tensorflow==1.15.*` or
  `pip3 install tensorflow-gpu==1.15.*`, depending on whether you want CPU or
  GPU version;
- Sonnet: `pip3 install dm-sonnet==1.*`;
- OpenAI Baselines:
  `pip3 install git+https://github.com/openai/baselines.git@master`.

Then:

- To run example PPO experiment on `academy_empty_goal` scenario, run
  `python3 -m gfootball.examples.run_ppo2 --level=academy_empty_goal_close`
- To run on `academy_pass_and_shoot_with_keeper` scenario, run
  `python3 -m gfootball.examples.run_ppo2 --level=academy_pass_and_shoot_with_keeper`

In order to train with nice replays being saved, run
`python3 -m gfootball.examples.run_ppo2 --dump_full_episodes=True --render=True`

In order to reproduce PPO results from the paper, please refer to:

- gfootball/examples/repro_checkpoint_easy.sh
- gfootball/examples/repro_scoring_easy.sh

## Playing the game

Please note that playing the game is implemented through an environment, so human-controlled players use the same interface as the agents. One important implication is that there is a single action per 100 ms reported to the environment, which might cause a lag effect when playing.


### Keyboard mappings
The game defines following keyboard mapping (for the `keyboard` player type):

* `ARROW UP` - run to the top.
* `ARROW DOWN` - run to the bottom.
* `ARROW LEFT` - run to the left.
* `ARROW RIGHT` - run to the right.
* `S` - short pass in the attack mode, pressure in the defense mode.
* `A` - high pass in the attack mode, sliding in the defense mode.
* `D` - shot in the the attack mode, team pressure in the defense mode.
* `W` - long pass in the the attack mode, goalkeeper pressure in the defense mode.
* `Q` - switch the active player in the defense mode.
* `C` - dribble in the attack mode.
* `E` - sprint.

### Play vs built-in AI
Run `python3 -m gfootball.play_game --action_set=full`. By default, it starts
the base scenario and the left player is controlled by the keyboard. Different
types of players are supported (gamepad, external bots, agents...). For possible
options run `python3 -m gfootball.play_game -helpfull`.

### Play vs pre-trained agent

In particular, one can play against agent trained with `run_ppo2` script with
the following command (notice no action_set flag, as PPO agent uses default
action set):
`python3 -m gfootball.play_game --players "keyboard:left_players=1;ppo2_cnn:right_players=1,checkpoint=$YOUR_PATH"`

### Trained checkpoints
We provide trained PPO checkpoints for the following scenarios:

  - [11_vs_11_easy_stochastic](https://storage.googleapis.com/grf_public/trained_models/11_vs_11_easy_stochastic_v2),
  - [academy_run_to_score_with_keeper](https://storage.googleapis.com/grf_public/trained_models/academy_run_to_score_with_keeper_v2).

In order to see the checkpoints playing, run
`python3 -m gfootball.play_game --players "ppo2_cnn:left_players=1,policy=gfootball_impala_cnn,checkpoint=$CHECKPOINT" --level=$LEVEL`,
where `$CHECKPOINT` is the path to downloaded checkpoint.

In order to train against a checkpoint, you can pass 'extra_players' argument to create_environment function.
For example extra_players='ppo2_cnn:right_players=1,policy=gfootball_impala_cnn,checkpoint=$CHECKPOINT'.
