# Play Tic-Tac-Toe with Reinforcement Learning

![gameplay-animation](figures/1_minimax_vs_minimax_game_play.gif)


## Content

Implement simple tictactoe game in Python with text-based output and learn how to play it using the Minmax algorithm, tabular Q-learning and deep Q-learning with dense and covolutional neural networks using a dual network architecture, dueling network and (prioritised) experience replay. The neural networks are implemented using Keras and TensorFlow. I took inspiration and validated the results with Casten Friedrich's tutorial on implementing reinforcement agents for tictactoe https://github.com/fcarsten/tic-tac-toe.

Please see my blog posts explaining the code in case of interest:
-  ![Computers learning Tic-Tac-Toe Pt. 1: Tabular Q-learning](https://kunkelalexander.github.io/blog/computers-learning-tic-tac-toe-tabular-q/)
![tabular_q](figures/readme_0.png)
-  ![Computers learning Tic-Tac-Toe Pt. 2: Deep Q-learning](https://kunkelalexander.github.io/blog/computers-learning-tic-tac-toe-deep-q/)
![deep_q](figures/readme_1.png)
-  ![Computers learning Tic-Tac-Toe Pt. 3: Optimisation](https://kunkelalexander.github.io/blog/computers-learning-tic-tac-toe-optimisation/)
<img src="figures/12_profiling.svg" height="600">
-  ![Computers learning Tic-Tac-Toe Pt. 4: Towards Rainbow DQN](https://kunkelalexander.github.io/blog/computers-learning-tic-tac-toe-advanced-deep-q/)
![advanced_deep_q](figures/readme_4.png)
![final_agents](ensemble_runs/26_dqn_variant_comparison.png)
The figures were generated using the respective Jupyter notebooks.

The source code can be found in the ```src``` folder. The game is implemented in ```src/tictactoe.py```. The training infrastructure is implemented in the files ```src/training_manager.py``` and ```src/game_manager.py```. The different agents are derived from the ```Agent``` class defined in ```src/agent.py```. They implement the different reinforcement learning strategies.


## Figures


## Usage

Install requirements via

```
pip install -r requirements.txt
```

Find a comparison of different network architectures for three test cases (play second against agent moving randomly, play first against minmax agent, play second against non-deterministic minmax agent) in the Jupyter notebook

```
analysis.ipynb
```
