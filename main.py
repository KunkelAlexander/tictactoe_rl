import argparse 

from tictactoe import TicTacToe
from training_manager import TrainingManager

# parse input parameters
argParser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argParser.add_argument("-n", "--n_episode",   type=int, help="Number of episodes for training.", default=50000)
argParser.add_argument("-ne","--n_eval",      type=int, help="Number of episodes per evaluation.", default=20)
argParser.add_argument("-ef","--eval_freq",   type=int, help="Frequency for evaluation in training episodes.", default=500)
argParser.add_argument("-w", "--window_size", type=int, help="Window size for plotting.", default=500)
argParser.add_argument("-s", "--board_size",  type=int, help="Size N of NxN board.", default=3)
argParser.add_argument("-a", "--agents",      type=str, help="Space separated list of agents", default="RANDOM_AGENT TABULAR_Q_AGENT")

args = argParser.parse_args()

agent_types = args.agents.split(sep=",")

training_manager = TrainingManager( game = TicTacToe(board_size  = args.board_size, agent_count = len(agent_types)), 
                                    agent_types = agent_types,
                                    gui  = gui)

training_manager.run_training(
                            board_size          = args.board_size, 
                            n_episode           = args.n_episode, 
                            n_eval              = args.n_eval,
                            eval_freq           = args.eval_freq,
                            window_size         = args.window_size, 
                            discount            = 0.95,
                            learning_rate_decay = 1, 
                            exploration_decay   = 1,
                            exploration         = 0,
                            learning_rate       = 0.9,
                            randomise_order     = False,
                            only_legal_actions  = True
                            )