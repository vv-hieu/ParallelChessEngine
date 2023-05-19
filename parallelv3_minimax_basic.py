import warnings
warnings.filterwarnings('ignore')

from ChessEngine import *
from numba.cuda.cudadrv.driver import CudaAPIError
from numba import cuda
import pandas as pd
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fen')
parser.add_argument('--search_depth')

args = parser.parse_args()
if args.search_depth is None:
    args.search_depth = 1
else:
    args.search_depth = int(args.search_depth)

game = ChessGame(args.fen)

cuda.profile_start()
start = time.time()
move, score = find_move(game, args.search_depth, version = 9)
end = time.time()
cuda.profile_stop()

fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR' if args.fen is None else args.fen

print(f'FEN             : {fen}')
print(f'Execution time  : {round(end - start, 5)} s')
print(f'Score           : {round(score, 3)}')
print(f'No. Evaluations : {get_evaluation_count()}')
print(f'Encoded move    : {encode_move(move)}')
