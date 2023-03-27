from ChessCore import *
from numba import cuda, types

# Profiling stuff

def get_evaluation_count():
    global evaluation_count
    return evaluation_count

# Move sorting

def move_score(game: ChessGame, move: Move):
    res = 0.0
    capture = game.get_piece(move.end_position())
    if capture != ChessPieces._E:
        res = 10.0 * ChessPieces.value(capture) - ChessPieces.value(game.get_piece(move.start_position()))
    return res

def sort_moves(game: ChessGame, moves: list[Move], best_to_worst: bool = True) -> list[Move]:
    res = moves + []
    res.sort(key=lambda move: move_score(game, move), reverse=best_to_worst)
    return res

# Sequential implementation
# Options: alpha_beta_prunning : Enable/disable alpha beta pruning
#          move_sorting        : Enable/disable move sorting

def evaluate_move_sequential(game: ChessGame, search_depth: int, alpha: float, beta: float, alpha_beta_prunning: bool = False, move_sorting: bool = False) -> float:
    global evaluation_count
    
    evaluation_count += 1

    if search_depth <= 0:
        return game.evaluate()

    if game.get_halfmove() >= 100:
        return 0.0

    moves = game.generate_all_moves()
    if moves is None or len(moves) <= 0:
        if game.is_current_side_in_check():
            return -1000000.0
        return 0.0

    if move_sorting:
        moves = sort_moves(game, moves, best_to_worst=True)

    for move in moves:
        game_copy = game.copy()
        game_copy.move(move)
        eval = -evaluate_move_sequential(game_copy, search_depth - 1, -beta, -alpha, alpha_beta_prunning, move_sorting)
        del game_copy

        if alpha_beta_prunning and eval >= beta:
            return beta

        if eval > alpha:
            alpha = eval

    return alpha

def find_move_sequencial(game: ChessGame, search_depth: int, alpha_beta_prunning: bool = False, move_sorting: bool = False):
    global evaluation_count
    evaluation_count = 1

    moves = game.generate_all_moves()
    if moves is None or len(moves) <= 0:
        return None

    if move_sorting:
        moves = sort_moves(game, moves, best_to_worst=True)
    
    bestEval = -1000000.0
    bestMove = None

    for move in moves:
        game_copy = game.copy()
        game_copy.move(move)
        eval = -evaluate_move_sequential(game_copy, search_depth - 1, -1000000.0, -bestEval, alpha_beta_prunning, move_sorting)
        del game_copy
        if eval > bestEval:
            bestMove = move
            bestEval = eval

    return bestMove

# Parallel implementation 1 - Parallel evaluation of moves at search_depth of 0

@cuda.jit
def evaluate_move_parallel_1_kernel(boards, n_boards, scores, perspective):
    id = cuda.grid(1)
    s_evaluation = cuda.shared.array(shape=0, dtype=np.float32)

    board_id = id // 64 # Board ID
    piece_id = id % 64  # Piece position on the board

    if id < n_boards * 64:
        # Evaluate each piece on the boards

        piece = boards[id]
        side  = piece & 0x08
        type  = piece & 0x07

        if side == 0:
            perspective = -perspective

        piece_value = 0.0
        if type == 2:
            piece_value = 1.0
        elif type == 3:
            piece_value = 3.0
        elif type == 4:
            piece_value = 3.0
        elif type == 5:
            piece_value = 5.0
        elif type == 6:
            piece_value = 9.0

        s_evaluation[cuda.threadIdx.x] = piece_value * perspective
    else:
        s_evaluation[cuda.threadIdx.x] = 0.0
    cuda.syncthreads()

    # Reduce to sum for each board
    stride = 32
    while stride > 0:
        if id < n_boards * 64 and \
        piece_id < stride:
            s_evaluation[cuda.threadIdx.x] += s_evaluation[cuda.threadIdx.x + stride]
        stride = stride // 2
        cuda.syncthreads()

    stride = (cuda.blockDim.x // 64) // 2
    while stride > 0:
        if piece_id == 0 and \
        id + stride * 64 < n_boards * 64 and \
        cuda.threadIdx.x + stride * 64 < cuda.blockDim.x and \
        s_evaluation[cuda.threadIdx.x] > s_evaluation[cuda.threadIdx.x + stride * 64]:
            s_evaluation[cuda.threadIdx.x] = s_evaluation[cuda.threadIdx.x + stride * 64]
        stride = stride // 2
        cuda.syncthreads()

    # Save result
    if cuda.threadIdx.x == 0:
        scores[cuda.blockIdx.x] = s_evaluation[0]

def evaluate_move_parallel_1(game: ChessGame, search_depth: int, alpha: float, beta: float, alpha_beta_prunning: bool = False, move_sorting: bool = False) -> float:
    global evaluation_count
    evaluation_count += 1

    if game.get_halfmove() >= 100:
        return 0.0

    moves = game.generate_all_moves()
    if moves is None or len(moves) <= 0:
        if game.is_current_side_in_check():
            return -1000000.0
        return 0.0

    if move_sorting:
        moves = sort_moves(game, moves, best_to_worst=True)

    if search_depth > 1:
        for move in moves:
            game_copy = game.copy()
            game_copy.move(move)
            eval = -evaluate_move_parallel_1(game_copy, search_depth - 1, -beta, -alpha, alpha_beta_prunning, move_sorting)
            del game_copy

            if alpha_beta_prunning and eval >= beta:
                return beta

            if eval > alpha:
                alpha = eval
        return alpha
    else:
        n_boards = len(moves)
        boards = []

        evaluation_count += n_boards

        for move in moves:
            game_copy = game.copy()
            game_copy.move(move)
            boards += game_copy.get_board()
            del game_copy

        gpu = cuda.get_current_device()
        block_size = gpu.MAX_THREADS_PER_BLOCK
        grid_size  = (n_boards * 64 - 1) // block_size + 1

        d_boards = cuda.to_device(np.array(boards, dtype=np.int32))
        d_scores = cuda.device_array(grid_size, dtype=np.float32)
        evaluate_move_parallel_1_kernel[grid_size, block_size, 0, block_size * 4](d_boards, n_boards, d_scores, -1.0 if game.side_to_move() else 1.0)
        scores = d_scores.copy_to_host()

        for score in scores:
            if alpha_beta_prunning and -score >= beta:
                return beta

            if -score > alpha:
                alpha = -score

        return alpha

def find_move_parallel_1(game: ChessGame, search_depth: int, alpha_beta_prunning: bool = False, move_sorting: bool = False):
    global evaluation_count
    evaluation_count = 1

    moves = game.generate_all_moves()
    if moves is None or len(moves) <= 0:
        return None

    if move_sorting:
        moves = sort_moves(game, moves, best_to_worst=True)

    bestEval = -1000000.0
    bestMove = None

    for move in moves:
        game_copy = game.copy()
        game_copy.move(move)
        eval = -evaluate_move_parallel_1(game_copy, search_depth - 1, -1000000.0, -bestEval, alpha_beta_prunning, move_sorting)
        del game_copy
        if eval > bestEval:
            bestMove = move
            bestEval = eval

    return bestMove

# Versions:
#
# - 0: Sequential minimax
# - 1: Sequential minimax with alpha - beta pruning
# - 2: Sequential minimax with alpha - beta pruning and move sorting
#
# - 3: Parallel on search_depth = 0
# - 4: Parallel on search_depth = 0 with alpha - beta pruning
# - 5: Parallel on search_depth = 0 with alpha - beta pruning and move sorting
#
#

def find_move(game: ChessGame, search_depth: int, version: int = 0):
    match version:
        case 0:
            return find_move_sequencial(game, search_depth)
        case 1:
            return find_move_sequencial(game, search_depth, alpha_beta_prunning=True)
        case 2:
            return find_move_sequencial(game, search_depth, alpha_beta_prunning=True, move_sorting=True)
        case 3: 
            return find_move_parallel_1(game, search_depth)
        case 4: 
            return find_move_parallel_1(game, search_depth, alpha_beta_prunning=True)
        case 5: 
            return find_move_parallel_1(game, search_depth, alpha_beta_prunning=True, move_sorting=True)
