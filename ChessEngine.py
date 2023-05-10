from ChessCore import *
from numba import cuda, types

# Profiling

def get_evaluation_count():
    global evaluation_count
    return evaluation_count

# Move sorting

def move_score(
    game, 
    move
):
    res = 0.0
    capture = game.get_piece(move.end_position())
    if capture != ChessPieces._E:
        res = 10.0 * ChessPieces.value(capture) - ChessPieces.value(game.get_piece(move.start_position()))
    return res

def sort_moves(
    game, 
    moves, 
    best_to_worst = True
):
    res = moves + []
    res.sort(key = lambda move: move_score(game, move), reverse = best_to_worst)
    return res

# Move encoding and decoding

def encode_move(
    move
):
    promote_to = 0x00
    if move.promote_to() is not None:
        promote_to = (move.promote_to() & 0x07) - 2

    res = 0x00

    res |= (move.start_position() & 0xFF) << 0 
    res |= (move.end_position() & 0xFF) << 8
    res |= promote_to << 16
    res |= (1 if move.en_passant_target() is not None else 0) << 19

    return res

def decode_move(
    move,
    side
):
    start_position = (move >>  0) & 0xFF
    end_position   = (move >>  8) & 0xFF
    promote_to     = (move >> 16) & 0x07
    en_passant     = (move >> 19) & 0x01 != 0

    if promote_to != 0x00:
        promote_to = (promote_to + 2) | (0x08 if side else 0x00)

    en_passant_target = end_position if en_passant else None

    return Move(start_position, end_position, en_passant_target, promote_to)

@cuda.jit(device = True, inline = True)
def encode_move_device(
    start_position, 
    end_position, 
    promote_to, 
    en_passant
):
    if promote_to > 0:
        promote_to -= 2

    res = 0x00

    res |= (start_position & 0xFF) << 0 
    res |= (end_position   & 0xFF) << 8
    res |= promote_to << 16
    res |= (1 if en_passant else 0) << 19

    return res

@cuda.jit(device = True, inline = True)
def decode_move_device(
    move,
    side
):
    start_position = (move >>  0) & 0xFF
    end_position   = (move >>  8) & 0xFF
    promote_to     = (move >> 16) & 0x07
    en_passant     = (move >> 19) & 0x01 != 0

    if promote_to != 0x00:
        promote_to = (promote_to + 2) | (0x08 if side else 0x00)
        
    return start_position, end_position, promote_to, en_passant

# Mutex locking/unlocking

@cuda.jit(device = True)
def lock(mutex):
    while cuda.atomic.compare_and_swap(mutex, 0, 1) != 0:
        pass
    cuda.threadfence()

@cuda.jit(device = True)
def unlock(mutex):
    cuda.threadfence()
    cuda.atomic.exch(mutex, 0, 0)

# Piece value function

piece_base_values         = np.array(ChessPieces.PIECE_TYPE_VALUES        , dtype=float)
piece_square_table_king   = np.array(ChessPieces.PIECE_SQUARE_TABLE_KING  , dtype=float)
piece_square_table_pawn   = np.array(ChessPieces.PIECE_SQUARE_TABLE_PAWN  , dtype=float)
piece_square_table_knight = np.array(ChessPieces.PIECE_SQUARE_TABLE_KNIGHT, dtype=float)
piece_square_table_bishop = np.array(ChessPieces.PIECE_SQUARE_TABLE_BISHOP, dtype=float)
piece_square_table_rook   = np.array(ChessPieces.PIECE_SQUARE_TABLE_ROOK  , dtype=float)
piece_square_table_queen  = np.array(ChessPieces.PIECE_SQUARE_TABLE_QUEEN , dtype=float)

@cuda.jit(device = True)
def piece_value_device(
    piece,
    position, 
    piece_base_values, 
    piece_square_table_king, 
    piece_square_table_pawn, 
    piece_square_table_knight, 
    piece_square_table_bishop, 
    piece_square_table_rook, 
    piece_square_table_queen
):
    piece_type = piece & 0x07
    piece_side = piece & 0x08

    base_value = piece_base_values[piece_type]

    positional_value = 0.0
    if position >= 0:
        r = position // 8
        c = position % 8
        if piece_side == 0:
            r = 7 - r
        position = r * 8 + c
        if piece_type == 0x01:
            positional_value = piece_square_table_king[position]
        elif piece_type == 0x02:
            positional_value = piece_square_table_pawn[position]
        elif piece_type == 0x03:
            positional_value = piece_square_table_knight[position]
        elif piece_type == 0x04:
            positional_value = piece_square_table_bishop[position]
        elif piece_type == 0x05:
            positional_value = piece_square_table_rook[position]
        elif piece_type == 0x06:
            positional_value = piece_square_table_queen[position]

    return (base_value + positional_value) * (-1.0 if piece_side == 0 else 1.0)

# Move generation function

@cuda.jit(device = True, inline = True)
def is_in_bound_device(
    r: int, 
    c: int
):
    return r >= 0 and r < 8 and c >= 0 and c < 8

@cuda.jit(device = True, inline = True)
def push_back_device(
    list, 
    list_count, 
    value
):
    list[list_count] = value
    return list_count + 1

@cuda.jit(device = True)
def get_king_position_device(
    board, 
    side
):
    king = (0x08 | 0x01) if side else 0x01
    for i in range(64):
        if board[i] == king:
            return i
    return -1

@cuda.jit(device = True)
def generate_moves_device(
    board, 
    side, 
    castle_WK, 
    castle_WQ, 
    castle_BK, 
    castle_BQ, 
    en_passant_target, 
    king_position, 
    position, 
    required_destinations, 
    pinned_pieces, 
    attacked_positions, 
    count, 
    out_moves
):
    piece = board[position]
    piece_side = (piece & 0x08 != 0)
    piece_type = piece & 0x07

    if side != piece_side or piece_type == 0:
        return count

    king_r = king_position // 8
    king_c = king_position % 8
    r      = position // 8
    c      = position % 8

    unpinned = (pinned_pieces & (1 << position)) == 0

    if piece_type == 0x05 or piece_type == 0x06: # Rook or queen
        i = 1
        while is_in_bound_device(r + i, c):
            position2 = (r + i) * 8 + c
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and (unpinned or king_c == c)
            if piece2 & 0x07 != 0:
                if side != (piece2 & 0x08 != 0):
                    if valid:
                        count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x00, False))
                break
            if valid:
                count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x00, False))
            i += 1
        i = 1
        while is_in_bound_device(r - i, c):
            position2 = (r - i) * 8 + c
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and (unpinned or king_c == c)
            if piece2 & 0x07 != 0:
                if side != (piece2 & 0x08 != 0):
                    if valid:
                        count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x00, False))
                break
            if valid:
                count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x00, False))
            i += 1
        i = 1
        while is_in_bound_device(r, c + i):
            position2 = r * 8 + c + i
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and (unpinned or king_r == r)
            if piece2 & 0x07 != 0:
                if side != (piece2 & 0x08 != 0):
                    if valid:
                        count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x00, False))
                break
            if valid:
                count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x00, False))
            i += 1
        i = 1
        while is_in_bound_device(r, c - i):
            position2 = r * 8 + c - i
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and (unpinned or king_r == r)
            if piece2 & 0x07 != 0:
                if side != (piece2 & 0x08 != 0):
                    if valid:
                        count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x00, False))
                break
            if valid:
                count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x00, False))
            i += 1

    if piece_type == 0x04 or piece_type == 0x06: # Bishop or queen
        i = 1
        while is_in_bound_device(r + i, c + i):
            position2 = (r + i) * 8 + c + i
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and (unpinned or king_r - r == king_c - c)
            if piece2 & 0x07 != 0:
                if side != (piece2 & 0x08 != 0):
                    if valid:
                        count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x00, False))
                break
            if valid:
                count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x00, False))
            i += 1
        i = 1
        while is_in_bound_device(r - i, c + i):
            position2 = (r - i) * 8 + c + i
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and (unpinned or r - king_r == king_c - c)
            if piece2 & 0x07 != 0:
                if side != (piece2 & 0x08 != 0):
                    if valid:
                        count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x00, False))
                break
            if valid:
                count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x00, False))
            i += 1
        i = 1
        while is_in_bound_device(r + i, c - i):
            position2 = (r + i) * 8 + c - i
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and (unpinned or r - king_r == king_c - c)
            if piece2 & 0x07 != 0:
                if side != (piece2 & 0x08 != 0):
                    if valid:
                        count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x00, False))
                break
            if valid:
                count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x00, False))
            i += 1
        i = 1
        while is_in_bound_device(r - i, c - i):
            position2 = (r - i) * 8 + c - i
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and (unpinned or king_r - r == king_c - c)
            if piece2 & 0x07 != 0:
                if side != (piece2 & 0x08 != 0):
                    if valid:
                        count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x00, False))
                break
            if valid:
                count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x00, False))
            i += 1

    if piece_type == 0x03: # Knight
        if is_in_bound_device(r + 2, c + 1):
            position2 = (r + 2) * 8 + c + 1
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and unpinned
            if valid and (piece2 & 0x07 == 0 or side != (piece2 & 0x08 != 0)):
                count = push_back_device(out_moves ,count, encode_move_device(position, position2, 0x00, False))
        if is_in_bound_device(r - 2, c + 1):
            position2 = (r - 2) * 8 + c + 1
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and unpinned
            if valid and (piece2 & 0x07 == 0 or side != (piece2 & 0x08 != 0)):
                count = push_back_device(out_moves ,count, encode_move_device(position, position2, 0x00, False))
        if is_in_bound_device(r + 2, c - 1):
            position2 = (r + 2) * 8 + c - 1
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and unpinned
            if valid and (piece2 & 0x07 == 0 or side != (piece2 & 0x08 != 0)):
                count = push_back_device(out_moves ,count, encode_move_device(position, position2, 0x00, False))
        if is_in_bound_device(r - 2, c - 1):
            position2 = (r - 2) * 8 + c - 1
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and unpinned
            if valid and (piece2 & 0x07 == 0 or side != (piece2 & 0x08 != 0)):
                count = push_back_device(out_moves ,count, encode_move_device(position, position2, 0x00, False))
        if is_in_bound_device(r + 1, c + 2):
            position2 = (r + 1) * 8 + c + 2
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and unpinned
            if valid and (piece2 & 0x07 == 0 or side != (piece2 & 0x08 != 0)):
                count = push_back_device(out_moves ,count, encode_move_device(position, position2, 0x00, False))
        if is_in_bound_device(r - 1, c + 2):
            position2 = (r - 1) * 8 + c + 2
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and unpinned
            if valid and (piece2 & 0x07 == 0 or side != (piece2 & 0x08 != 0)):
                count = push_back_device(out_moves ,count, encode_move_device(position, position2, 0x00, False))
        if is_in_bound_device(r + 1, c - 2):
            position2 = (r + 1) * 8 + c - 2
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and unpinned
            if valid and (piece2 & 0x07 == 0 or side != (piece2 & 0x08 != 0)):
                count = push_back_device(out_moves ,count, encode_move_device(position, position2, 0x00, False))
        if is_in_bound_device(r - 1, c - 2):
            position2 = (r - 1) * 8 + c - 2
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and unpinned
            if valid and (piece2 & 0x07 == 0 or side != (piece2 & 0x08 != 0)):
                count = push_back_device(out_moves ,count, encode_move_device(position, position2, 0x00, False))

    if piece_type == 0x02: # Pawn
        first_r = 1 if side else 6
        last_r  = 7 if side else 0
        pawn_dr = 1 if side else -1

        if is_in_bound_device(r + pawn_dr, c):
            position2 = (r + pawn_dr) * 8 + c
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and (unpinned or king_c == c)
            if piece2 & 0x07 == 0:
                if valid:
                    if r + pawn_dr == last_r:
                        count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x03, False))
                        count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x04, False))
                        count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x05, False))
                        count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x06, False))
                    else:
                        count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x00, False))
                if r == first_r:
                    position3 = (r + pawn_dr * 2) * 8 + c
                    piece3 = board[position3]
                    valid = required_destinations & (1 << position3) != 0 and (unpinned or king_c == c)
                    if valid and piece3 & 0x07 == 0:
                        count = push_back_device(out_moves ,count, encode_move_device(position, position3, 0x00, False))
        if is_in_bound_device(r + pawn_dr, c - 1):
            position2 = (r + pawn_dr) * 8 + c - 1
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and (unpinned or r - king_r == pawn_dr * (king_c - c))
            if valid:
                if piece2 & 0x07 != 0 and side != (piece2 & 0x08 != 0):
                    if r + pawn_dr == last_r:
                        count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x03, False))
                        count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x04, False))
                        count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x05, False))
                        count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x06, False))
                    else:
                        count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x00, False))
                if position2 == en_passant_target:
                    left, right = 0, 0
                    i = c - 2
                    while i >= 0:
                        piece_left = board[r * 8 + i]
                        if piece_left & 0x07 != 0:
                            if piece_left == (0x08 if side else 0x00) | 0x01:
                                left = 1
                            elif piece_left == (0x00 if side else 0x08) | 0x05 or piece_left == (0x00 if side else 0x08) | 0x06:
                                left = 2
                            break
                        i -= 1
                    i = c + 1
                    while i < 8:
                        piece_right = board[r * 8 + i]
                        if piece_right & 0x07 != 0:
                            if piece_right == (0x08 if side else 0x00) | 0x01:
                                riight = 1
                            elif piece_right == (0x00 if side else 0x08) | 0x05 or piece_right == (0x00 if side else 0x08) | 0x06:
                                right = 2
                            break
                        i += 1
                    if left + right != 3:
                        count = push_back_device(out_moves ,count, encode_move_device(position, position2, 0x00, True))
        if is_in_bound_device(r + pawn_dr, c + 1):
            position2 = (r + pawn_dr) * 8 + c + 1
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and (unpinned or king_r - r == pawn_dr * (king_c - c))
            if valid:
                if piece2 & 0x07 != 0 and side != (piece2 & 0x08 != 0):
                    if r + pawn_dr == last_r:
                        count = push_back_device(out_moves ,count, encode_move_device(position, position2, 0x03, False))
                        count = push_back_device(out_moves ,count, encode_move_device(position, position2, 0x04, False))
                        count = push_back_device(out_moves ,count, encode_move_device(position, position2, 0x05, False))
                        count = push_back_device(out_moves ,count, encode_move_device(position, position2, 0x06, False))
                    else:
                        count = push_back_device(out_moves ,count, encode_move_device(position, position2, 0x00, False))
                if position2 == en_passant_target:
                    left, right = 0, 0
                    i = c - 1
                    while i >= 0:
                        piece_left = board[r * 8 + i]
                        if piece_left & 0x07 != 0:
                            if piece_left == (0x08 if side else 0x00) | 0x01:
                                left = 1
                            elif piece_left == (0x00 if side else 0x08) | 0x05 or piece_left == (0x00 if side else 0x08) | 0x06:
                                left = 2
                            break
                        i -= 1
                    i = c + 2
                    while i < 8:
                        piece_right = board[r * 8 + i]
                        if piece_right & 0x07 != 0:
                            if piece_right == (0x08 if side else 0x00) | 0x01:
                                riight = 1
                            elif piece_right == (0x00 if side else 0x08) | 0x05 or piece_right == (0x00 if side else 0x08) | 0x06:
                                right = 2
                            break
                        i += 1
                    if left + right != 3:
                        count = push_back_device(out_moves ,count, encode_move_device(position, position2, 0x00, True))

    if piece_type == 0x01: # King
        if is_in_bound_device(r + 1, c + 1):
            position2 = (r + 1) * 8 + c + 1
            piece2 = board[position2]
            valid = attacked_positions & (1 << position2) == 0
            if valid and (side != (piece2 & 0x08 != 0) or piece2 & 0x07 == 0):
                count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x00, False))
        if is_in_bound_device(r + 1, c):
            position2 = (r + 1) * 8 + c
            piece2 = board[position2]
            valid = attacked_positions & (1 << position2) == 0
            if valid and (side != (piece2 & 0x08 != 0) or piece2 & 0x07 == 0):
                count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x00, False))
        if is_in_bound_device(r + 1, c - 1):
            position2 = (r + 1) * 8 + c - 1
            piece2 = board[position2]
            valid = attacked_positions & (1 << position2) == 0
            if valid and (side != (piece2 & 0x08 != 0) or piece2 & 0x07 == 0):
                count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x00, False))
        if is_in_bound_device(r, c + 1):
            position2 = r * 8 + c + 1
            piece2 = board[position2]
            valid = attacked_positions & (1 << position2) == 0
            if valid and (side != (piece2 & 0x08 != 0) or piece2 & 0x07 == 0):
                count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x00, False))
        if is_in_bound_device(r, c - 1):
            position2 = r * 8 + c - 1
            piece2 = board[position2]
            valid = attacked_positions & (1 << position2) == 0
            if valid and (side != (piece2 & 0x08 != 0) or piece2 & 0x07 == 0):
                count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x00, False))
        if is_in_bound_device(r - 1, c + 1):
            position2 = (r - 1) * 8 + c + 1
            piece2 = board[position2]
            valid = attacked_positions & (1 << position2) == 0
            if valid and (side != (piece2 & 0x08 != 0) or piece2 & 0x07 == 0):
                count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x00, False))
        if is_in_bound_device(r - 1, c):
            position2 = (r - 1) * 8 + c
            piece2 = board[position2]
            valid = attacked_positions & (1 << position2) == 0
            if valid and (side != (piece2 & 0x08 != 0) or piece2 & 0x07 == 0):
                count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x00, False))
        if is_in_bound_device(r - 1, c - 1):
            position2 = (r - 1) * 8 + c - 1
            piece2 = board[position2]
            valid = attacked_positions & (1 << position2) == 0
            if valid and (side != (piece2 & 0x08 != 0) or piece2 & 0x07 == 0):
                count = push_back_device(out_moves, count, encode_move_device(position, position2, 0x00, False))

        if side:
            if castle_WK and (attacked_positions & (0x07 << 4)) == 0:
                if (board[5] & 0x07) == 0 and (board[6] & 0x07) == 0:
                    count = push_back_device(out_moves, count, encode_move_device(4, 6, 0x00, False))
            if castle_WQ and (attacked_positions & (0x07 << 2)) == 0:
                if (board[1] & 0x07) == 0 and (board[2] & 0x07) == 0 and (board[3] & 0x07) == 0:
                    count = push_back_device(out_moves, count, encode_move_device(4, 2, 0x00, False))
        else:
            if castle_BK and (attacked_positions & (0x07 << 60)) == 0:
                if (board[61] & 0x07) == 0 and (board[62] & 0x07) == 0:
                    count = push_back_device(out_moves, count, encode_move_device(60, 62, 0x00, False))
            if castle_BQ and (attacked_positions & (0x07 << 58)) == 0:
                if (board[57] & 0x07) == 0 and (board[58] & 0x07) == 0 and (board[59] & 0x07) == 0:
                    count = push_back_device(out_moves, count, encode_move_device(60, 58, 0x00, False))

    return count

@cuda.jit(device = True)
def generate_all_moves_device(
    board, 
    side, 
    castle_WK, 
    castle_WQ, 
    castle_BK, 
    castle_BQ, 
    en_passant_target, 
    halfmoves, 
    out_moves
):
    if halfmoves >= 100:
        return 0, types.int64(0x0000000000000000)

    required_destinations = types.int64(0xffffffffffffffff) # 64-bit int
    pinned_pieces         = types.int64(0x0000000000000000) # 64-bit int
    attacked_positions    = types.int64(0x0000000000000000) # 64-bit int

    king_position = get_king_position_device(board, side)
    if king_position >= 0:
        r = king_position // 8
        c = king_position % 8

        # Check by pawn
        if side:
            if is_in_bound_device(r + 1, c - 1):
                if board[(r + 1) * 8 + c - 1] == 0x02:
                    required_destinations &= 1 << ((r + 1) * 8 + c - 1)
            if is_in_bound_device(r + 1, c + 1):
                if board[(r + 1) * 8 + c + 1] == 0x02:
                    required_destinations &= 1 << ((r + 1) * 8 + c + 1)
        else:
            if is_in_bound_device(r - 1, c - 1):
                if board[(r - 1) * 8 + c - 1] == 0x08 | 0x02:
                    required_destinations &= 1 << ((r - 1) * 8 + c - 1)
            if is_in_bound_device(r - 1, c + 1):
                if board[(r - 1) * 8 + c + 1] == 0x08 | 0x02:
                    required_destinations &= 1 << ((r - 1) * 8 + c + 1)

        # Check by knight
        knight = (0x00 if side else 0x08) | 0x03
        if is_in_bound_device(r - 2, c - 1):
            if board[(r - 2) * 8 + c - 1] == knight:
                required_destinations &= 1 << ((r - 2) * 8 + c - 1)
        if is_in_bound_device(r - 2, c + 1):
            if board[(r - 2) * 8 + c + 1] == knight:
                required_destinations &= 1 << ((r - 2) * 8 + c + 1)
        if is_in_bound_device(r + 2, c - 1):
            if board[(r + 2) * 8 + c - 1] == knight:
                required_destinations &= 1 << ((r + 2) * 8 + c - 1)
        if is_in_bound_device(r + 2, c + 1):
            if board[(r + 2) * 8 + c + 1] == knight:
                required_destinations &= 1 << ((r + 2) * 8 + c + 1)
        if is_in_bound_device(r - 1, c - 2):
            if board[(r - 1) * 8 + c - 2] == knight:
                required_destinations &= 1 << ((r - 1) * 8 + c - 2)
        if is_in_bound_device(r - 1, c + 2):
            if board[(r - 1) * 8 + c + 2] == knight:
                required_destinations &= 1 << ((r - 1) * 8 + c + 2)
        if is_in_bound_device(r + 1, c - 2):
            if board[(r + 1) * 8 + c - 2] == knight:
                required_destinations &= 1 << ((r + 1) * 8 + c - 2)
        if is_in_bound_device(r + 1, c + 2):
            if board[(r + 1) * 8 + c + 2] == knight:
                required_destinations &= 1 << ((r + 1) * 8 + c + 2)

        # Check by sliding pieces
        bishop = (0x00 if side else 0x08) | 0x04
        rook   = (0x00 if side else 0x08) | 0x05
        queen  = (0x00 if side else 0x08) | 0x06
        i = 1
        rd = types.int64(0x0000000000000000)
        pinned = -1
        found = False
        while is_in_bound_device(r + i, c):
            piece = board[(r + i) * 8 + c]
            empty = piece & 0x07 == 0
            if pinned >= 0:
                if piece == rook or piece == queen:
                    pinned_pieces |= 1 << pinned
                    break
                if not empty:
                    break
            else:
                if piece == rook or piece == queen:
                    rd |= 1 << (r + i) * 8 + c
                    found = True
                    break
                elif not empty:
                    pinned = (r + i) * 8 + c
            rd |= 1 << ((r + i) * 8 + c)
            i += 1
        if found:
            required_destinations &= rd
        i = 1
        rd = types.int64(0x0000000000000000)
        pinned = -1
        found = False
        while is_in_bound_device(r - i, c):
            piece = board[(r - i) * 8 + c]
            empty = piece & 0x07 == 0
            if pinned >= 0:
                if piece == rook or piece == queen:
                    pinned_pieces |= 1 << pinned
                    break
                if not empty:
                    break
            else:
                if piece == rook or piece == queen:
                    rd |= 1 << ((r - i) * 8 + c)
                    found = True
                    break
                elif not empty:
                    pinned = (r - i) * 8 + c
            rd |= 1 << (r - i) * 8 + c
            i += 1
        if found:
            required_destinations &= rd
        i = 1
        rd = types.int64(0x0000000000000000)
        pinned = -1
        found = False
        while is_in_bound_device(r, c + i):
            piece = board[r * 8 + c + i]
            empty = piece & 0x07 == 0
            if pinned >= 0:
                if piece == rook or piece == queen:
                    pinned_pieces |= 1 << pinned
                    break
                if not empty:
                    break
            else:
                if piece == rook or piece == queen:
                    rd |= 1 << (r * 8 + c + i)
                    found = True
                    break
                elif not empty:
                    pinned = r * 8 + c + i
            rd |= 1 << r * 8 + c + i
            i += 1
        if found:
            required_destinations &= rd
        i = 1
        rd = types.int64(0x0000000000000000)
        pinned = -1
        found = False
        while is_in_bound_device(r, c - i):
            piece = board[r * 8 + c - i]
            empty = piece & 0x07 == 0
            if pinned >= 0:
                if piece == rook or piece == queen:
                    pinned_pieces |= 1 << pinned
                    break
                if not empty:
                    break
            else:
                if piece == rook or piece == queen:
                    rd |= 1 << (r * 8 + c - i)
                    found = True
                    break
                elif not empty:
                    pinned = r * 8 + c - i
            rd |= 1 << r * 8 + c - i
            i += 1
        if found:
            required_destinations &= rd
        i = 1
        rd = types.int64(0x0000000000000000)
        pinned = -1
        found = False
        while is_in_bound_device(r + i, c + i):
            piece = board[(r + i) * 8 + c + i]
            empty = piece & 0x07 == 0
            if pinned >= 0:
                if piece == bishop or piece == queen:
                    pinned_pieces |= 1 << pinned
                    break
                if not empty:
                    break
            else:
                if piece == bishop or piece == queen:
                    rd |= 1 << ((r + i) * 8 + c + i)
                    found = True
                    break
                elif not empty:
                    pinned = (r + i) * 8 + c + i
            rd |= 1 << (r + i) * 8 + c + i
            i += 1
        if found:
            required_destinations &= rd
        i = 1
        rd = types.int64(0x0000000000000000)
        pinned = -1
        found = False
        while is_in_bound_device(r - i, c + i):
            piece = board[(r - i) * 8 + c + i]
            empty = piece & 0x07 == 0
            if pinned >= 0:
                if piece == bishop or piece == queen:
                    pinned_pieces |= 1 << pinned
                    break
                if not empty:
                    break
            else:
                if piece == bishop or piece == queen:
                    rd |= 1 << ((r - i) * 8 + c + i)
                    found = True
                    break
                elif not empty:
                    pinned = (r - i) * 8 + c + i
            rd |= 1 << (r - i) * 8 + c + i
            i += 1
        if found:
            required_destinations &= rd
        i = 1
        rd = types.int64(0x0000000000000000)
        pinned = -1
        found = False
        while is_in_bound_device(r + i, c - i):
            piece = board[(r + i) * 8 + c - i]
            empty = piece & 0x07 == 0
            if pinned >= 0:
                if piece == bishop or piece == queen:
                    pinned_pieces |= 1 << pinned
                    break
                if not empty:
                    break
            else:
                if piece == bishop or piece == queen:
                    rd |= 1 << ((r + i) * 8 + c - i)
                    found = True
                    break
                elif not empty:
                    pinned = (r + i) * 8 + c - i
            rd |= 1 << (r + i) * 8 + c - i
            i += 1
        if found:
            required_destinations &= rd
        i = 1
        rd = types.int64(0x0000000000000000)
        pinned = -1
        found = False
        while is_in_bound_device(r - i, c - i):
            piece = board[(r - i) * 8 + c - i]
            empty = piece & 0x07 == 0
            if pinned >= 0:
                if piece == bishop or piece == queen:
                    pinned_pieces |= 1 << pinned
                    break
                if not empty:
                    break
            else:
                if piece == bishop or piece == queen:
                    rd |= 1 << ((r - i) * 8 + c - i)
                    found = True
                    break
                elif not empty:
                    pinned = (r - i) * 8 + c - i
            rd |= 1 << (r - i) * 8 + c - i
            i += 1
        if found:
            required_destinations &= rd

    for i in range(64):
        piece = board[i]
        piece_side = piece & 0x08 != 0
        piece_type = piece & 0x07
        if side != piece_side and piece_type != 0:
            r = i // 8
            c = i % 8

            if piece_type == 0x05 or piece_type == 0x06:
                i = 1
                while is_in_bound_device(r + i, c):
                    position2 = (r + i) * 8 + c
                    attacked_positions |= 1 << position2
                    if position2 != king_position and board[position2] & 0x07 != 0:
                        break
                    i += 1
                i = 1
                while is_in_bound_device(r - i, c):
                    position2 = (r - i) * 8 + c
                    attacked_positions |= 1 << position2
                    if position2 != king_position and board[position2] & 0x07 != 0:
                        break
                    i += 1
                i = 1
                while is_in_bound_device(r, c + i):
                    position2 = r * 8 + c + i
                    attacked_positions |= 1 << position2
                    if position2 != king_position and board[position2] & 0x07 != 0:
                        break
                    i += 1
                i = 1
                while is_in_bound_device(r, c - i):
                    position2 = r * 8 + c - i
                    attacked_positions |= 1 << position2
                    if position2 != king_position and board[position2] & 0x07 != 0:
                        break
                    i += 1

            if piece_type == 0x04 or piece_type == 0x06:
                i = 1
                while is_in_bound_device(r + i, c + i):
                    position2 = (r + i) * 8 + c + i
                    attacked_positions |= 1 << position2
                    if position2 != king_position and board[position2] & 0x07 != 0:
                        break
                    i += 1
                i = 1
                while is_in_bound_device(r - i, c + i):
                    position2 = (r - i) * 8 + c + i
                    attacked_positions |= 1 << position2
                    if position2 != king_position and board[position2] & 0x07 != 0:
                        break
                    i += 1
                i = 1
                while is_in_bound_device(r + i, c - i):
                    position2 = (r + i) * 8 + c - i
                    attacked_positions |= 1 << position2
                    if position2 != king_position and board[position2] & 0x07 != 0:
                        break
                    i += 1
                i = 1
                while is_in_bound_device(r - i, c - i):
                    position2 = (r - i) * 8 + c - i
                    attacked_positions |= 1 << position2
                    if position2 != king_position and board[position2] & 0x07 != 0:
                        break
                    i += 1
            
            if piece_type == 0x03:
                if is_in_bound_device(r + 2, c + 1):
                    attacked_positions |= 1 << ((r + 2) * 8 + c + 1)
                if is_in_bound_device(r + 2, c - 1):
                    attacked_positions |= 1 << ((r + 2) * 8 + c - 1)
                if is_in_bound_device(r - 2, c + 1):
                    attacked_positions |= 1 << ((r - 2) * 8 + c + 1)
                if is_in_bound_device(r - 2, c - 1):
                    attacked_positions |= 1 << ((r - 2) * 8 + c - 1)
                if is_in_bound_device(r + 1, c + 2):
                    attacked_positions |= 1 << ((r + 1) * 8 + c + 2)
                if is_in_bound_device(r + 1, c - 2):
                    attacked_positions |= 1 << ((r + 1) * 8 + c - 2)
                if is_in_bound_device(r - 1, c + 2):
                    attacked_positions |= 1 << ((r - 1) * 8 + c + 2)
                if is_in_bound_device(r - 1, c - 2):
                    attacked_positions |= 1 << ((r - 1) * 8 + c - 2)

            if piece_type == 0x02:
                pawn_dr = 1 if piece_side else -1
                if is_in_bound_device(r + pawn_dr, c - 1):
                    attacked_positions |= 1 << ((r + pawn_dr) * 8 + c - 1)
                if is_in_bound_device(r + pawn_dr, c + 1):
                    attacked_positions |= 1 << ((r + pawn_dr) * 8 + c + 1)

            if piece_type == 0x01:
                if is_in_bound_device(r + 1, c + 1):
                    attacked_positions |= 1 << ((r + 1) * 8 + c + 1)
                if is_in_bound_device(r, c + 1):
                    attacked_positions |= 1 << (r * 8 + c + 1)
                if is_in_bound_device(r - 1, c + 1):
                    attacked_positions |= 1 << ((r - 1) * 8 + c + 1)
                if is_in_bound_device(r + 1, c):
                    attacked_positions |= 1 << ((r + 1) * 8 + c)
                if is_in_bound_device(r - 1, c):
                    attacked_positions |= 1 << ((r - 1) * 8 + c)
                if is_in_bound_device(r + 1, c - 1):
                    attacked_positions |= 1 << ((r + 1) * 8 + c - 1)
                if is_in_bound_device(r, c - 1):
                    attacked_positions |= 1 << (r * 8 + c - 1)
                if is_in_bound_device(r - 1, c - 1):
                    attacked_positions |= 1 << ((r - 1) * 8 + c - 1)

    count = 0
    for i in range(64):
        count = generate_moves_device(board, side, castle_WK, castle_WQ, castle_BK, castle_BQ, en_passant_target, king_position, i, required_destinations, pinned_pieces, attacked_positions, count, out_moves)
    return count, attacked_positions

@cuda.jit(device = True)
def generate_all_moves_sorted_device(
    board, 
    side, 
    castle_WK, 
    castle_WQ, 
    castle_BK, 
    castle_BQ, 
    en_passant_target, 
    halfmoves, 
    out_moves,
    move_scores,
    piece_base_values
):
    n_moves, attacked_positions = generate_all_moves_device(board, side, castle_WK, castle_WQ, castle_BK, castle_BQ, en_passant_target, halfmoves, out_moves)
    
    for i in range(n_moves):
        start_position, end_position, promote_to, en_passant = decode_move_device(out_moves[i], side)
        piece1 = board[start_position]
        piece2 = board[end_position]

        score = 0.0
        if (piece2 & 0x07) != 0x00:
            score = 10.0 * piece_base_values[piece2 & 0x07] - piece_base_values[piece1 & 0x07]

        move_scores[i] = score

    for i in range(n_moves):
        max_idx = i
        for j in range(i + 1, n_moves):
            if move_scores[j] > move_scores[max_idx]:
                max_idx = j
        if max_idx != i:
            temp = move_scores[i]
            move_scores[i] = move_scores[max_idx]
            move_scores[max_idx] = temp

            temp = out_moves[i]
            out_moves[i] = out_moves[max_idx]
            out_moves[max_idx] = temp

    return n_moves, attacked_positions

# Make a move on the GPU

@cuda.jit(device = True)
def move_device(
    board,
    side, 
    move,
    castle_WK, 
    castle_WQ, 
    castle_BK, 
    castle_BQ, 
    en_passant_target,
    halfmove
):
    captured_piece = 0x00

    start_position, end_position, promote_to, en_passant = decode_move_device(move, side)

    r1 = start_position // 8
    c1 = start_position % 8
    r2 = end_position // 8
    c2 = end_position % 8

    piece1 = board[start_position]
    piece2 = board[end_position]

    captured_piece = piece2

    board[start_position] = 0x00
    board[end_position]   = piece1

    if promote_to != 0x00:
        board[end_position] = promote_to

    if en_passant:
        re = r2 + (-1 if side else 1)
        ce = c2
        captured_piece = board[re * 8 + ce]
        board[re * 8 + ce] = 0x00

    if (piece1 & 0x07) == 0x01:
        dc = c2 - c1
        if dc == 2:
            board[start_position + 3] = 0x00
            board[start_position + 1] = (0x08 if side else 0x00) | 0x05
        elif dc == -2:
            board[start_position - 4] = 0x00
            board[start_position - 1] = (0x08 if side else 0x00) | 0x05

        if side:
            castle_WK = False
            castle_WQ = False
        else:
            castle_BK = False
            castle_BQ = False
    elif (piece1 & 0x07) == 0x05:
        if side:
            if c1 == 7:
                castle_WK = False
            elif c1 == 0:
                castle_WQ = False
        else:
            if c1 == 7:
                castle_BK = False
            elif c1 == 0:
                castle_BQ = False

    if (piece2 & 0x07) == 0x05:
        if side:
            if c2 == 7:
                castle_BK = False
            elif c2 == 0:
                castle_BQ = False
        else:
            if c2 == 7:
                castle_WK = False
            elif c2 == 0:
                castle_WQ = False

    if (piece1 & 0x07) == 0x02 and c1 == c2 and (r2 - r1 == 2 or r2 - r1 == -2):
        en_passant_target = end_position + (-8 if side else 8)
    else:
        en_passant_target = -1

    pawn_advanced = (piece1 & 0x07) == 0x02
    captured      = (captured_piece & 0x07) != 0x00
    if pawn_advanced or captured:
        halfmove = 0
    else:
        halfmove += 1

    side = not side

    return side, castle_WK, castle_WQ, castle_BK, castle_BQ, en_passant_target, halfmove

# Sequential implementation

def evaluate_move_sequential(
    game, 
    search_depth, 
    alpha, 
    beta, 
    alpha_beta_prunning = False, 
    move_sorting        = False
):
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

def find_move_sequencial(
    game, 
    search_depth, 
    alpha_beta_prunning = False, 
    move_sorting        = False
):
    global evaluation_count
    evaluation_count = 1

    moves = game.generate_all_moves()
    if moves is None or len(moves) <= 0:
        return None, 0.0

    if move_sorting:
        moves = sort_moves(game, moves, best_to_worst=True)
    
    best_eval = -1000000.0
    best_move = None

    for move in moves:
        game_copy = game.copy()
        game_copy.move(move)
        eval = -evaluate_move_sequential(game_copy, search_depth - 1, -1000000.0, -best_eval, alpha_beta_prunning, move_sorting)
        del game_copy
        if eval > best_eval:
            best_move = move
            best_eval = eval

    return best_move, best_eval

# Parallel implementation 1

@cuda.jit
def evaluate_move_parallel_1_kernel(
    boards, 
    n_boards, 
    scores, 
    perspective
):
    c_piece_base_values         = cuda.const.array_like(piece_base_values)
    c_piece_square_table_king   = cuda.const.array_like(piece_square_table_king)
    c_piece_square_table_pawn   = cuda.const.array_like(piece_square_table_pawn)
    c_piece_square_table_knight = cuda.const.array_like(piece_square_table_knight)
    c_piece_square_table_bishop = cuda.const.array_like(piece_square_table_bishop)
    c_piece_square_table_rook   = cuda.const.array_like(piece_square_table_rook)
    c_piece_square_table_queen  = cuda.const.array_like(piece_square_table_queen)

    id = cuda.grid(1)
    s_evaluation = cuda.shared.array(shape = 0, dtype = np.float32) # Dynamic shared memory, size = n_threads * size(float32)

    board_id = id // 64 # Board ID
    piece_id = id % 64  # Piece position on the board

    if id < n_boards * 64:
        # Evaluate each piece on the boards

        piece = boards[id]
        piece_value = piece_value_device(piece, piece_id, 
            c_piece_base_values, 
            c_piece_square_table_king, 
            c_piece_square_table_pawn, 
            c_piece_square_table_knight, 
            c_piece_square_table_bishop, 
            c_piece_square_table_rook, 
            c_piece_square_table_queen)

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

    # Save result
    if piece_id == 0:
        scores[board_id] = s_evaluation[cuda.threadIdx.x]

def evaluate_move_parallel_1(
    game, 
    search_depth, 
    alpha, 
    beta, 
    alpha_beta_prunning = False, 
    move_sorting        = False
):
    global evaluation_count
    evaluation_count += 1

    if game.get_halfmove() >= 100:
        return 0.0

    if search_depth == 0:
        return game.evaluate()

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

        for move in moves:
            game_copy = game.copy()
            game_copy.move(move)
            boards += game_copy.get_board()
            del game_copy

        gpu = cuda.get_current_device()
        block_size = gpu.MAX_THREADS_PER_BLOCK
        grid_size  = (n_boards * 64 - 1) // block_size + 1

        d_boards = cuda.to_device(np.array(boards, dtype=np.int8))
        d_scores = cuda.device_array(n_boards, dtype=np.float32)
        
        evaluate_move_parallel_1_kernel[grid_size, block_size, 0, block_size * 4](d_boards, n_boards, d_scores, -1.0 if game.side_to_move() else 1.0)
        evaluation_count += n_boards

        scores = d_scores.copy_to_host()
        for score in scores:
            if alpha_beta_prunning and -score >= beta:
                return beta

            if -score > alpha:
                alpha = -score

        return alpha

def find_move_parallel_1(
    game, 
    search_depth, 
    alpha_beta_prunning = False, 
    move_sorting        = False
):
    global evaluation_count
    evaluation_count = 1

    moves = game.generate_all_moves()
    if moves is None or len(moves) <= 0:
        return None, 0.0

    if move_sorting:
        moves = sort_moves(game, moves, best_to_worst=True)
    
    best_eval = -1000000.0
    best_move = None

    if search_depth > 1:
        for move in moves:
            game_copy = game.copy()
            game_copy.move(move)
            eval = -evaluate_move_parallel_1(game_copy, search_depth - 1, -1000000.0, -best_eval, alpha_beta_prunning, move_sorting)
            del game_copy
            if eval > best_eval:
                best_move = move
                best_eval = eval
    else:
        n_boards = len(moves)
        boards = []

        for move in moves:
            game_copy = game.copy()
            game_copy.move(move)
            boards += game_copy.get_board()
            del game_copy

        gpu = cuda.get_current_device()
        block_size = gpu.MAX_THREADS_PER_BLOCK
        grid_size  = (n_boards * 64 - 1) // block_size + 1

        d_boards = cuda.to_device(np.array(boards, dtype=np.int8))
        d_scores = cuda.device_array(n_boards, dtype=np.float32)

        evaluate_move_parallel_1_kernel[grid_size, block_size, 0, block_size * 4](d_boards, n_boards, d_scores, -1.0 if game.side_to_move() else 1.0)
        evaluation_count += n_boards

        scores = d_scores.copy_to_host()

        index = 0
        for score in scores:
            if -score > best_eval:
                best_move = moves[index]
                best_eval = -score
            index += 1

    return best_move, best_eval

# Parallel implementation 2

@cuda.jit
def evaluate_move_parallel_2_kernel(
    boards,
    n_boards,
    side,
    castling_rights,
    en_passant_targets,
    halfmoves,
    search_depth,
    scores,
    evaluation_count,
    alpha,
    beta,
    alpha_beta_prunning,
    move_sorting
):
    c_piece_base_values         = cuda.const.array_like(piece_base_values)
    c_piece_square_table_king   = cuda.const.array_like(piece_square_table_king)
    c_piece_square_table_pawn   = cuda.const.array_like(piece_square_table_pawn)
    c_piece_square_table_knight = cuda.const.array_like(piece_square_table_knight)
    c_piece_square_table_bishop = cuda.const.array_like(piece_square_table_bishop)
    c_piece_square_table_rook   = cuda.const.array_like(piece_square_table_rook)
    c_piece_square_table_queen  = cuda.const.array_like(piece_square_table_queen)

    board_id = cuda.blockIdx.x

    s_shared_i8  = cuda.shared.array(shape = 0, dtype = np.int8)
    s_shared_i32 = cuda.shared.array(shape = 0, dtype = np.int32)
    s_shared_f32 = cuda.shared.array(shape = 0, dtype = np.float32)

    s_boards             = s_shared_i8 [                        : 64 * (search_depth + 1)]
    s_castling_rights    = s_shared_i8 [64 * (search_depth + 1) : 65 * (search_depth + 1)]
    s_en_passant_targets = s_shared_i8 [65 * (search_depth + 1) : 66 * (search_depth + 1)]
    s_halfmoves          = s_shared_i8 [66 * (search_depth + 1) : 67 * (search_depth + 1)]
    s_status             = s_shared_i32[17 * (search_depth + 1) : 18 * (search_depth + 1)]
    s_alphas             = s_shared_f32[18 * (search_depth + 1) : 19 * (search_depth + 1)]
    s_betas              = s_shared_f32[19 * (search_depth + 1) : 20 * (search_depth + 1)]

    if cuda.threadIdx.x < 64:
        s_boards[cuda.threadIdx.x] = boards[board_id * 64 + cuda.threadIdx.x]
        if cuda.threadIdx.x == 0:
            s_castling_rights[0]    = castling_rights[board_id]
            s_en_passant_targets[0] = en_passant_targets[board_id]
            s_halfmoves[0]          = halfmoves[board_id]
            s_alphas[0]             = alpha
            s_betas[0]              = beta

            for i in range(search_depth + 1):
                s_status[i] = 1
            s_status[0] = 0
    cuda.syncthreads()

    out_moves   = cuda.local.array(256, dtype = np.int32)
    move_scores = cuda.local.array(256, dtype = np.float32)

    if cuda.threadIdx.x <= search_depth:
        _side = side
        if cuda.threadIdx.x % 2 != 0:
            _side = not side

        while True:
            while cuda.atomic.add(s_status, cuda.threadIdx.x, 0) == 1:
                continue

            if cuda.atomic.add(s_status, cuda.threadIdx.x, 0) > 1:
                break
            
            cuda.atomic.add(evaluation_count, 0, 1)
            if cuda.threadIdx.x == search_depth:
                evaluation = 0.0
                for i in range(64):
                    evaluation += piece_value_device(s_boards[cuda.threadIdx.x * 64 + i], i, 
                        c_piece_base_values,
                        c_piece_square_table_king,
                        c_piece_square_table_pawn,
                        c_piece_square_table_knight,
                        c_piece_square_table_bishop,
                        c_piece_square_table_rook,
                        c_piece_square_table_queen)
                    
                s_alphas[cuda.threadIdx.x] = evaluation * (1.0 if _side else -1.0)
            else:
                if s_halfmoves[cuda.threadIdx.x] >= 100:
                    s_alphas[cuda.threadIdx.x] = 0.0
                else:
                    castling_right = s_castling_rights[cuda.threadIdx.x]
                    castle_WK = (castling_right & 0x01 != 0)
                    castle_WQ = (castling_right & 0x02 != 0)
                    castle_BK = (castling_right & 0x04 != 0)
                    castle_BQ = (castling_right & 0x08 != 0)

                    if move_sorting:
                        n_moves, attacked_positions = generate_all_moves_sorted_device(s_boards[cuda.threadIdx.x * 64:], _side, castle_WK, castle_WQ, castle_BK, castle_BQ, s_en_passant_targets[cuda.threadIdx.x], s_halfmoves[cuda.threadIdx.x], out_moves, move_scores, c_piece_base_values)
                    else:
                        n_moves, attacked_positions = generate_all_moves_device(s_boards[cuda.threadIdx.x * 64:], _side, castle_WK, castle_WQ, castle_BK, castle_BQ, s_en_passant_targets[cuda.threadIdx.x], s_halfmoves[cuda.threadIdx.x], out_moves)

                    if n_moves <= 0:
                        king_position = get_king_position_device(s_boards[cuda.threadIdx.x * 64:], _side)
                        if (attacked_positions & (1 << king_position)) == 0:
                            s_alphas[cuda.threadIdx.x] = 0.0
                    else:
                        for move_index in range(n_moves):
                            for i in range(64):
                                s_boards[cuda.threadIdx.x * 64 + 64 + i] = s_boards[cuda.threadIdx.x * 64 + i]

                            _, _castle_WK, _castle_WQ, _castle_BK, _castle_BQ, _en_passant_target, _halfmove = move_device(
                                s_boards[cuda.threadIdx.x * 64 + 64 : cuda.threadIdx.x * 64 + 128], _side, out_moves[move_index], 
                                castle_WK, castle_WQ, castle_BK, castle_BQ, s_en_passant_targets[cuda.threadIdx.x], s_halfmoves[cuda.threadIdx.x])
                            _castling_right = 0x00
                            if _castle_WK:
                                _castling_right |= (1 << 0)
                            if _castle_WQ:
                                _castling_right |= (1 << 1)
                            if _castle_BK:
                                _castling_right |= (1 << 2)
                            if _castle_BQ:
                                _castling_right |= (1 << 3)
                            s_castling_rights[cuda.threadIdx.x + 1]    = _castling_right
                            s_en_passant_targets[cuda.threadIdx.x + 1] = _en_passant_target
                            s_halfmoves[cuda.threadIdx.x + 1]          = _halfmove
                            s_alphas[cuda.threadIdx.x + 1]             = -s_betas[cuda.threadIdx.x]
                            s_betas[cuda.threadIdx.x + 1]              = -s_alphas[cuda.threadIdx.x]

                            cuda.atomic.exch(s_status, cuda.threadIdx.x + 1, 0)
                            while cuda.atomic.add(s_status, cuda.threadIdx.x + 1, 0) == 0:
                                continue

                            if alpha_beta_prunning and -s_alphas[cuda.threadIdx.x + 1] >= s_betas[cuda.threadIdx.x]:
                                s_alphas[cuda.threadIdx.x] = s_betas[cuda.threadIdx.x]
                                break
                            
                            if -s_alphas[cuda.threadIdx.x + 1] > s_alphas[cuda.threadIdx.x]:
                                s_alphas[cuda.threadIdx.x] = -s_alphas[cuda.threadIdx.x + 1]

            if cuda.threadIdx.x == 0:
                for i in range(search_depth + 1):
                    cuda.atomic.exch(s_status, i, 2)
            else:
                cuda.atomic.exch(s_status, cuda.threadIdx.x, 1)
    cuda.syncthreads()    

    if cuda.threadIdx.x == 0:
        scores[board_id] = s_alphas[0]

def evaluate_move_parallel_2(
    game, 
    search_depth,
    alpha, 
    beta, 
    alpha_beta_prunning = False, 
    move_sorting        = False
):
    global evaluation_count
    evaluation_count += 1

    if game.get_halfmove() >= 100:
        return 0.0

    if search_depth == 0:
        return game.evaluate()

    moves = game.generate_all_moves()
    if moves is None or len(moves) <= 0:
        if game.is_current_side_in_check():
            return -1000000.0
        return 0.0

    if move_sorting:
        moves = sort_moves(game, moves, best_to_worst=True)

    first_move = moves[0]
    moves.pop(0)

    game_copy = game.copy()
    game_copy.move(first_move)
    eval = -evaluate_move_parallel_2(game_copy, search_depth - 1, -beta, -alpha, alpha_beta_prunning, move_sorting)
    del game_copy

    if alpha_beta_prunning and eval >= beta:
        return beta

    if eval > alpha:
        alpha = eval

    if len(moves) > 0:
        n_boards = len(moves)
        boards             = []
        castling_rights    = []
        en_passant_targets = []
        halfmoves          = []

        for move in moves:
            game_copy = game.copy()
            game_copy.move(move)

            boards += game_copy.get_board()
            castling_right = 0x00
            if game_copy.m_castle_WK:
                castling_right |= (1 << 0)
            if game_copy.m_castle_WQ:
                castling_right |= (1 << 1)
            if game_copy.m_castle_BK:
                castling_right |= (1 << 2)
            if game_copy.m_castle_BQ:
                castling_right |= (1 << 3)
            castling_rights.append(castling_right)
            en_passant_targets.append(-1 if game_copy.m_en_passant_target is None else game_copy.m_en_passant_target)
            halfmoves.append(game_copy.get_halfmove())
            del game_copy   

        block_size = 64
        while block_size < search_depth:
            block_size *= 2
        grid_size  = n_boards

        d_boards             = cuda.to_device(np.array(boards            , dtype = np.int8))
        d_casting_rights     = cuda.to_device(np.array(castling_rights   , dtype = np.int8))
        d_en_passant_targets = cuda.to_device(np.array(en_passant_targets, dtype = np.int8))
        d_halfmoves          = cuda.to_device(np.array(halfmoves         , dtype = np.int8))

        d_scores           = cuda.device_array(n_boards, dtype = np.float32)
        d_evaluation_count = cuda.to_device(np.zeros(1, dtype = np.int32))

        evaluate_move_parallel_2_kernel[grid_size, block_size, 0, search_depth * 80](d_boards, n_boards, not game.side_to_move(), 
            d_casting_rights, d_en_passant_targets, d_halfmoves, search_depth - 1, d_scores, d_evaluation_count, -beta, -alpha, alpha_beta_prunning, move_sorting)

        scores = d_scores.copy_to_host()
        evaluation_count += d_evaluation_count.copy_to_host()[0]

        for score in scores:
            if alpha_beta_prunning and -score >= beta:
                return beta

            if -score > alpha:
                alpha = -score

    return alpha

def find_move_parallel_2(
    game, 
    search_depth,
    alpha_beta_prunning = False, 
    move_sorting        = False
):
    global evaluation_count
    evaluation_count = 1

    moves = game.generate_all_moves()
    if moves is None or len(moves) <= 0:
        return None, 0.0

    if move_sorting:
        moves = sort_moves(game, moves, best_to_worst=True)

    first_move = moves[0]
    moves.pop(0)

    game_copy = game.copy()
    game_copy.move(first_move)
    best_eval = -evaluate_move_parallel_2(game_copy, search_depth - 1, -1000000.0, 1000000.0, alpha_beta_prunning, move_sorting)
    best_move = first_move
    del game_copy

    if len(moves) > 0:
        n_boards = len(moves)
        boards             = []
        castling_rights    = []
        en_passant_targets = []
        halfmoves          = []

        for move in moves:
            game_copy = game.copy()
            game_copy.move(move)

            boards += game_copy.get_board()
            castling_right = 0x00
            if game_copy.m_castle_WK:
                castling_right |= (1 << 0)
            if game_copy.m_castle_WQ:
                castling_right |= (1 << 1)
            if game_copy.m_castle_BK:
                castling_right |= (1 << 2)
            if game_copy.m_castle_BQ:
                castling_right |= (1 << 3)
            castling_rights.append(castling_right)
            en_passant_targets.append(-1 if game_copy.m_en_passant_target is None else game_copy.m_en_passant_target)
            halfmoves.append(game_copy.get_halfmove())
            del game_copy   

        block_size = 64
        grid_size  = n_boards

        d_boards             = cuda.to_device(np.array(boards            , dtype = np.int8))
        d_casting_rights     = cuda.to_device(np.array(castling_rights   , dtype = np.int8))
        d_en_passant_targets = cuda.to_device(np.array(en_passant_targets, dtype = np.int8))
        d_halfmoves          = cuda.to_device(np.array(halfmoves         , dtype = np.int8))

        d_scores           = cuda.device_array(n_boards, dtype = np.float32)
        d_evaluation_count = cuda.to_device(np.zeros(1, dtype = np.int32))

        evaluate_move_parallel_2_kernel[grid_size, block_size, 0, search_depth * 80](d_boards, n_boards, not game.side_to_move(), 
            d_casting_rights, d_en_passant_targets, d_halfmoves, search_depth - 1, d_scores, d_evaluation_count, -1000000.0, -best_eval, alpha_beta_prunning, move_sorting)

        scores = d_scores.copy_to_host()
        evaluation_count += d_evaluation_count.copy_to_host()[0]

        index = 0
        for score in scores:
            if -score > best_eval:
                best_eval = -score
                best_move = moves[index]
            index += 1

    return best_move, best_eval

# Parallel implementation 3

@cuda.jit
def evaluate_move_parallel_3_kernel(
    boards,
    n_boards,
    side,
    castling_rights,
    en_passant_targets,
    halfmoves,
    search_depth,
    evaluation_count,
    alpha_beta,
    out_move_index,
    mutex,
    alpha_beta_prunning,
    move_sorting
):
    c_piece_base_values         = cuda.const.array_like(piece_base_values)
    c_piece_square_table_king   = cuda.const.array_like(piece_square_table_king)
    c_piece_square_table_pawn   = cuda.const.array_like(piece_square_table_pawn)
    c_piece_square_table_knight = cuda.const.array_like(piece_square_table_knight)
    c_piece_square_table_bishop = cuda.const.array_like(piece_square_table_bishop)
    c_piece_square_table_rook   = cuda.const.array_like(piece_square_table_rook)
    c_piece_square_table_queen  = cuda.const.array_like(piece_square_table_queen)

    board_id = cuda.blockIdx.x

    s_shared_i8  = cuda.shared.array(shape = 0, dtype = np.int8)
    s_shared_i32 = cuda.shared.array(shape = 0, dtype = np.int32)
    s_shared_f32 = cuda.shared.array(shape = 0, dtype = np.float32)

    s_boards             = s_shared_i8 [                        : 64 * (search_depth + 1)]
    s_castling_rights    = s_shared_i8 [64 * (search_depth + 1) : 65 * (search_depth + 1)]
    s_en_passant_targets = s_shared_i8 [65 * (search_depth + 1) : 66 * (search_depth + 1)]
    s_halfmoves          = s_shared_i8 [66 * (search_depth + 1) : 67 * (search_depth + 1)]
    s_status             = s_shared_i32[17 * (search_depth + 1) : 18 * (search_depth + 1)]
    s_alphas             = s_shared_f32[18 * (search_depth + 1) : 19 * (search_depth + 1)]
    s_betas              = s_shared_f32[19 * (search_depth + 1) : 20 * (search_depth + 1)]

    s_should_break = cuda.shared.array(shape = 1, dtype = np.int32)

    if cuda.threadIdx.x < 64:
        s_boards[cuda.threadIdx.x] = boards[board_id * 64 + cuda.threadIdx.x]
        if cuda.threadIdx.x == 0:
            s_castling_rights[0]    = castling_rights[board_id]
            s_en_passant_targets[0] = en_passant_targets[board_id]
            s_halfmoves[0]          = halfmoves[board_id]

            lock(mutex)

            s_should_break[0] = 0

            s_alphas[0] = -alpha_beta[1]
            s_betas[0]  = -alpha_beta[0]
            
            if alpha_beta[0] >= alpha_beta[1]:
                s_should_break[0] = 1

            unlock(mutex)

            for i in range(search_depth + 1):
                s_status[i] = 1
            s_status[0] = 0
    cuda.syncthreads()

    if alpha_beta_prunning and s_should_break[0] != 0:
        return

    out_moves   = cuda.local.array(256, dtype = np.int32)
    move_scores = cuda.local.array(256, dtype = np.float32)

    if cuda.threadIdx.x <= search_depth:
        _side = side
        if cuda.threadIdx.x % 2 != 0:
            _side = not side

        while True:
            while cuda.atomic.add(s_status, cuda.threadIdx.x, 0) == 1:
                continue

            if cuda.atomic.add(s_status, cuda.threadIdx.x, 0) > 1:
                break
            
            cuda.atomic.add(evaluation_count, 0, 1)
            if cuda.threadIdx.x == search_depth:
                evaluation = 0.0
                for i in range(64):
                    evaluation += piece_value_device(s_boards[cuda.threadIdx.x * 64 + i], i, 
                        c_piece_base_values,
                        c_piece_square_table_king,
                        c_piece_square_table_pawn,
                        c_piece_square_table_knight,
                        c_piece_square_table_bishop,
                        c_piece_square_table_rook,
                        c_piece_square_table_queen)
                    
                s_alphas[cuda.threadIdx.x] = evaluation * (1.0 if _side else -1.0)
            else:
                if s_halfmoves[cuda.threadIdx.x] >= 100:
                    s_alphas[cuda.threadIdx.x] = 0.0
                else:
                    castling_right = s_castling_rights[cuda.threadIdx.x]
                    castle_WK = (castling_right & 0x01 != 0)
                    castle_WQ = (castling_right & 0x02 != 0)
                    castle_BK = (castling_right & 0x04 != 0)
                    castle_BQ = (castling_right & 0x08 != 0)

                    if move_sorting:
                        n_moves, attacked_positions = generate_all_moves_sorted_device(s_boards[cuda.threadIdx.x * 64:], _side, castle_WK, castle_WQ, castle_BK, castle_BQ, s_en_passant_targets[cuda.threadIdx.x], s_halfmoves[cuda.threadIdx.x], out_moves, move_scores, c_piece_base_values)
                    else:
                        n_moves, attacked_positions = generate_all_moves_device(s_boards[cuda.threadIdx.x * 64:], _side, castle_WK, castle_WQ, castle_BK, castle_BQ, s_en_passant_targets[cuda.threadIdx.x], s_halfmoves[cuda.threadIdx.x], out_moves)

                    if n_moves <= 0:
                        king_position = get_king_position_device(s_boards[cuda.threadIdx.x * 64:], _side)
                        if (attacked_positions & (1 << king_position)) == 0:
                            s_alphas[cuda.threadIdx.x] = 0.0
                    else:
                        for move_index in range(n_moves):
                            for i in range(64):
                                s_boards[cuda.threadIdx.x * 64 + 64 + i] = s_boards[cuda.threadIdx.x * 64 + i]

                            _, _castle_WK, _castle_WQ, _castle_BK, _castle_BQ, _en_passant_target, _halfmove = move_device(
                                s_boards[cuda.threadIdx.x * 64 + 64 : cuda.threadIdx.x * 64 + 128], _side, out_moves[move_index], 
                                castle_WK, castle_WQ, castle_BK, castle_BQ, s_en_passant_targets[cuda.threadIdx.x], s_halfmoves[cuda.threadIdx.x])
                            _castling_right = 0x00
                            if _castle_WK:
                                _castling_right |= (1 << 0)
                            if _castle_WQ:
                                _castling_right |= (1 << 1)
                            if _castle_BK:
                                _castling_right |= (1 << 2)
                            if _castle_BQ:
                                _castling_right |= (1 << 3)
                            s_castling_rights[cuda.threadIdx.x + 1]    = _castling_right
                            s_en_passant_targets[cuda.threadIdx.x + 1] = _en_passant_target
                            s_halfmoves[cuda.threadIdx.x + 1]          = _halfmove
                            s_alphas[cuda.threadIdx.x + 1]             = -s_betas[cuda.threadIdx.x]
                            s_betas[cuda.threadIdx.x + 1]              = -s_alphas[cuda.threadIdx.x]

                            cuda.atomic.exch(s_status, cuda.threadIdx.x + 1, 0)
                            while cuda.atomic.add(s_status, cuda.threadIdx.x + 1, 0) == 0:
                                continue

                            if alpha_beta_prunning and -s_alphas[cuda.threadIdx.x + 1] >= s_betas[cuda.threadIdx.x]:
                                s_alphas[cuda.threadIdx.x] = s_betas[cuda.threadIdx.x]
                                break
                            
                            if -s_alphas[cuda.threadIdx.x + 1] > s_alphas[cuda.threadIdx.x]:
                                s_alphas[cuda.threadIdx.x] = -s_alphas[cuda.threadIdx.x + 1]

            if cuda.threadIdx.x == 0:
                for i in range(search_depth + 1):
                    cuda.atomic.exch(s_status, i, 2)
            else:
                cuda.atomic.exch(s_status, cuda.threadIdx.x, 1)
    cuda.syncthreads()    

    if cuda.threadIdx.x == 0:
        lock(mutex)
        eval = -s_alphas[0]
        if eval > alpha_beta[0] or (eval == alpha_beta[0] and board_id < out_move_index[0]):
            alpha_beta[0] = eval
            out_move_index[0] = board_id
        unlock(mutex)

def evaluate_move_parallel_3(
    game, 
    search_depth,
    alpha, 
    beta, 
    alpha_beta_prunning = False, 
    move_sorting        = False
):
    global evaluation_count
    evaluation_count += 1

    if game.get_halfmove() >= 100:
        return 0.0

    if search_depth == 0:
        return game.evaluate()

    moves = game.generate_all_moves()
    if moves is None or len(moves) <= 0:
        if game.is_current_side_in_check():
            return -1000000.0
        return 0.0

    if move_sorting:
        moves = sort_moves(game, moves, best_to_worst=True)

    first_move = moves[0]
    moves.pop(0)

    game_copy = game.copy()
    game_copy.move(first_move)
    eval = -evaluate_move_parallel_3(game_copy, search_depth - 1, -beta, -alpha, alpha_beta_prunning, move_sorting)
    del game_copy

    if alpha_beta_prunning and eval >= beta:
        return beta

    if eval > alpha:
        alpha = eval

    if len(moves) > 0:
        n_boards = len(moves)
        boards             = []
        castling_rights    = []
        en_passant_targets = []
        halfmoves          = []

        for move in moves:
            game_copy = game.copy()
            game_copy.move(move)

            boards += game_copy.get_board()
            castling_right = 0x00
            if game_copy.m_castle_WK:
                castling_right |= (1 << 0)
            if game_copy.m_castle_WQ:
                castling_right |= (1 << 1)
            if game_copy.m_castle_BK:
                castling_right |= (1 << 2)
            if game_copy.m_castle_BQ:
                castling_right |= (1 << 3)
            castling_rights.append(castling_right)
            en_passant_targets.append(-1 if game_copy.m_en_passant_target is None else game_copy.m_en_passant_target)
            halfmoves.append(game_copy.get_halfmove())
            del game_copy   

        block_size = 64
        while block_size < search_depth:
            block_size *= 2
        grid_size  = n_boards

        d_boards             = cuda.to_device(np.array(boards            , dtype = np.int8))
        d_casting_rights     = cuda.to_device(np.array(castling_rights   , dtype = np.int8))
        d_en_passant_targets = cuda.to_device(np.array(en_passant_targets, dtype = np.int8))
        d_halfmoves          = cuda.to_device(np.array(halfmoves         , dtype = np.int8))
        d_alpha_beta         = cuda.to_device(np.array([alpha, beta]     , dtype = np.float32))
        d_out_move_index     = cuda.to_device(np.array([-1]              , dtype = np.int32))
        d_mutex              = cuda.to_device(np.array([0]               , dtype = np.int32))
        d_evaluation_count   = cuda.to_device(np.zeros(1, dtype = np.int32))

        evaluate_move_parallel_3_kernel[grid_size, block_size, 0, search_depth * 80](d_boards, n_boards, not game.side_to_move(), 
            d_casting_rights, d_en_passant_targets, d_halfmoves, search_depth - 1, d_evaluation_count, d_alpha_beta, d_out_move_index, d_mutex, alpha_beta_prunning, move_sorting)

        evaluation_count += d_evaluation_count.copy_to_host()[0]

        alpha = d_alpha_beta.copy_to_host()[0]

    return alpha

def find_move_parallel_3(
    game, 
    search_depth,
    alpha_beta_prunning = False, 
    move_sorting        = False
):
    global evaluation_count
    evaluation_count = 1

    moves = game.generate_all_moves()
    if moves is None or len(moves) <= 0:
        return None, 0.0

    if move_sorting:
        moves = sort_moves(game, moves, best_to_worst=True)

    first_move = moves[0]
    moves.pop(0)

    game_copy = game.copy()
    game_copy.move(first_move)
    best_eval = -evaluate_move_parallel_3(game_copy, search_depth - 1, -1000000.0, 1000000.0, alpha_beta_prunning, move_sorting)
    best_move = first_move
    del game_copy

    if len(moves) > 0:
        n_boards = len(moves)
        boards             = []
        castling_rights    = []
        en_passant_targets = []
        halfmoves          = []

        for move in moves:
            game_copy = game.copy()
            game_copy.move(move)

            boards += game_copy.get_board()
            castling_right = 0x00
            if game_copy.m_castle_WK:
                castling_right |= (1 << 0)
            if game_copy.m_castle_WQ:
                castling_right |= (1 << 1)
            if game_copy.m_castle_BK:
                castling_right |= (1 << 2)
            if game_copy.m_castle_BQ:
                castling_right |= (1 << 3)
            castling_rights.append(castling_right)
            en_passant_targets.append(-1 if game_copy.m_en_passant_target is None else game_copy.m_en_passant_target)
            halfmoves.append(game_copy.get_halfmove())
            del game_copy   

        block_size = 64
        grid_size  = n_boards

        d_boards             = cuda.to_device(np.array(boards                , dtype = np.int8))
        d_casting_rights     = cuda.to_device(np.array(castling_rights       , dtype = np.int8))
        d_en_passant_targets = cuda.to_device(np.array(en_passant_targets    , dtype = np.int8))
        d_halfmoves          = cuda.to_device(np.array(halfmoves             , dtype = np.int8))
        d_alpha_beta         = cuda.to_device(np.array([best_eval, 1000000.0], dtype = np.float32))
        d_out_move_index     = cuda.to_device(np.array([-1]                  , dtype = np.int32))
        d_mutex              = cuda.to_device(np.array([0]                   , dtype = np.int32))
        d_evaluation_count   = cuda.to_device(np.zeros(1, dtype = np.int32))

        evaluate_move_parallel_3_kernel[grid_size, block_size, 0, search_depth * 80](d_boards, n_boards, not game.side_to_move(), 
            d_casting_rights, d_en_passant_targets, d_halfmoves, search_depth - 1, d_evaluation_count, d_alpha_beta, d_out_move_index, d_mutex, alpha_beta_prunning, move_sorting)

        evaluation_count += d_evaluation_count.copy_to_host()[0]

        move_index = d_out_move_index.copy_to_host()[0]
        if move_index >= 0:
            best_eval = d_alpha_beta.copy_to_host()[0]
            best_move = moves[move_index]

    return best_move, best_eval

# Versions:
#
# -   0: Sequential minimax
# -   1: Sequential minimax with alpha - beta pruning
# -   2: Sequential minimax with alpha - beta pruning and move sorting
#
# -   3: Parallel v1
# -   4: Parallel v1 with alpha - beta pruning
# -   5: Parallel v1 with alpha - beta pruning and move sorting
#
# -   6: Parallel v2
# -   7: Parallel v2 with alpha - beta pruning
# -   8: Parallel v2 with alpha - beta pruning and move sorting
#
# -   9: Parallel v3
# -  10: Parallel v3 with alpha - beta pruning
# -  11: Parallel v3 with alpha - beta pruning and move sorting
#

def find_move(game, search_depth, version):
    match version:
        case 0:
            return find_move_sequencial(game, search_depth)
        case 1:
            return find_move_sequencial(game, search_depth, alpha_beta_prunning = True)
        case 2:
            return find_move_sequencial(game, search_depth, alpha_beta_prunning = True, move_sorting = True)
        case 3: 
            return find_move_parallel_1(game, search_depth)
        case 4: 
            return find_move_parallel_1(game, search_depth, alpha_beta_prunning = True)
        case 5: 
            return find_move_parallel_1(game, search_depth, alpha_beta_prunning = True, move_sorting = True)
        case 6:
            return find_move_parallel_2(game, search_depth)
        case 7:
            return find_move_parallel_2(game, search_depth, alpha_beta_prunning = True)
        case 8:
            return find_move_parallel_2(game, search_depth, alpha_beta_prunning = True, move_sorting = True)
        case 9:
            return find_move_parallel_3(game, search_depth)
        case 10:
            return find_move_parallel_3(game, search_depth, alpha_beta_prunning = True)
        case 11:
            return find_move_parallel_3(game, search_depth, alpha_beta_prunning = True, move_sorting = True)

def compile_kernels():
    game = ChessGame('k7/8/8/8/8/8/8/K7 w - - 0 1')
    find_move(game, 1, 3)
    find_move(game, 2, 6)
    find_move(game, 2, 9)

compile_kernels()