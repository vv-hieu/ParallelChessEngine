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

# Device-side piece value function

piece_base_values         = np.array(ChessPieces.PIECE_TYPE_VALUES        , dtype=float)
piece_square_table_king   = np.array(ChessPieces.PIECE_SQUARE_TABLE_KING  , dtype=float)
piece_square_table_pawn   = np.array(ChessPieces.PIECE_SQUARE_TABLE_PAWN  , dtype=float)
piece_square_table_knight = np.array(ChessPieces.PIECE_SQUARE_TABLE_KNIGHT, dtype=float)
piece_square_table_bishop = np.array(ChessPieces.PIECE_SQUARE_TABLE_BISHOP, dtype=float)
piece_square_table_rook   = np.array(ChessPieces.PIECE_SQUARE_TABLE_ROOK  , dtype=float)
piece_square_table_queen  = np.array(ChessPieces.PIECE_SQUARE_TABLE_QUEEN , dtype=float)

@cuda.jit(device=True)
def piece_value_device(piece: int, position: int, 
piece_base_values, 
piece_square_table_king, 
piece_square_table_pawn, 
piece_square_table_knight, 
piece_square_table_bishop, 
piece_square_table_rook, 
piece_square_table_queen):
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

# Device-side move generation function

@cuda.jit(device=True, inline=True)
def is_in_bound_device(r: int, c: int):
    return r >= 0 and r < 8 and c >= 0 and c < 8

@cuda.jit(device=True)
def push_back(list, list_count, value):
    list[list_count] = value
    return list_count + 1

@cuda.jit(device=True, inline=True)
def encode_move(start_position: int, end_position: int, promote_to: int, en_passant: bool):
    if promote_to > 0:
        promote_to -= 2

    res = 0x00

    res |= (start_position & 0xFF) << 0 
    res |= (end_position   & 0xFF) << 8
    res |= promote_to << 16
    res |= (1 if en_passant else 0) << 19

    return res

@cuda.jit(device=True, inline=True)
def decode_move(move: int):
    start_position = (move >>  0) & 0xFF
    end_position   = (move >>  8) & 0xFF
    promote_to     = (move >> 16) & 0x07
    en_passant     = (move >> 19) & 0x01 > 0

    if promote_to > 0:
        promote_to += 2

    return start_position, end_position, promote_to, en_passant

@cuda.jit(device=True)
def get_king_position_device(board, side: bool):
    king = (0x08 | 0x01) if side else 0x01
    for i in range(64):
        if board[i] == king:
            return i
    return -1

@cuda.jit(device=True)
def generate_moves_device(board, side: bool, castle_WK: bool, castle_WQ: bool, castle_BK: bool, castle_BQ: bool, en_passant_target: int, king_position: int, position: int, required_destinations: int, pinned_pieces: int, attacked_positions: int, count: int, out_moves):
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
                        count = push_back(out_moves, count, encode_move(position, position2, 0x00, False))
                break
            if valid:
                count = push_back(out_moves, count, encode_move(position, position2, 0x00, False))
            i += 1
        i = 1
        while is_in_bound_device(r - i, c):
            position2 = (r - i) * 8 + c
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and (unpinned or king_c == c)
            if piece2 & 0x07 != 0:
                if side != (piece2 & 0x08 != 0):
                    if valid:
                        count = push_back(out_moves, count, encode_move(position, position2, 0x00, False))
                break
            if valid:
                count = push_back(out_moves, count, encode_move(position, position2, 0x00, False))
            i += 1
        i = 1
        while is_in_bound_device(r, c + i):
            position2 = r * 8 + c + i
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and (unpinned or king_r == r)
            if piece2 & 0x07 != 0:
                if side != (piece2 & 0x08 != 0):
                    if valid:
                        count = push_back(out_moves, count, encode_move(position, position2, 0x00, False))
                break
            if valid:
                count = push_back(out_moves, count, encode_move(position, position2, 0x00, False))
            i += 1
        i = 1
        while is_in_bound_device(r, c - i):
            position2 = r * 8 + c - i
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and (unpinned or king_r == r)
            if piece2 & 0x07 != 0:
                if side != (piece2 & 0x08 != 0):
                    if valid:
                        count = push_back(out_moves, count, encode_move(position, position2, 0x00, False))
                break
            if valid:
                count = push_back(out_moves, count, encode_move(position, position2, 0x00, False))
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
                        count = push_back(out_moves, count, encode_move(position, position2, 0x00, False))
                break
            if valid:
                count = push_back(out_moves, count, encode_move(position, position2, 0x00, False))
            i += 1
        i = 1
        while is_in_bound_device(r - i, c + i):
            position2 = (r - i) * 8 + c + i
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and (unpinned or r - king_r == king_c - c)
            if piece2 & 0x07 != 0:
                if side != (piece2 & 0x08 != 0):
                    if valid:
                        count = push_back(out_moves, count, encode_move(position, position2, 0x00, False))
                break
            if valid:
                count = push_back(out_moves, count, encode_move(position, position2, 0x00, False))
            i += 1
        i = 1
        while is_in_bound_device(r + i, c - i):
            position2 = (r + i) * 8 + c - i
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and (unpinned or r - king_r == king_c - c)
            if piece2 & 0x07 != 0:
                if side != (piece2 & 0x08 != 0):
                    if valid:
                        count = push_back(out_moves, count, encode_move(position, position2, 0x00, False))
                break
            if valid:
                count = push_back(out_moves, count, encode_move(position, position2, 0x00, False))
            i += 1
        i = 1
        while is_in_bound_device(r - i, c - i):
            position2 = (r - i) * 8 + c - i
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and (unpinned or king_r - r == king_c - c)
            if piece2 & 0x07 != 0:
                if side != (piece2 & 0x08 != 0):
                    if valid:
                        count = push_back(out_moves, count, encode_move(position, position2, 0x00, False))
                break
            if valid:
                count = push_back(out_moves, count, encode_move(position, position2, 0x00, False))
            i += 1

    if piece_type == 0x03: # Knight
        if is_in_bound_device(r + 2, c + 1):
            position2 = (r + 2) * 8 + c + 1
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and unpinned
            if valid and (piece2 & 0x07 == 0 or side != (piece2 & 0x08 != 0)):
                count = push_back(out_moves ,count, encode_move(position, position2, 0x00, False))
        if is_in_bound_device(r - 2, c + 1):
            position2 = (r - 2) * 8 + c + 1
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and unpinned
            if valid and (piece2 & 0x07 == 0 or side != (piece2 & 0x08 != 0)):
                count = push_back(out_moves ,count, encode_move(position, position2, 0x00, False))
        if is_in_bound_device(r + 2, c - 1):
            position2 = (r + 2) * 8 + c - 1
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and unpinned
            if valid and (piece2 & 0x07 == 0 or side != (piece2 & 0x08 != 0)):
                count = push_back(out_moves ,count, encode_move(position, position2, 0x00, False))
        if is_in_bound_device(r - 2, c - 1):
            position2 = (r - 2) * 8 + c - 1
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and unpinned
            if valid and (piece2 & 0x07 == 0 or side != (piece2 & 0x08 != 0)):
                count = push_back(out_moves ,count, encode_move(position, position2, 0x00, False))
        if is_in_bound_device(r + 1, c + 2):
            position2 = (r + 1) * 8 + c + 2
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and unpinned
            if valid and (piece2 & 0x07 == 0 or side != (piece2 & 0x08 != 0)):
                count = push_back(out_moves ,count, encode_move(position, position2, 0x00, False))
        if is_in_bound_device(r - 1, c + 2):
            position2 = (r - 1) * 8 + c + 2
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and unpinned
            if valid and (piece2 & 0x07 == 0 or side != (piece2 & 0x08 != 0)):
                count = push_back(out_moves ,count, encode_move(position, position2, 0x00, False))
        if is_in_bound_device(r + 1, c - 2):
            position2 = (r + 1) * 8 + c - 2
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and unpinned
            if valid and (piece2 & 0x07 == 0 or side != (piece2 & 0x08 != 0)):
                count = push_back(out_moves ,count, encode_move(position, position2, 0x00, False))
        if is_in_bound_device(r - 1, c - 2):
            position2 = (r - 1) * 8 + c - 2
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and unpinned
            if valid and (piece2 & 0x07 == 0 or side != (piece2 & 0x08 != 0)):
                count = push_back(out_moves ,count, encode_move(position, position2, 0x00, False))

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
                        count = push_back(out_moves, count, encode_move(position, position2, 0x03, False))
                        count = push_back(out_moves, count, encode_move(position, position2, 0x04, False))
                        count = push_back(out_moves, count, encode_move(position, position2, 0x05, False))
                        count = push_back(out_moves, count, encode_move(position, position2, 0x06, False))
                    else:
                        count = push_back(out_moves, count, encode_move(position, position2, 0x00, False))
                if r == first_r:
                    position3 = (r + pawn_dr * 2) * 8 + c
                    piece3 = board[position3]
                    valid = required_destinations & (1 << position3) != 0 and (unpinned or king_c == c)
                    if valid and piece3 & 0x07 == 0:
                        count = push_back(out_moves ,count, encode_move(position, position3, 0x00, False))
        if is_in_bound_device(r + pawn_dr, c - 1):
            position2 = (r + pawn_dr) * 8 + c - 1
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and (unpinned or r - king_r == pawn_dr * (king_c - c))
            if valid:
                if piece2 & 0x07 != 0 and side != (piece2 & 0x08 != 0):
                    if r + pawn_dr == last_r:
                        count = push_back(out_moves, count, encode_move(position, position2, 0x03, False))
                        count = push_back(out_moves, count, encode_move(position, position2, 0x04, False))
                        count = push_back(out_moves, count, encode_move(position, position2, 0x05, False))
                        count = push_back(out_moves, count, encode_move(position, position2, 0x06, False))
                    else:
                        count = push_back(out_moves, count, encode_move(position, position2, 0x00, False))
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
                        count = push_back(out_moves ,count, encode_move(position, position2, 0x00, True))
        if is_in_bound_device(r + pawn_dr, c + 1):
            position2 = (r + pawn_dr) * 8 + c + 1
            piece2 = board[position2]
            valid = required_destinations & (1 << position2) != 0 and (unpinned or king_r - r == pawn_dr * (king_c - c))
            if valid:
                if piece2 & 0x07 != 0 and side != (piece2 & 0x08 != 0):
                    if r + pawn_dr == last_r:
                        count = push_back(out_moves ,count, encode_move(position, position2, 0x03, False))
                        count = push_back(out_moves ,count, encode_move(position, position2, 0x04, False))
                        count = push_back(out_moves ,count, encode_move(position, position2, 0x05, False))
                        count = push_back(out_moves ,count, encode_move(position, position2, 0x06, False))
                    else:
                        count = push_back(out_moves ,count, encode_move(position, position2, 0x00, False))
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
                        count = push_back(out_moves ,count, encode_move(position, position2, 0x00, True))

    if piece_type == 0x01: # King
        if is_in_bound_device(r + 1, c + 1):
            position2 = (r + 1) * 8 + c + 1
            piece2 = board[position2]
            valid = attacked_positions & (1 << position2) == 0
            if valid and (side != (piece2 & 0x08 != 0) or piece2 & 0x07 == 0):
                count = push_back(out_moves, count, encode_move(position, position2, 0x00, False))
        if is_in_bound_device(r + 1, c):
            position2 = (r + 1) * 8 + c
            piece2 = board[position2]
            valid = attacked_positions & (1 << position2) == 0
            if valid and (side != (piece2 & 0x08 != 0) or piece2 & 0x07 == 0):
                count = push_back(out_moves, count, encode_move(position, position2, 0x00, False))
        if is_in_bound_device(r + 1, c - 1):
            position2 = (r + 1) * 8 + c - 1
            piece2 = board[position2]
            valid = attacked_positions & (1 << position2) == 0
            if valid and (side != (piece2 & 0x08 != 0) or piece2 & 0x07 == 0):
                count = push_back(out_moves, count, encode_move(position, position2, 0x00, False))
        if is_in_bound_device(r, c + 1):
            position2 = r * 8 + c + 1
            piece2 = board[position2]
            valid = attacked_positions & (1 << position2) == 0
            if valid and (side != (piece2 & 0x08 != 0) or piece2 & 0x07 == 0):
                count = push_back(out_moves, count, encode_move(position, position2, 0x00, False))
        if is_in_bound_device(r, c - 1):
            position2 = r * 8 + c - 1
            piece2 = board[position2]
            valid = attacked_positions & (1 << position2) == 0
            if valid and (side != (piece2 & 0x08 != 0) or piece2 & 0x07 == 0):
                count = push_back(out_moves, count, encode_move(position, position2, 0x00, False))
        if is_in_bound_device(r - 1, c + 1):
            position2 = (r - 1) * 8 + c + 1
            piece2 = board[position2]
            valid = attacked_positions & (1 << position2) == 0
            if valid and (side != (piece2 & 0x08 != 0) or piece2 & 0x07 == 0):
                count = push_back(out_moves, count, encode_move(position, position2, 0x00, False))
        if is_in_bound_device(r - 1, c):
            position2 = (r - 1) * 8 + c
            piece2 = board[position2]
            valid = attacked_positions & (1 << position2) == 0
            if valid and (side != (piece2 & 0x08 != 0) or piece2 & 0x07 == 0):
                count = push_back(out_moves, count, encode_move(position, position2, 0x00, False))
        if is_in_bound_device(r - 1, c - 1):
            position2 = (r - 1) * 8 + c - 1
            piece2 = board[position2]
            valid = attacked_positions & (1 << position2) == 0
            if valid and (side != (piece2 & 0x08 != 0) or piece2 & 0x07 == 0):
                count = push_back(out_moves, count, encode_move(position, position2, 0x00, False))

        if side:
            if castle_WK and (attacked_positions & (0x07 << 4)) == 0:
                count = push_back(out_moves, count, encode_move(4, 6, 0x00, False))
            if castle_WQ and (attacked_positions & (0x07 << 2)) == 0:
                count = push_back(out_moves, count, encode_move(4, 2, 0x00, False))
        else:
            if castle_BK and (attacked_positions & (0x07 << 60)) == 0:
                count = push_back(out_moves, count, encode_move(60, 62, 0x00, False))
            if castle_BQ and (attacked_positions & (0x07 << 58)) == 0:
                count = push_back(out_moves, count, encode_move(60, 58, 0x00, False))

    return count

@cuda.jit(device=True)
def generate_all_moves_device(board, side: bool, castle_WK: bool, castle_WQ: bool, castle_BK: bool, castle_BQ: bool, en_passant_target: int, halfmoves: int, out_moves):
    if halfmoves >= 100:
        return 0

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
    return count

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

    return best_move

# Parallel implementation 1 - Parallel evaluation of moves at search_depth of 0
# Options: alpha_beta_prunning : Enable/disable alpha beta pruning
#          move_sorting        : Enable/disable move sorting
# Description: 1 block use <MAX_THREADS_PER_BLOCK> threads, every 64 threads evaluate a chess board, and reduce locally to find the best moves 

@cuda.jit
def evaluate_move_parallel_1_kernel(boards, n_boards, scores, perspective):
    c_piece_base_values         = cuda.const.array_like(piece_base_values)
    c_piece_square_table_king   = cuda.const.array_like(piece_square_table_king)
    c_piece_square_table_pawn   = cuda.const.array_like(piece_square_table_pawn)
    c_piece_square_table_knight = cuda.const.array_like(piece_square_table_knight)
    c_piece_square_table_bishop = cuda.const.array_like(piece_square_table_bishop)
    c_piece_square_table_rook   = cuda.const.array_like(piece_square_table_rook)
    c_piece_square_table_queen  = cuda.const.array_like(piece_square_table_queen)

    id = cuda.grid(1)
    s_evaluation = cuda.shared.array(shape=0, dtype=np.float32)

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

    best_eval = -1000000.0
    best_move = None

    for move in moves:
        game_copy = game.copy()
        game_copy.move(move)
        eval = -evaluate_move_parallel_1(game_copy, search_depth - 1, -1000000.0, -best_eval, alpha_beta_prunning, move_sorting)
        del game_copy
        if eval > best_eval:
            best_move = move
            best_eval = eval

    return best_move

# Parallel implementation 2

def evaluate_move_parallel_2(game: ChessGame, search_depth: int):
    global evaluation_count
    evaluation_count += 1

    if game.get_halfmove() >= 100:
        return 0.0

    moves = game.generate_all_moves()
    if moves is None or len(moves) <= 0:
        if game.is_current_side_in_check():
            return -1000000.0
        return 0.0

    first_move = moves[0]
    moves.remove(first_move)

    #if move_sorting:
    #    moves = sort_moves(game, moves, best_to_worst=True)

    game_copy = game.copy()
    game_copy.move(first_move)
    best_eval = -evaluate_move_parallel_2(game_copy, search_depth - 1)
    del game_copy

    if len(moves) > 0:
        pass

    return best_eval

def find_move_parallel_2(game: ChessGame, search_depth: int):
    global evaluation_count
    evaluation_count = 1

    moves = game.generate_all_moves()
    if moves is None or len(moves) <= 0:
        return None

    #if move_sorting:
    #    moves = sort_moves(game, moves, best_to_worst=True)

    first_move = moves[0]
    moves.remove(first_move)

    game_copy = game.copy()
    game_copy.move(first_move)
    best_eval = -evaluate_move_parallel_2(game_copy, search_depth - 1)
    best_move = first_move
    del game_copy

    if len(moves) > 0:
        pass

    return best_move

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
