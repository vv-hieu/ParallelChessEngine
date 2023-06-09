import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np

# Move contains start and end position

class Move:
    def __init__(self, 
        start_position, 
        end_position, 
        en_passant_target = None, 
        promote_to        = None
    ):
        self.m_start_position    = start_position
        self.m_end_position      = end_position
        self.m_en_passant_target = en_passant_target
        self.m_promote_to        = promote_to
        
    def start_position(self):
        return self.m_start_position    

    def end_position(self):
        return self.m_end_position    
    
    def en_passant_target(self):
        return self.m_en_passant_target

    def promote_to(self):
        return self.m_promote_to

# Chess pieces are represented using 4 bits:
#   + First 3 bits : piece type (6 types + 1 type for empty)
#   + 4th bit      : which side the piece belongs to

class ChessPieces:
    # CONSTANTS

    # Sides
    PIECE_SIDE_BLACK  = int(0x00)
    PIECE_SIDE_WHITE  = int(0x08)

    # Types
    PIECE_TYPE_EMPTY  = int(0x00)
    PIECE_TYPE_KING   = int(0x01)
    PIECE_TYPE_PAWN   = int(0x02)
    PIECE_TYPE_KNIGHT = int(0x03)
    PIECE_TYPE_BISHOP = int(0x04)
    PIECE_TYPE_ROOK   = int(0x05)
    PIECE_TYPE_QUEEN  = int(0x06)

    PIECE_TYPE_MASK   = int(0x07)

    # Empty
    PIECE_EMPTY = PIECE_TYPE_EMPTY

    # Empty notation
    _E = PIECE_EMPTY

    # White pieces
    PIECE_WHITE_KING   = PIECE_SIDE_WHITE | PIECE_TYPE_KING
    PIECE_WHITE_PAWN   = PIECE_SIDE_WHITE | PIECE_TYPE_PAWN
    PIECE_WHITE_KNIGHT = PIECE_SIDE_WHITE | PIECE_TYPE_KNIGHT
    PIECE_WHITE_BISHOP = PIECE_SIDE_WHITE | PIECE_TYPE_BISHOP
    PIECE_WHITE_ROOK   = PIECE_SIDE_WHITE | PIECE_TYPE_ROOK
    PIECE_WHITE_QUEEN  = PIECE_SIDE_WHITE | PIECE_TYPE_QUEEN

    # White pieces notation
    WK = PIECE_WHITE_KING
    WP = PIECE_WHITE_PAWN
    WN = PIECE_WHITE_KNIGHT
    WB = PIECE_WHITE_BISHOP
    WR = PIECE_WHITE_ROOK
    WQ = PIECE_WHITE_QUEEN

    # Black pieces
    PIECE_BLACK_KING   = PIECE_SIDE_BLACK | PIECE_TYPE_KING
    PIECE_BLACK_PAWN   = PIECE_SIDE_BLACK | PIECE_TYPE_PAWN
    PIECE_BLACK_KNIGHT = PIECE_SIDE_BLACK | PIECE_TYPE_KNIGHT
    PIECE_BLACK_BISHOP = PIECE_SIDE_BLACK | PIECE_TYPE_BISHOP
    PIECE_BLACK_ROOK   = PIECE_SIDE_BLACK | PIECE_TYPE_ROOK
    PIECE_BLACK_QUEEN  = PIECE_SIDE_BLACK | PIECE_TYPE_QUEEN

    # Black pieces notation
    BK = PIECE_BLACK_KING
    BP = PIECE_BLACK_PAWN
    BN = PIECE_BLACK_KNIGHT
    BB = PIECE_BLACK_BISHOP
    BR = PIECE_BLACK_ROOK
    BQ = PIECE_BLACK_QUEEN

    # Names and notations
    PIECE_TYPES_NAMES     = [ "Empty", "King", "Pawn", "Knight", "Bishop", "Rook", "Queen" ]
    PIECE_TYPES_NOTATIONS = [ "E", "K", "P", "N", "B", "R", "Q" ]

    # Piece value (https://en.wikipedia.org/wiki/Chess_piece_relative_value#Standard_valuations)
    # King has no value
    PIECE_TYPE_VALUES = [ 0.0, 0.0, 1.0, 3.0, 3.0, 5.0, 9.0 ]

    # Piece square tables (https://www.chessprogramming.org/Simplified_Evaluation_Function)
    PIECE_SQUARE_TABLE_KING   = [
         0.20,  0.30,  0.10,  0.00,  0.00,  0.10,  0.30,  0.20,
         0.20,  0.20,  0.00,  0.00,  0.00,  0.00,  0.20,  0.20,
        -0.10, -0.20, -0.20, -0.20, -0.20, -0.20, -0.20, -0.10,
        -0.20, -0.30, -0.30, -0.40, -0.40, -0.30, -0.30, -0.20,
        -0.30, -0.40, -0.40, -0.50, -0.50, -0.40, -0.40, -0.30,
        -0.30, -0.40, -0.40, -0.50, -0.50, -0.40, -0.40, -0.30,
        -0.30, -0.40, -0.40, -0.50, -0.50, -0.40, -0.40, -0.30,
        -0.30, -0.40, -0.40, -0.50, -0.50, -0.40, -0.40, -0.30
    ]
    PIECE_SQUARE_TABLE_PAWN   = [
         0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,
         0.05,  0.10,  0.10, -0.20, -0.20,  0.10,  0.10,  0.05,
         0.05, -0.05, -0.10,  0.00,  0.00, -0.10, -0.05,  0.05,
         0.00,  0.00,  0.00,  0.20,  0.20,  0.00,  0.00,  0.00,
         0.05,  0.05,  0.10,  0.25,  0.25,  0.10,  0.05,  0.05,
         0.10,  0.10,  0.20,  0.30,  0.30,  0.20,  0.10,  0.10,
         0.50,  0.50,  0.50,  0.50,  0.50,  0.50,  0.50,  0.50,
         0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00
    ]
    PIECE_SQUARE_TABLE_KNIGHT = [
        -0.50, -0.40, -0.30, -0.30, -0.30, -0.30, -0.40, -0.50,
        -0.40, -0.20,  0.00,  0.05,  0.05,  0.00, -0.20, -0.40,
        -0.30,  0.05,  0.10,  0.15,  0.15,  0.10,  0.05, -0.30,
        -0.30,  0.00,  0.15,  0.20,  0.20,  0.15,  0.00, -0.30,
        -0.30,  0.05,  0.15,  0.20,  0.20,  0.15,  0.05, -0.30,
        -0.30,  0.00,  0.10,  0.15,  0.15,  0.10,  0.00, -0.30,
        -0.40, -0.20,  0.00,  0.00,  0.00,  0.00, -0.20, -0.40,
        -0.50, -0.40, -0.30, -0.30, -0.30, -0.30, -0.40, -0.50
    ]
    PIECE_SQUARE_TABLE_BISHOP = [
        -0.20, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.20,
        -0.10,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, -0.10,
        -0.10,  0.00,  0.05,  0.10,  0.10,  0.05,  0.00, -0.10,
        -0.10,  0.05,  0.05,  0.10,  0.10,  0.05,  0.05, -0.10,
        -0.10,  0.00,  0.10,  0.10,  0.10,  0.10,  0.00, -0.10,
        -0.10,  0.10,  0.10,  0.10,  0.10,  0.10,  0.10, -0.10,
        -0.10,  0.05,  0.00,  0.00,  0.00,  0.00,  0.05, -0.10,
        -0.20, -0.10, -0.10, -0.10, -0.10, -0.10, -0.10, -0.20
    ]
    PIECE_SQUARE_TABLE_ROOK   = [
          0.00,  0.00,  0.00,  0.05,  0.05,  0.00,  0.00,  0.00,
         -0.05,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, -0.05,
         -0.05,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, -0.05,
         -0.05,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, -0.05,
         -0.05,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, -0.05,
         -0.05,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, -0.05,
          0.05,  0.10,  0.10,  0.10,  0.10,  0.10,  0.10,  0.05,
          0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00
    ]
    PIECE_SQUARE_TABLE_QUEEN  = [
        -0.20, -0.10, -0.10, -0.05, -0.05, -0.10, -0.10, -0.20,
        -0.10,  0.00,  0.05,  0.00,  0.00,  0.00,  0.00, -0.10,
        -0.10,  0.05,  0.05,  0.05,  0.05,  0.05,  0.00, -0.10,
         0.00,  0.00,  0.05,  0.05,  0.05,  0.05,  0.00, -0.05,
        -0.05,  0.00,  0.05,  0.05,  0.05,  0.05,  0.00, -0.05,
        -0.10,  0.00,  0.05,  0.05,  0.05,  0.05,  0.00, -0.10,
        -0.10,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00, -0.10,
        -0.20, -0.10, -0.10, -0.05, -0.05, -0.10, -0.10, -0.20
    ]  

    # UTILITY METHODS

    # Create piece dynamically
    def piece(
        type, 
        side
    ):
        return type | (ChessPieces.PIECE_SIDE_WHITE if side else ChessPieces.PIECE_SIDE_BLACK)

    # Check if piece is empty
    def is_empty(
        piece
    ):
        return piece & ChessPieces.PIECE_TYPE_MASK == ChessPieces.PIECE_EMPTY

    # Check which side the piece belongs to
    # True: white, False: black
    def side(
        piece
    ):
        return piece & ChessPieces.PIECE_SIDE_WHITE == ChessPieces.PIECE_SIDE_WHITE

    # Get piece's type
    def type(
        piece 
    ):
        return piece & ChessPieces.PIECE_TYPE_MASK

    # Get piece's name, including which side the piece belongs to
    def name(
        piece, 
        notation = False
    ):
        if ChessPieces.is_empty(piece):
            if notation:
                return ' ' + ChessPieces.PIECE_TYPES_NOTATIONS[0]
            else:
                return ChessPieces.PIECE_TYPES_NAMES[0]

        isWhite = ChessPieces.side(piece)

        if notation:
            if isWhite:
                res = 'W'
            else:
                res = 'B'
            res = res + ChessPieces.PIECE_TYPES_NOTATIONS[ChessPieces.type(piece)]
            
        else:
            if isWhite:
                res = 'White '
            else:
                res = 'Black '
            res = res + ChessPieces.PIECE_TYPES_NAMES[ChessPieces.type(piece)]
        
        return res

    # Get piece's value
    def value(
        piece, 
        position = -1
    ):
        # Base value
        base_value = ChessPieces.PIECE_TYPE_VALUES[ChessPieces.type(piece)]

        # Positional value
        positional_value = 0
        if position >= 0:
            r = position // 8
            c = position % 8
            if not ChessPieces.side(piece):
                r = 7 - r
            position = r * 8 + c
            if ChessPieces.type(piece) == ChessPieces.PIECE_TYPE_KING:
                positional_value = ChessPieces.PIECE_SQUARE_TABLE_KING[position]
            elif ChessPieces.type(piece) == ChessPieces.PIECE_TYPE_PAWN:
                positional_value = ChessPieces.PIECE_SQUARE_TABLE_PAWN[position]
            elif ChessPieces.type(piece) == ChessPieces.PIECE_TYPE_KNIGHT:
                positional_value = ChessPieces.PIECE_SQUARE_TABLE_KNIGHT[position]
            elif ChessPieces.type(piece) == ChessPieces.PIECE_TYPE_BISHOP:
                positional_value = ChessPieces.PIECE_SQUARE_TABLE_BISHOP[position]
            elif ChessPieces.type(piece) == ChessPieces.PIECE_TYPE_ROOK:
                positional_value = ChessPieces.PIECE_SQUARE_TABLE_ROOK[position]
            elif ChessPieces.type(piece) == ChessPieces.PIECE_TYPE_QUEEN:
                positional_value = ChessPieces.PIECE_SQUARE_TABLE_QUEEN[position]

        return base_value + positional_value

# Chess game simulates a game of chess

class ChessGame:
    # Icons
    ICON_BK = plt.imread('icons/black_king.png')
    ICON_BP = plt.imread('icons/black_pawn.png')
    ICON_BN = plt.imread('icons/black_knight.png')
    ICON_BB = plt.imread('icons/black_bishop.png')
    ICON_BR = plt.imread('icons/black_rook.png')
    ICON_BQ = plt.imread('icons/black_queen.png')
    ICON_WK = plt.imread('icons/white_king.png')
    ICON_WP = plt.imread('icons/white_pawn.png')
    ICON_WN = plt.imread('icons/white_knight.png')
    ICON_WB = plt.imread('icons/white_bishop.png')
    ICON_WR = plt.imread('icons/white_rook.png')
    ICON_WQ = plt.imread('icons/white_queen.png')

    # Constructor
    def __init__(self, 
        fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
    ):
        fen_components = fen.split(' ')

        # Piece placements
        self.m_board = []
        rows = fen_components[0].split('/')
        rows.reverse()
        for row in rows:
            for ch in row:
                if '1' <= ch <= '8':
                    self.m_board += [ChessPieces._E] * (ord(ch) - ord('0'))
                elif 'a' <= ch <= 'z':
                    match ch:
                        case 'k':
                            self.m_board.append(ChessPieces.BK)
                        case 'p':
                            self.m_board.append(ChessPieces.BP)
                        case 'n':
                            self.m_board.append(ChessPieces.BN)
                        case 'b':
                            self.m_board.append(ChessPieces.BB)
                        case 'r':
                            self.m_board.append(ChessPieces.BR)
                        case 'q':
                            self.m_board.append(ChessPieces.BQ)
                elif 'A' <= ch <= 'Z':
                    match ch:
                        case 'K':
                            self.m_board.append(ChessPieces.WK)
                        case 'P':
                            self.m_board.append(ChessPieces.WP)
                        case 'N':
                            self.m_board.append(ChessPieces.WN)
                        case 'B':
                            self.m_board.append(ChessPieces.WB)
                        case 'R':
                            self.m_board.append(ChessPieces.WR)
                        case 'Q':
                            self.m_board.append(ChessPieces.WQ)

        i = 1

        # Active side
        self.m_current_side = True
        if i < len(fen_components):
            match fen_components[i]:
                case 'w':
                    self.m_current_side = True
                    i += 1
                case 'b':
                    self.m_current_side = False
                    i += 1

        # Castling rights
        self.m_castle_WK = True
        self.m_castle_WQ = True
        self.m_castle_BK = True
        self.m_castle_BQ = True
        if i < len(fen_components):
            self.m_castle_WK = False
            self.m_castle_WQ = False
            self.m_castle_BK = False
            self.m_castle_BQ = False
            for ch in fen_components[i]:
                    match ch:
                        case 'K':
                            self.m_castle_WK = True
                        case 'Q':
                            self.m_castle_WQ = True
                        case 'k':
                            self.m_castle_BK = True
                        case 'q':
                            self.m_castle_BQ = True
            i += 1

        # Possible En passant targets
        self.m_en_passant_target = None
        if i < len(fen_components):
            if fen_components[i] != '-':
                c = ord(fen_components[i][0]) - ord('a')
                r = ord(fen_components[i][1]) - ord('1')
                self.m_en_passant_target = int(r * 8 + c)
            i += 1

        # Halfmove clock
        self.m_halfmove = 0
        if i < len(fen_components):
            self.m_halfmove = int(fen_components[i])
            i += 1

        # Fullmove count
        self.m_fullmove = 0
        if i < len(fen_components):
            self.m_fullmove = int(fen_components[i])
            i += 1

    # Create a copy
    def copy(self):
        res = ChessGame()

        res.m_board             = self.m_board + []
        res.m_current_side      = self.m_current_side
        res.m_castle_BK         = self.m_castle_BK
        res.m_castle_BQ         = self.m_castle_BQ
        res.m_castle_WK         = self.m_castle_WK
        res.m_castle_WQ         = self.m_castle_WQ
        res.m_en_passant_target = self.m_en_passant_target
        res.m_halfmove          = self.m_halfmove
        res.m_fullmove          = self.m_fullmove

        return res

    # Display game info
    def show(self):
        print('Pieces Placements:')
        print('    a  b  c  d  e  f  g  h ')
        for i in range(8):
            print(
                f' {8 - i} ' +
                f'{ChessPieces.name(self.m_board[(7 - i) * 8 + 0], True)} ' +
                f'{ChessPieces.name(self.m_board[(7 - i) * 8 + 1], True)} ' +
                f'{ChessPieces.name(self.m_board[(7 - i) * 8 + 2], True)} ' +
                f'{ChessPieces.name(self.m_board[(7 - i) * 8 + 3], True)} ' +
                f'{ChessPieces.name(self.m_board[(7 - i) * 8 + 4], True)} ' +
                f'{ChessPieces.name(self.m_board[(7 - i) * 8 + 5], True)} ' +
                f'{ChessPieces.name(self.m_board[(7 - i) * 8 + 6], True)} ' +
                f'{ChessPieces.name(self.m_board[(7 - i) * 8 + 7], True)} '
            )

        print('')

        side_name = 'White' if self.m_current_side else 'Black'
        print(f'Active Side: {side_name}')

        print('Castling Rights:')
        print(f' +) WK: {self.m_castle_WK}')
        print(f' +) WQ: {self.m_castle_WQ}')
        print(f' +) BK: {self.m_castle_BK}')
        print(f' +) BQ: {self.m_castle_BQ}')

        print(f'En Passant Target: {self.m_en_passant_target}')

        print(f'Halfmove: {self.m_halfmove}')
        print(f'Fullmove: {self.m_fullmove}')

    # Plot
    def plot(self, 
        moves     = None, 
        positions = None, 
        ax        = None
    ):
        board = np.ones((8, 8)) - np.indices((8, 8)).sum(axis=0) % 2

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        ax.tick_params(axis='both', labelsize=24)
        ax.set_xticks(np.arange(8))
        ax.set_xticklabels([ 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H' ])
        ax.set_yticks(np.arange(8))
        ax.set_yticklabels([ '8', '7', '6', '5', '4', '3', '2', '1' ])

        ax.imshow(board, cmap=ListedColormap([
            [0.6, 0.45, 0.2, 1.0], 
            [0.8, 0.65, 0.4, 1.0]]))

        for r in range(8):
            for c in range(8):
                piece = self.get_piece(r * 8 + c)
                extent = (c - 0.5, c + 0.5, 7 - r + 0.5, 7 - r - 0.5)
                match piece:
                    case ChessPieces.BK:
                        ax.imshow(ChessGame.ICON_BK, extent=extent)
                    case ChessPieces.BP:
                        ax.imshow(ChessGame.ICON_BP, extent=extent)
                    case ChessPieces.BN:
                        ax.imshow(ChessGame.ICON_BN, extent=extent)
                    case ChessPieces.BB:
                        ax.imshow(ChessGame.ICON_BB, extent=extent)
                    case ChessPieces.BR:
                        ax.imshow(ChessGame.ICON_BR, extent=extent)
                    case ChessPieces.BQ:
                        ax.imshow(ChessGame.ICON_BQ, extent=extent)
                    case ChessPieces.WK:
                        ax.imshow(ChessGame.ICON_WK, extent=extent)
                    case ChessPieces.WP:
                        ax.imshow(ChessGame.ICON_WP, extent=extent)
                    case ChessPieces.WN:
                        ax.imshow(ChessGame.ICON_WN, extent=extent)
                    case ChessPieces.WB:
                        ax.imshow(ChessGame.ICON_WB, extent=extent)
                    case ChessPieces.WR:
                        ax.imshow(ChessGame.ICON_WR, extent=extent)
                    case ChessPieces.WQ:
                        ax.imshow(ChessGame.ICON_WQ, extent=extent)

        if moves is not None:
            for move in moves:
                r0 = move.start_position() // 8
                c0 = move.start_position() - (r0 * 8)
                r1 = move.end_position() // 8
                c1 = move.end_position() - (r1 * 8)
                ax.arrow(c0, 7.0 - r0, c1 - c0, r0 - r1, width=0.1, length_includes_head=True, color=(1.0, 0.0, 0.0, 1.0))

        
        if positions is not None:
            x = [position % 8 for position in positions]
            y = [7 - position // 8 for position in positions]
            ax.scatter(x, y)
                

        ax.imshow(board, alpha = 0.0)

    # Current side to move
    def side_to_move(self):
        return self.m_current_side

    # Check if a piece belongs to the current side to move
    def is_side_to_move(self, 
        piece
    ):
        return ChessPieces.side(piece) == self.m_current_side

    # Set board
    def set_board(self, board: list[int]):
        self.m_board = list(board)

    # Get entire board
    def get_board(self):
        return self.m_board + []

    # Get a piece at a position
    def get_piece(self, 
        position
    ):
        return self.m_board[position]

    # Get king position
    def get_king_position(self, 
        side
    ):
        king = ChessPieces.WK if side else ChessPieces.BK
        for i in range(64):
            if self.get_piece(i) == king:
                return i
        return -1

    # Halfmove
    def get_halfmove(self):
        return self.m_halfmove

    # Halfmove
    def get_fullmove(self):
        return self.m_fullmove

    # Check if a position is in bound
    def is_in_bound(
        r, 
        c
    ):
        if r < 0 or r >= 8 or c < 0 or c >= 8:
            return False
        
        return True

    # Check if a position is currently in check
    def is_in_check(self, 
        position, 
        side, 
        exclude = None,
        include = None
    ):
        r = position // 8
        c = position % 8

        # Check by pawns
        if side:
            if ChessGame.is_in_bound(r + 1, c - 1):
                piece2 = self.get_piece((r + 1) * 8 + c - 1)
                if piece2 == ChessPieces.BP:
                    return True
            if ChessGame.is_in_bound(r + 1, c + 1):
                piece2 = self.get_piece((r + 1) * 8 + c + 1)
                if piece2 == ChessPieces.BP:
                    return True
        else:
            if ChessGame.is_in_bound(r - 1, c - 1):
                piece2 = self.get_piece((r - 1) * 8 + c - 1)
                if piece2 == ChessPieces.WP:
                    return True
            if ChessGame.is_in_bound(r - 1, c + 1):
                piece2 = self.get_piece((r - 1) * 8 + c + 1)
                if piece2 == ChessPieces.WP:
                    return True

        # Check by knights
        dirs = [ (-2, -1), (-1, -2), (2, -1), (-1, 2), (-2, 1), (1, -2), (2, 1), (1, 2) ]
        other_knight = ChessPieces.BN if side else ChessPieces.WN
        for dir in dirs:
            r2 = r + dir[1]
            c2 = c + dir[0]
            if ChessGame.is_in_bound(r2, c2):
                if self.get_piece(r2 * 8 + c2) == other_knight:
                    return True
        
        # Check by sliding pieces
        dirs = [ (0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1) ]
        other_bishop = ChessPieces.BB if side else ChessPieces.WB
        other_rook   = ChessPieces.BR if side else ChessPieces.WR
        other_queen  = ChessPieces.BQ if side else ChessPieces.WQ
        other_king   = ChessPieces.BK if side else ChessPieces.WK
        for dir in dirs[:4]:
            i = 1
            while True:
                r2 = r + dir[1] * i
                c2 = c + dir[0] * i
                if ChessGame.is_in_bound(r2, c2):
                    piece2 = self.get_piece(r2 * 8 + c2)
                    if piece2 == other_rook or piece2 == other_queen or (piece2 == other_king and i == 1):
                        return True
                    elif not (ChessPieces.is_empty(piece2) or (exclude is not None and r2 * 8 + c2 in exclude)) or \
                    (include is not None and r2 * 8 + c2 in include):
                        break
                    i += 1
                else:
                    break
        for dir in dirs[4:]:
            i = 1
            while True:
                r2 = r + dir[1] * i
                c2 = c + dir[0] * i
                if ChessGame.is_in_bound(r2, c2):
                    piece2 = self.get_piece(r2 * 8 + c2)
                    if piece2 == other_bishop or piece2 == other_queen or (piece2 == other_king and i == 1):
                        return True
                    elif not (ChessPieces.is_empty(piece2) or (exclude is not None and r2 * 8 + c2 in exclude))or \
                    (include is not None and r2 * 8 + c2 in include):
                        break
                    i += 1
                else:
                    break
        
        return False

    # Check if current side is in check
    def is_current_side_in_check(self, 
        exclude = None, 
        include = None
    ):
        king_position = self.get_king_position(self.m_current_side)
        return self.is_in_check(king_position, self.m_current_side, exclude=exclude, include=include)

    # Generate all possible moves
    def generate_all_moves(self):
        if self.m_halfmove >= 100:
            return []

        king_position = self.get_king_position(self.m_current_side)

        r = king_position // 8
        c = king_position % 8

        required_destinations = None
        pinned_pieces = {}
        if king_position >= 0:
            # Check by pawns
            if self.m_current_side:
                if ChessGame.is_in_bound(r + 1, c - 1):
                    piece2 = self.get_piece((r + 1) * 8 + c - 1)
                    if piece2 == ChessPieces.BP:
                        if required_destinations is None:
                            required_destinations = { (r + 1) * 8 + c - 1 }
                        else:
                            required_destinations = required_destinations.intersection({ (r + 1) * 8 + c - 1 })
                if ChessGame.is_in_bound(r + 1, c + 1):
                    piece2 = self.get_piece((r + 1) * 8 + c + 1)
                    if piece2 == ChessPieces.BP:
                        if required_destinations is None:
                            required_destinations = { (r + 1) * 8 + c + 1 }
                        else:
                            required_destinations = required_destinations.intersection({ (r + 1) * 8 + c + 1 })
            else:
                if ChessGame.is_in_bound(r - 1, c - 1):
                    piece2 = self.get_piece((r - 1) * 8 + c - 1)
                    if piece2 == ChessPieces.WP:
                        if required_destinations is None:
                            required_destinations = { (r - 1) * 8 + c - 1 }
                        else:
                            required_destinations = required_destinations.intersection({ (r - 1) * 8 + c - 1 })
                if ChessGame.is_in_bound(r - 1, c + 1):
                    piece2 = self.get_piece((r - 1) * 8 + c + 1)
                    if piece2 == ChessPieces.WP:
                        if required_destinations is None:
                            required_destinations = { (r - 1) * 8 + c + 1 }
                        else:
                            required_destinations = required_destinations.intersection({ (r - 1) * 8 + c + 1 })

            # Check by knights
            dirs = [ (-2, -1), (-1, -2), (2, -1), (-1, 2), (-2, 1), (1, -2), (2, 1), (1, 2) ]
            other_knight = ChessPieces.BN if self.m_current_side else ChessPieces.WN
            for dir in dirs:
                r2 = r + dir[1]
                c2 = c + dir[0]
                if ChessGame.is_in_bound(r2, c2):
                    if self.get_piece(r2 * 8 + c2) == other_knight:
                        if required_destinations is None:
                            required_destinations = { r2 * 8 + c2 }
                        else:
                            required_destinations = required_destinations.intersection({ r2 * 8 + c2 })
        
            # Check by sliding pieces
            dirs = [ (0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1) ]
            other_bishop = ChessPieces.BB if self.m_current_side else ChessPieces.WB
            other_rook   = ChessPieces.BR if self.m_current_side else ChessPieces.WR
            other_queen  = ChessPieces.BQ if self.m_current_side else ChessPieces.WQ
            other_king   = ChessPieces.BK if self.m_current_side else ChessPieces.WK
            for dir in dirs[:4]:
                i = 1
                pinned = None
                rd = set()
                found = False
                while True:
                    r2 = r + dir[1] * i
                    c2 = c + dir[0] * i
                    if ChessGame.is_in_bound(r2, c2):
                        piece2 = self.get_piece(r2 * 8 + c2)
                        if pinned is not None:
                            if piece2 == other_rook or piece2 == other_queen or (piece2 == other_king and i == 1):
                                pinned_pieces[pinned] = r2 * 8 + c2
                                break
                            if not ChessPieces.is_empty(piece2):
                                break
                        else:
                            if piece2 == other_rook or piece2 == other_queen or (piece2 == other_king and i == 1):
                                rd.add(r2 * 8 + c2)
                                found = True
                                break
                            if not ChessPieces.is_empty(piece2):
                                if self.is_side_to_move(piece2):
                                    pinned = r2 * 8 + c2
                                else:
                                    break
                        rd.add(r2 * 8 + c2)
                        i += 1
                    else:
                        break
                if found:
                    if required_destinations is None:
                        required_destinations = rd
                    else:
                        required_destinations = required_destinations.intersection(rd)
            for dir in dirs[4:]:
                i = 1
                pinned = None
                rd = set()
                found = False
                while True:
                    r2 = r + dir[1] * i
                    c2 = c + dir[0] * i
                    if ChessGame.is_in_bound(r2, c2):
                        piece2 = self.get_piece(r2 * 8 + c2)
                        if pinned is not None:
                            if piece2 == other_bishop or piece2 == other_queen or (piece2 == other_king and i == 1):
                                pinned_pieces[pinned] = r2 * 8 + c2
                                break
                            if not ChessPieces.is_empty(piece2):
                                break
                        else:
                            if piece2 == other_bishop or piece2 == other_queen or (piece2 == other_king and i == 1):
                                rd.add(r2 * 8 + c2)
                                found = True
                                break
                            if not ChessPieces.is_empty(piece2):
                                if self.is_side_to_move(piece2):
                                    pinned = r2 * 8 + c2
                                else:
                                    break
                        rd.add(r2 * 8 + c2)
                        i += 1
                    else:
                        break
                if found:
                    if required_destinations is None:
                        required_destinations = rd
                    else:
                        required_destinations = required_destinations.intersection(rd)

        res = []
        for i in range(64):
            res += self.generate_moves(self.m_board[i], i, required_destinations=required_destinations, pinned_pieces=pinned_pieces)
        return res

    # Generate all possible moves of a single piece
    def generate_moves(self, 
        piece, 
        position, 
        required_destinations = None, 
        pinned_pieces         = {}
    ):
        res = []
        if self.is_side_to_move(piece) and not ChessPieces.is_empty(piece):
            piece_type = ChessPieces.type(piece)

            r = position // 8
            c = position - (r * 8)

            pinned_by = pinned_pieces.get(position, None)
            pinned_dir = None
            if pinned_by is not None:
                rp = pinned_by // 8
                cp = pinned_by - (rp * 8)
                pinned_dir = (cp - c, rp - r)

            # Sliding pieces (bishop, rook, queen, king)
            if piece_type == ChessPieces.PIECE_TYPE_BISHOP or \
            piece_type == ChessPieces.PIECE_TYPE_ROOK or \
            piece_type == ChessPieces.PIECE_TYPE_KING or \
            piece_type == ChessPieces.PIECE_TYPE_QUEEN:
                dirs = [ (0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1) ]
                if piece_type == ChessPieces.PIECE_TYPE_BISHOP:
                    dirs = dirs[4:]
                elif piece_type == ChessPieces.PIECE_TYPE_ROOK:
                    dirs = dirs[:4]
                for dir in dirs:
                    i = 1
                    while piece_type != ChessPieces.PIECE_TYPE_KING or i <= 1:
                        r2 = r + dir[1] * i
                        c2 = c + dir[0] * i
                        if ChessGame.is_in_bound(r2, c2):
                            piece2 = self.get_piece(r2 * 8 + c2)
                            if piece_type == ChessPieces.PIECE_TYPE_KING:
                                if not self.is_in_check(r2 * 8 + c2, self.m_current_side, exclude=[position]) and (ChessPieces.is_empty(piece2) or not self.is_side_to_move(piece2)):
                                    res.append(Move(position, r2 * 8 + c2))
                            else:
                                if required_destinations is None or r2 * 8 + c2 in required_destinations:
                                    dr = r2 - r
                                    dc = c2 - c
                                    check_pinned = True
                                    if pinned_dir is not None:
                                        val = dr * pinned_dir[0] - dc * pinned_dir[1]
                                        check_pinned = val == 0
                                    if not ChessPieces.is_empty(piece2):
                                        if check_pinned and not self.is_side_to_move(piece2):
                                            res.append(Move(position, r2 * 8 + c2))
                                        break
                                    else:
                                        if check_pinned:
                                            res.append(Move(position, r2 * 8 + c2))
                                if not ChessPieces.is_empty(piece2):
                                    break
                            i += 1
                        else:
                            break

                if piece_type == ChessPieces.PIECE_TYPE_KING:
                    if self.m_current_side:
                        if self.m_castle_WK:
                            if not self.is_in_check(4, True) and \
                            not self.is_in_check(5, True) and \
                            not self.is_in_check(6, True) and \
                            ChessPieces.is_empty(self.get_piece(5)) and \
                            ChessPieces.is_empty(self.get_piece(6)):
                                res.append(Move(4, 6))
                        if self.m_castle_WQ:
                            if not self.is_in_check(4, True) and \
                            not self.is_in_check(3, True) and \
                            not self.is_in_check(2, True) and \
                            ChessPieces.is_empty(self.get_piece(3)) and \
                            ChessPieces.is_empty(self.get_piece(2)) and \
                            ChessPieces.is_empty(self.get_piece(1)):
                                res.append(Move(4, 2))
                    else:
                        if self.m_castle_BK:
                            if not self.is_in_check(60, False) and \
                            not self.is_in_check(61, False) and \
                            not self.is_in_check(62, False) and \
                            ChessPieces.is_empty(self.get_piece(61)) and \
                            ChessPieces.is_empty(self.get_piece(62)):
                                res.append(Move(60, 62))
                        if self.m_castle_BQ:
                            if not self.is_in_check(60, False) and \
                            not self.is_in_check(59, False) and \
                            not self.is_in_check(58, False) and \
                            ChessPieces.is_empty(self.get_piece(59)) and \
                            ChessPieces.is_empty(self.get_piece(58)) and \
                            ChessPieces.is_empty(self.get_piece(57)):
                                res.append(Move(60, 58))
                        
            # Knights
            elif piece_type == ChessPieces.PIECE_TYPE_KNIGHT:
                dirs = [ (-2, -1), (-1, -2), (2, -1), (-1, 2), (-2, 1), (1, -2), (2, 1), (1, 2) ]
                for dir in dirs:
                    r2 = r + dir[1]
                    c2 = c + dir[0]
                    dr = r2 - r
                    dc = c2 - c
                    check_pinned = True
                    if pinned_dir is not None:
                        val = dr * pinned_dir[0] - dc * pinned_dir[1]
                        check_pinned = val == 0
                    if ChessGame.is_in_bound(r2, c2) and (required_destinations is None or r2 * 8 + c2 in required_destinations) and check_pinned:
                        piece2 = self.get_piece(r2 * 8 + c2)
                        if ChessPieces.is_empty(piece2) or ChessPieces.side(piece2) != self.m_current_side:
                            res.append(Move(position, r2 * 8 + c2))

            # Pawns
            elif piece_type == ChessPieces.PIECE_TYPE_PAWN:
                if self.m_current_side:
                    is_promoting = r == 6
                    double = r == 1
                    advance_dir = (0, 1)
                else:
                    is_promoting = r == 1
                    double = r == 6
                    advance_dir = (0, -1)

                # Pawn advance
                r2 = r + advance_dir[1]
                c2 = c + advance_dir[0]
                check_pinned = True
                if pinned_dir is not None:
                    val = advance_dir[1] * pinned_dir[0] - advance_dir[0] * pinned_dir[1]
                    check_pinned = val == 0
                if ChessGame.is_in_bound(r2, c2):
                    piece2 = self.get_piece(r2 * 8 + c2)
                    if ChessPieces.is_empty(piece2):
                        if (required_destinations is None or r2 * 8 + c2 in required_destinations) and check_pinned:
                            if is_promoting:
                                res.append(Move(position, r2 * 8 + c2, promote_to=ChessPieces.piece(ChessPieces.PIECE_TYPE_KNIGHT, self.    m_current_side)))
                                res.append(Move(position, r2 * 8 + c2, promote_to=ChessPieces.piece(ChessPieces.PIECE_TYPE_BISHOP, self.    m_current_side)))
                                res.append(Move(position, r2 * 8 + c2, promote_to=ChessPieces.piece(ChessPieces.PIECE_TYPE_ROOK, self.  m_current_side)))
                                res.append(Move(position, r2 * 8 + c2, promote_to=ChessPieces.piece(ChessPieces.PIECE_TYPE_QUEEN, self. m_current_side)))
                            else:
                                res.append(Move(position, r2 * 8 + c2))
                        r3 = r + advance_dir[1] * 2
                        c3 = c + advance_dir[0] * 2
                        if double and ChessGame.is_in_bound(r3, c3):
                            piece3 = self.get_piece(r3 * 8 + c3)
                            if ChessPieces.is_empty(piece3) and (required_destinations is None or r3 * 8 + c3 in required_destinations) and check_pinned:
                                res.append(Move(position, r3 * 8 + c3))

                # Pawn capture
                dr = advance_dir[1]
                dc = -1
                check_pinned = True
                if pinned_dir is not None:
                    val = dr * pinned_dir[0] - dc * pinned_dir[1]
                    check_pinned = val == 0
                en_passant = None
                if self.m_en_passant_target is not None and self.m_en_passant_target == r2 * 8 + c2 - 1:
                    en_passant = self.m_en_passant_target
                if ChessGame.is_in_bound(r2, c2 - 1) and (required_destinations is None or r2 * 8 + c2 - 1 in required_destinations) and check_pinned:
                    piece2 = self.get_piece(r2 * 8 + c2 - 1)
                    if (en_passant is not None and not self.is_current_side_in_check(exclude=[position, position - 1], include=[r2 * 8 + c2 - 1]))or (en_passant is None and ChessPieces.side(piece2) != self.m_current_side and not ChessPieces.is_empty(piece2)):
                        if is_promoting:
                            res.append(Move(position, r2 * 8 + c2 - 1, promote_to=ChessPieces.piece(ChessPieces.PIECE_TYPE_KNIGHT, self.m_current_side), en_passant_target=en_passant))
                            res.append(Move(position, r2 * 8 + c2 - 1, promote_to=ChessPieces.piece(ChessPieces.PIECE_TYPE_BISHOP, self.m_current_side), en_passant_target=en_passant))
                            res.append(Move(position, r2 * 8 + c2 - 1, promote_to=ChessPieces.piece(ChessPieces.PIECE_TYPE_ROOK, self.m_current_side), en_passant_target=en_passant))
                            res.append(Move(position, r2 * 8 + c2 - 1, promote_to=ChessPieces.piece(ChessPieces.PIECE_TYPE_QUEEN, self.m_current_side), en_passant_target=en_passant))
                        else:
                            res.append(Move(position, r2 * 8 + c2 - 1, en_passant_target=en_passant))

                dr = advance_dir[1]
                dc = 1
                check_pinned = True
                if pinned_dir is not None:
                    val = dr * pinned_dir[0] - dc * pinned_dir[1]
                    check_pinned = val == 0
                en_passant = None
                if self.m_en_passant_target is not None and self.m_en_passant_target == r2 * 8 + c2 + 1:
                    en_passant = self.m_en_passant_target
                if ChessGame.is_in_bound(r2, c2 + 1) and (required_destinations is None or r2 * 8 + c2 + 1 in required_destinations) and check_pinned:
                    piece2 = self.get_piece(r2 * 8 + c2 + 1)
                    if (en_passant is not None and not self.is_current_side_in_check(exclude=[position, position + 1], include=[r2 * 8 + c2 + 1]))or (en_passant is None and ChessPieces.side(piece2) != self.m_current_side and not ChessPieces.is_empty(piece2)):
                        if is_promoting:
                            res.append(Move(position, r2 * 8 + c2 + 1, promote_to=ChessPieces.piece(ChessPieces.PIECE_TYPE_KNIGHT, self.m_current_side), en_passant_target=en_passant))
                            res.append(Move(position, r2 * 8 + c2 + 1, promote_to=ChessPieces.piece(ChessPieces.PIECE_TYPE_BISHOP, self.m_current_side), en_passant_target=en_passant))
                            res.append(Move(position, r2 * 8 + c2 + 1, promote_to=ChessPieces.piece(ChessPieces.PIECE_TYPE_ROOK, self.m_current_side), en_passant_target=en_passant))
                            res.append(Move(position, r2 * 8 + c2 + 1, promote_to=ChessPieces.piece(ChessPieces.PIECE_TYPE_QUEEN, self.m_current_side), en_passant_target=en_passant))
                        else:
                            res.append(Move(position, r2 * 8 + c2 + 1, en_passant_target=en_passant))

        return res

    # Make a move
    def move(self, 
        move
    ):
        r1 = move.start_position() // 8
        c1 = move.start_position() % 8
        r2 = move.end_position() // 8
        c2 = move.end_position() % 8

        piece1 = self.get_piece(move.start_position())
        piece2 = self.get_piece(move.end_position())

        self.m_board[move.start_position()] = ChessPieces._E
        self.m_board[move.end_position()]   = piece1

        res = piece2

        # Promotion
        if move.promote_to() is not None:
            self.m_board[move.end_position()] = move.promote_to()

        # En passant target
        if move.en_passant_target() is not None:
            r = move.en_passant_target() // 8 + (-1 if ChessPieces.side(piece1) else 1)
            c = move.en_passant_target() % 8

            res = self.get_piece(r * 8 + c)
            self.m_board[r * 8 + c] = ChessPieces._E

        # Castling
        piece1_type = ChessPieces.type(piece1)
        piece2_type = ChessPieces.type(piece2)
        if piece1_type == ChessPieces.PIECE_TYPE_KING:
            dc = c2 - c1
            if dc == 2:
                self.m_board[move.start_position() + 3] = ChessPieces._E
                self.m_board[move.start_position() + 1] = ChessPieces.WR if ChessPieces.side(piece1) else ChessPieces.BR
            elif dc == -2:
                self.m_board[move.start_position() - 4] = ChessPieces._E
                self.m_board[move.start_position() - 1] = ChessPieces.WR if ChessPieces.side(piece1) else ChessPieces.BR

        # Remove castling rights
        if self.m_current_side:
            if piece1_type == ChessPieces.PIECE_TYPE_KING:
                self.m_castle_WK = False
                self.m_castle_WQ = False
            elif piece1_type == ChessPieces.PIECE_TYPE_ROOK:
                if c1 == 7:
                    self.m_castle_WK = False
                elif c1 == 0:
                    self.m_castle_WQ = False

            if piece2_type == ChessPieces.PIECE_TYPE_ROOK:
                if c2 == 7:
                    self.m_castle_BK = False
                elif c2 == 0:
                    self.m_castle_BQ = False
        else:
            if piece1_type == ChessPieces.PIECE_TYPE_KING:
                self.m_castle_BK = False
                self.m_castle_BQ = False
            elif piece1_type == ChessPieces.PIECE_TYPE_ROOK:
                if c1 == 7:
                    self.m_castle_BK = False
                elif c1 == 0:
                    self.m_castle_BQ = False

            if piece2_type == ChessPieces.PIECE_TYPE_ROOK:
                if c2 == 7:
                    self.m_castle_WK = False
                elif c2 == 0:
                    self.m_castle_WQ = False

        # Update en passant target
        if piece1_type == ChessPieces.PIECE_TYPE_PAWN and c1 == c2 and (r2 - r1 == 2 or r2 - r1 == -2):
            self.m_en_passant_target = move.end_position() + (-8 if self.m_current_side else 8)
        else:
            self.m_en_passant_target = None

        # Reset halfmove clock if necessary
        pawn_advanced = piece1_type == ChessPieces.PIECE_TYPE_PAWN
        captured      = not ChessPieces.is_empty(res)
        if pawn_advanced or captured:
            self.m_halfmove = 0
        else:
            self.m_halfmove += 1

        # Increment fullmove count of black's move
        if not self.m_current_side:
            self.m_fullmove += 1

        # Switch side:
        self.m_current_side = not self.m_current_side

        return res

    # Evaluate the board
    def evaluate(self):
        res = 0.0
        for i in range(64):
            res += self.evaluate_piece(i)
        return res

    # Evaluate a piece
    # White piece gives positive values
    # Black piece gives negative values
    def evaluate_piece(self, 
        position
    ):
        piece = self.get_piece(position)
        return ChessPieces.value(piece, position) * (1.0 if self.is_side_to_move(piece) else -1.0)
