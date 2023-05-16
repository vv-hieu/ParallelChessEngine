import pygame
import math
import asyncio
from ChessCore import *
from ChessEngine import *

PROPERTY_WINDOW_WIDTH  = 960
PROPERTY_WINDOW_HEIGHT = 640
PROPERTY_WINDOW_ICON   = pygame.image.load('icons/white_king.png')

PROPERTY_BOARD_SIZE              = min(PROPERTY_WINDOW_WIDTH, PROPERTY_WINDOW_HEIGHT)
PROPERTY_TILE_SIZE               = PROPERTY_BOARD_SIZE / 8
PROPERTY_PROMOTE_ICON_SIZE       = PROPERTY_TILE_SIZE * 1.5
PROPERTY_PROMOTE_ICON_SPACING    = PROPERTY_BOARD_SIZE * 0.01
PROPERTY_PROMOTE_WINDOW_SIZE     = PROPERTY_PROMOTE_ICON_SIZE * 2.0 + PROPERTY_PROMOTE_ICON_SPACING * 3.0

PROPERTY_COLOR_DARK              = (0.60 * 255, 0.45 * 255, 0.20 * 255)
PROPERTY_COLOR_LIGHT             = (0.80 * 255, 0.65 * 255, 0.40 * 255)
PROPERTY_COLOR_SELECTED_DARK     = (0.25 * 255, 0.55 * 255, 0.40 * 255)
PROPERTY_COLOR_SELECTED_LIGHT    = (0.35 * 255, 0.75 * 255, 0.50 * 255)
PROPERTY_COLOR_DESTINATION_DARK  = (0.60 * 255, 0.20 * 255, 0.20 * 255)
PROPERTY_COLOR_DESTINATION_LIGHT = (0.80 * 255, 0.30 * 255, 0.30 * 255)
PROPERTY_COLOR_RECENT_DARK       = (0.25 * 255, 0.45 * 255, 0.60 * 255)
PROPERTY_COLOR_RECENT_LIGHT      = (0.40 * 255, 0.65 * 255, 0.85 * 255)

PROPERTY_CHESSENGINE_VERSION      = 11
PROPERTY_CHESSENGINE_SEARCH_DEPTH = 6

PIECE_ICON_KING_WHITE   = pygame.transform.smoothscale(pygame.image.load('icons/white_king.png'), (PROPERTY_TILE_SIZE, PROPERTY_TILE_SIZE))
PIECE_ICON_KING_BLACK   = pygame.transform.smoothscale(pygame.image.load('icons/black_king.png'), (PROPERTY_TILE_SIZE, PROPERTY_TILE_SIZE))
PIECE_ICON_PAWN_WHITE   = pygame.transform.smoothscale(pygame.image.load('icons/white_pawn.png'), (PROPERTY_TILE_SIZE, PROPERTY_TILE_SIZE))
PIECE_ICON_PAWN_BLACK   = pygame.transform.smoothscale(pygame.image.load('icons/black_pawn.png'), (PROPERTY_TILE_SIZE, PROPERTY_TILE_SIZE))
PIECE_ICON_KNIGHT_WHITE = pygame.transform.smoothscale(pygame.image.load('icons/white_knight.png'), (PROPERTY_TILE_SIZE, PROPERTY_TILE_SIZE))
PIECE_ICON_KNIGHT_BLACK = pygame.transform.smoothscale(pygame.image.load('icons/black_knight.png'), (PROPERTY_TILE_SIZE, PROPERTY_TILE_SIZE))
PIECE_ICON_BISHOP_WHITE = pygame.transform.smoothscale(pygame.image.load('icons/white_bishop.png'), (PROPERTY_TILE_SIZE, PROPERTY_TILE_SIZE))
PIECE_ICON_BISHOP_BLACK = pygame.transform.smoothscale(pygame.image.load('icons/black_bishop.png'), (PROPERTY_TILE_SIZE, PROPERTY_TILE_SIZE))
PIECE_ICON_ROOK_WHITE   = pygame.transform.smoothscale(pygame.image.load('icons/white_rook.png'), (PROPERTY_TILE_SIZE, PROPERTY_TILE_SIZE))
PIECE_ICON_ROOK_BLACK   = pygame.transform.smoothscale(pygame.image.load('icons/black_rook.png'), (PROPERTY_TILE_SIZE, PROPERTY_TILE_SIZE))
PIECE_ICON_QUEEN_WHITE  = pygame.transform.smoothscale(pygame.image.load('icons/white_queen.png'), (PROPERTY_TILE_SIZE, PROPERTY_TILE_SIZE))
PIECE_ICON_QUEEN_BLACK  = pygame.transform.smoothscale(pygame.image.load('icons/black_queen.png'), (PROPERTY_TILE_SIZE, PROPERTY_TILE_SIZE))

PIECE_ICON_WHITES = [None, PIECE_ICON_KING_WHITE, PIECE_ICON_PAWN_WHITE, PIECE_ICON_KNIGHT_WHITE, 
            PIECE_ICON_BISHOP_WHITE, PIECE_ICON_ROOK_WHITE, PIECE_ICON_QUEEN_WHITE]

PIECE_ICON_BLACKS = [None, PIECE_ICON_KING_BLACK, PIECE_ICON_PAWN_BLACK, PIECE_ICON_KNIGHT_BLACK, 
            PIECE_ICON_BISHOP_BLACK, PIECE_ICON_ROOK_BLACK, PIECE_ICON_QUEEN_BLACK]

PROMOTE_PIECE_ICON_WHITE_KNIGHT = pygame.transform.smoothscale(pygame.image.load('icons/white_knight.png'), (PROPERTY_PROMOTE_ICON_SIZE, PROPERTY_PROMOTE_ICON_SIZE))
PROMOTE_PIECE_ICON_BLACK_KNIGHT = pygame.transform.smoothscale(pygame.image.load('icons/black_knight.png'), (PROPERTY_PROMOTE_ICON_SIZE, PROPERTY_PROMOTE_ICON_SIZE))
PROMOTE_PIECE_ICON_WHITE_BISHOP = pygame.transform.smoothscale(pygame.image.load('icons/white_bishop.png'), (PROPERTY_PROMOTE_ICON_SIZE, PROPERTY_PROMOTE_ICON_SIZE))
PROMOTE_PIECE_ICON_BLACK_BISHOP = pygame.transform.smoothscale(pygame.image.load('icons/black_bishop.png'), (PROPERTY_PROMOTE_ICON_SIZE, PROPERTY_PROMOTE_ICON_SIZE))
PROMOTE_PIECE_ICON_WHITE_ROOK   = pygame.transform.smoothscale(pygame.image.load('icons/white_rook.png'), (PROPERTY_PROMOTE_ICON_SIZE, PROPERTY_PROMOTE_ICON_SIZE))
PROMOTE_PIECE_ICON_BLACK_ROOK   = pygame.transform.smoothscale(pygame.image.load('icons/black_rook.png'), (PROPERTY_PROMOTE_ICON_SIZE, PROPERTY_PROMOTE_ICON_SIZE))
PROMOTE_PIECE_ICON_WHITE_QUEEN  = pygame.transform.smoothscale(pygame.image.load('icons/white_queen.png'), (PROPERTY_PROMOTE_ICON_SIZE, PROPERTY_PROMOTE_ICON_SIZE))
PROMOTE_PIECE_ICON_BLACK_QUEEN  = pygame.transform.smoothscale(pygame.image.load('icons/black_queen.png'), (PROPERTY_PROMOTE_ICON_SIZE, PROPERTY_PROMOTE_ICON_SIZE))

PROMOTE_PIECE_ICON_WHITES = [PROMOTE_PIECE_ICON_WHITE_KNIGHT, PROMOTE_PIECE_ICON_WHITE_BISHOP, PROMOTE_PIECE_ICON_WHITE_ROOK, PROMOTE_PIECE_ICON_WHITE_QUEEN]

PROMOTE_PIECE_ICON_BLACKS = [PROMOTE_PIECE_ICON_BLACK_KNIGHT, PROMOTE_PIECE_ICON_BLACK_BISHOP, PROMOTE_PIECE_ICON_BLACK_ROOK, PROMOTE_PIECE_ICON_BLACK_QUEEN]

pygame.init()
pygame.display.set_caption('Chess')
pygame.display.set_icon(PROPERTY_WINDOW_ICON)

def get_piece_icon(piece):
    if ChessPieces.side(piece):
        return PIECE_ICON_WHITES[ChessPieces.type(piece)]
    else:
        return PIECE_ICON_BLACKS[ChessPieces.type(piece)]

class IScene:
    def __init__(self):
        self.m_next_scene = None

    def draw(self):
        pass

    def update(self):
        pass

    def handle_event(self, event):
        pass

    def switch_scene(self, new_scene):
        self.m_next_scene = new_scene

    def next_scene(self):
        return self.m_next_scene

class MainMenuScene(IScene):
    def __init__(self):
        super().__init__()

        self.m_font_size = int(PROPERTY_TILE_SIZE * 0.75)
        self.m_font = pygame.font.Font('freesansbold.ttf', self.m_font_size)

        self.m_button_0 = self.m_font.render('PLAY AS WHITE', True, (0, 0, 0), (255, 255, 255))
        self.m_button_0_rect = self.m_button_0.get_rect()
        self.m_button_0_rect.center = (
            0.5 * PROPERTY_WINDOW_WIDTH,
            0.5 * (PROPERTY_WINDOW_HEIGHT - self.m_font_size),
        )

        self.m_button_1 = self.m_font.render('PLAY AS BLACK', True, (255, 255, 255), (0, 0, 0))
        self.m_button_1_rect = self.m_button_0.get_rect()
        self.m_button_1_rect.center = (
            0.5 * PROPERTY_WINDOW_WIDTH,
            0.5 * (PROPERTY_WINDOW_HEIGHT + self.m_font_size),
        )

    def draw(self, screen):
        for row in range(8):
            for col in range(8):
                r = 7 - row
                c = col

                rect = pygame.Rect(
                    c * PROPERTY_TILE_SIZE + 0.5 * (PROPERTY_WINDOW_WIDTH - PROPERTY_BOARD_SIZE), 
                    r * PROPERTY_TILE_SIZE + 0.5 * (PROPERTY_WINDOW_HEIGHT - PROPERTY_BOARD_SIZE), 
                    PROPERTY_TILE_SIZE, 
                    PROPERTY_TILE_SIZE)

                color = PROPERTY_COLOR_DARK if (row + col) % 2 == 0 else PROPERTY_COLOR_LIGHT

                pygame.draw.rect(screen, color, rect)

        screen.blit(self.m_button_0, self.m_button_0_rect)
        screen.blit(self.m_button_1, self.m_button_1_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            if self.m_button_0_rect.collidepoint(x, y):
                self.switch_scene(GameScene(player_side = True))
            elif self.m_button_1_rect.collidepoint(x, y):
                self.switch_scene(GameScene(player_side = False))

class GameScene(IScene):
    def __init__(self, player_side = True):
        super().__init__()

        self.m_chess_game   = ChessGame()
        self.m_player_side  = player_side
        self.m_recent_move  = None
        self.m_moves        = self.m_chess_game.generate_all_moves()
        self.m_selected     = -1
        self.m_possibles    = []
        self.m_promote_move = None

    def move(self, move):
        if move is not None:
            self.m_chess_game.move(move)
            self.m_recent_move  = move
            self.m_moves        = self.m_chess_game.generate_all_moves()
            self.m_selected     = -1
            self.m_possibles    = []
            self.m_promote_move = None

            if len(self.m_moves) == 0:
                if not self.m_chess_game.is_current_side_in_check():
                    self.switch_scene(ResultScene(self, 0))
                else:
                    self.switch_scene(ResultScene(self, -1 if self.m_chess_game.side_to_move() else 1))

    def select_piece(self, position):
        if self.m_chess_game.side_to_move() != self.m_player_side:
            self.m_selected  = -1
            self.m_possibles = []
            return

        if position == self.m_selected:
            self.m_selected  = -1
            self.m_possibles = []
        else:
            piece = self.m_chess_game.get_piece(position)
            if (ChessPieces.side(piece) == self.m_player_side) and not ChessPieces.is_empty(piece):
                self.m_selected = position
                self.m_possibles = [move for move in self.m_moves if move.start_position() == position]

    def draw(self, screen):
        for row in range(8):
            for col in range(8):
                position = row * 8 + col

                r = 7 - row if self.m_player_side else row
                c = col

                rect = pygame.Rect(
                    c * PROPERTY_TILE_SIZE + 0.5 * (PROPERTY_WINDOW_WIDTH - PROPERTY_BOARD_SIZE), 
                    r * PROPERTY_TILE_SIZE + 0.5 * (PROPERTY_WINDOW_HEIGHT - PROPERTY_BOARD_SIZE), 
                    PROPERTY_TILE_SIZE, 
                    PROPERTY_TILE_SIZE)

                if self.m_recent_move is not None and \
                    (position == self.m_recent_move.start_position() or position == self.m_recent_move.end_position()):
                    color = PROPERTY_COLOR_RECENT_DARK if (row + col) % 2 == 0 else PROPERTY_COLOR_RECENT_LIGHT
                elif position == self.m_selected:
                    color = PROPERTY_COLOR_SELECTED_DARK if (row + col) % 2 == 0 else PROPERTY_COLOR_SELECTED_LIGHT
                elif position in [move.end_position() for move in self.m_possibles]:
                    color = PROPERTY_COLOR_DESTINATION_DARK if (row + col) % 2 == 0 else PROPERTY_COLOR_DESTINATION_LIGHT
                else:
                    color = PROPERTY_COLOR_DARK if (row + col) % 2 == 0 else PROPERTY_COLOR_LIGHT

                pygame.draw.rect(screen, color, rect)

                piece = self.m_chess_game.get_piece(position)
                if not ChessPieces.is_empty(piece):
                    screen.blit(get_piece_icon(piece), rect)

        if self.m_promote_move is not None:
            rect = pygame.Rect(
                0.5 * (PROPERTY_WINDOW_WIDTH - PROPERTY_PROMOTE_WINDOW_SIZE), 
                0.5 * (PROPERTY_WINDOW_HEIGHT - PROPERTY_PROMOTE_WINDOW_SIZE),
                PROPERTY_PROMOTE_WINDOW_SIZE, 
                PROPERTY_PROMOTE_WINDOW_SIZE)
            pygame.draw.rect(screen, (0, 0, 0), rect)

            promote_icons = PROMOTE_PIECE_ICON_WHITES if self.m_player_side else PROMOTE_PIECE_ICON_BLACKS

            for row in range(2):
                for col in range(2):
                    index = row * 2 + col

                    rect = pygame.Rect(
                        col * (PROPERTY_PROMOTE_ICON_SIZE + PROPERTY_PROMOTE_ICON_SPACING) + 0.5 * (PROPERTY_WINDOW_WIDTH -     PROPERTY_PROMOTE_WINDOW_SIZE) + PROPERTY_PROMOTE_ICON_SPACING, 
                        row * (PROPERTY_PROMOTE_ICON_SIZE + PROPERTY_PROMOTE_ICON_SPACING) + 0.5 * (PROPERTY_WINDOW_HEIGHT -    PROPERTY_PROMOTE_WINDOW_SIZE) + PROPERTY_PROMOTE_ICON_SPACING,
                        PROPERTY_PROMOTE_ICON_SIZE, 
                        PROPERTY_PROMOTE_ICON_SIZE)

                    pygame.draw.rect(screen, PROPERTY_COLOR_DARK if (row + col) % 2 == 0 else PROPERTY_COLOR_LIGHT, rect)
                    screen.blit(promote_icons[index], rect)

    def update(self):
        if self.m_chess_game.side_to_move() != self.m_player_side:
            move, _ = find_move(self.m_chess_game, PROPERTY_CHESSENGINE_SEARCH_DEPTH, PROPERTY_CHESSENGINE_VERSION)
            self.move(move)
                
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            if self.m_promote_move is not None:
                x -= 0.5 * (PROPERTY_WINDOW_WIDTH - PROPERTY_PROMOTE_WINDOW_SIZE) + PROPERTY_PROMOTE_ICON_SPACING
                y -= 0.5 * (PROPERTY_WINDOW_HEIGHT - PROPERTY_PROMOTE_WINDOW_SIZE) + PROPERTY_PROMOTE_ICON_SPACING

                promote_to = [
                    ChessPieces.piece(ChessPieces.PIECE_TYPE_KNIGHT, self.m_player_side),
                    ChessPieces.piece(ChessPieces.PIECE_TYPE_BISHOP, self.m_player_side),
                    ChessPieces.piece(ChessPieces.PIECE_TYPE_ROOK  , self.m_player_side),
                    ChessPieces.piece(ChessPieces.PIECE_TYPE_QUEEN , self.m_player_side)
                ]

                for row in range(2):
                    for col in range(2):
                        xi = col * (PROPERTY_PROMOTE_ICON_SIZE + PROPERTY_PROMOTE_ICON_SPACING)
                        yi = row * (PROPERTY_PROMOTE_ICON_SIZE + PROPERTY_PROMOTE_ICON_SPACING)

                        if xi <= x <= xi + PROPERTY_PROMOTE_ICON_SIZE and \
                            yi <= y <= yi + PROPERTY_PROMOTE_ICON_SIZE:
                            self.m_promote_move.m_promote_to = promote_to[row * 2 + col]
                            self.move(self.m_promote_move)
                            break

            else:
                x -= 0.5 * (PROPERTY_WINDOW_WIDTH - PROPERTY_BOARD_SIZE)
                y -= 0.5 * (PROPERTY_WINDOW_HEIGHT - PROPERTY_BOARD_SIZE)
                r = 7 - math.floor(y / PROPERTY_TILE_SIZE) if self.m_player_side else math.floor(y / PROPERTY_TILE_SIZE)
                c = math.floor(x / PROPERTY_TILE_SIZE)
                if 0 <= r < 8 and 0 <= c < 8:
                    position = r * 8 + c
                    if position in [move.end_position() for move in self.m_possibles]:
                        for move in self.m_possibles:
                            if move.start_position() == self.m_selected and move.end_position() == position:
                                if move.promote_to() is not None:
                                    self.m_promote_move = move
                                else:
                                    self.move(move)
                                break
                    else:
                        self.select_piece(position)
     
class ResultScene(IScene):
    def __init__(self, game_scene, result):
        super().__init__()

        self.m_game_scene = game_scene

        self.m_font_size = int(PROPERTY_TILE_SIZE * 0.5)
        self.m_font = pygame.font.Font('freesansbold.ttf', self.m_font_size)

        if result == 0:  # Draw
            self.m_title = self.m_font.render('DRAW', True, (64, 64, 64), (128, 128, 128))
        elif result > 0: # White wins
            self.m_title = self.m_font.render('WHITE WINS', True, (0, 0, 0), (255, 255, 255))
        elif result < 0: # Black wins
            self.m_title = self.m_font.render('BLACK WINS', True, (255, 255, 255), (0, 0, 0))

        self.m_title_rect = self.m_title.get_rect()
        self.m_title_rect.center = (
            0.5 * PROPERTY_WINDOW_WIDTH,
            PROPERTY_BOARD_SIZE - self.m_title_rect.height * 0.5,
        )

    def draw(self, screen):
        if self.m_game_scene is not None and issubclass(type(self.m_game_scene), IScene):
            self.m_game_scene.draw(screen)

        screen.blit(self.m_title, self.m_title_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.switch_scene(MainMenuScene())

class Game:
    def __init__(self):
        self.m_screen  = pygame.display.set_mode((PROPERTY_WINDOW_WIDTH, PROPERTY_WINDOW_HEIGHT))
        self.m_running = True
        self.m_scene   = MainMenuScene()

    def update(self):
        if self.m_scene is not None and issubclass(type(self.m_scene), IScene):
            self.m_scene.update()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.m_running = False
                return
            if self.m_scene is not None and issubclass(type(self.m_scene), IScene):
                self.m_scene.handle_event(event)

    def draw(self):
        if self.m_scene is not None and issubclass(type(self.m_scene), IScene):
            self.m_scene.draw(self.m_screen)
        pygame.display.flip()

    def handle_scene_switching(self):
        if self.m_scene is not None and issubclass(type(self.m_scene), IScene):
            next_scene = self.m_scene.next_scene()
            if next_scene is not None:
                self.m_scene = next_scene

    async def run(self):
        while self.m_running:
            self.update()
            self.handle_events()
            self.draw()
            self.handle_scene_switching()

        await asyncio.sleep(0.0)

if __name__ == '__main__':
    game = Game()
    asyncio.run(game.run())
