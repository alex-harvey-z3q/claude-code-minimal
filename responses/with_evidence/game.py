import time
from board import Board

class MinesweeperGame:
    def __init__(self, width: int, height: int, mine_count: int):
        self.board = Board(width, height, mine_count)
        self.game_over = False
        self.start_time = None
        
    def play_move(self, x: int, y: int, action: str) -> bool:
        """Handle a move and return False if game is lost"""
        if self.game_over:
            return True
            
        if not self.start_time:
            self.start_time = time.time()
            
        if action == 'f':
            self.board.toggle_flag(x, y)
            return True
            
        if not self.board.mines_placed:
            self.board.place_mines(x, y)
            
        if not self.board.reveal(x, y):
            self.game_over = True
            return False
            
        if self.board.is_win():
            self.game_over = True
            
        return True

    def get_display_board(self) -> str:
        """Return string representation of current board state"""
        from constants import MINE, FLAG, UNOPENED, EMPTY
        
        result = []
        result.append('   ' + ' '.join(str(i % 10) for i in range(self.board.width)))
        result.append('   ' + '-' * (2 * self.board.width - 1))
        
        for y in range(self.board.height):
            row = f'{y:2d}|'
            for x in range(self.board.width):
                cell = self.board.cells[y][x]
                if cell.is_flagged:
                    row += FLAG
                elif not cell.is_opened:
                    row += UNOPENED
                elif cell.is_mine:
                    row += MINE
                elif cell.adjacent_mines == 0:
                    row += EMPTY
                else:
                    row += str(cell.adjacent_mines)
                row += ' '
            result.append(row)
        return '\n'.join(result)
