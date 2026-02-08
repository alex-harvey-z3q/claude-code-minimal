import unittest
from board import Board
from game import MinesweeperGame

class TestMinesweeper(unittest.TestCase):
    def test_board_creation(self):
        board = Board(9, 9, 10)
        self.assertEqual(board.width, 9)
        self.assertEqual(board.height, 9)
        self.assertEqual(board.mine_count, 10)
        
    def test_mine_placement(self):
        board = Board(9, 9, 10)
        board.place_mines(4, 4)
        
        # Count total mines
        mine_count = sum(cell.is_mine 
                        for row in board.cells 
                        for cell in row)
        self.assertEqual(mine_count, 10)
        
        # Check first click safety
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                self.assertFalse(board.cells[4+dy][4+dx].is_mine)
                
    def test_win_condition(self):
        game = MinesweeperGame(3, 3, 1)
        game.board.place_mines(0, 0)
        
        # Reveal all non-mine cells
        for y in range(3):
            for x in range(3):
                if not game.board.cells[y][x].is_mine:
                    game.play_move(x, y, 'r')
                    
        self.assertTrue(game.board.is_win())
        
    def test_flag_toggle(self):
        board = Board(9, 9, 10)
        board.toggle_flag(0, 0)
        self.assertTrue(board.cells[0][0].is_flagged)
        board.toggle_flag(0, 0)
        self.assertFalse(board.cells[0][0].is_flagged)

if __name__ == '__main__':
    unittest.main()
