import unittest
from minesweeper import Minesweeper, Cell

class TestMinesweeper(unittest.TestCase):
    def test_board_initialization(self):
        game = Minesweeper(5, 5, 5)
        self.assertEqual(len(game.board), 5)
        self.assertEqual(len(game.board[0]), 5)
        
        mine_count = sum(cell.is_mine 
                        for row in game.board 
                        for cell in row)
        self.assertEqual(mine_count, 5)

    def test_reveal_empty_cell(self):
        game = Minesweeper(5, 5, 0)
        game.reveal(0, 0)
        self.assertTrue(game.board[0][0].is_revealed)
        self.assertTrue(game.won)
        self.assertTrue(game.game_over)

    def test_reveal_mine(self):
        game = Minesweeper(5, 5, 25)  # All mines
        game.reveal(0, 0)
        self.assertTrue(game.game_over)
        self.assertFalse(game.won)

    def test_flag_cell(self):
        game = Minesweeper(5, 5, 5)
        game.toggle_flag(0, 0)
        self.assertTrue(game.board[0][0].is_flagged)
        game.toggle_flag(0, 0)
        self.assertFalse(game.board[0][0].is_flagged)

    def test_win_condition(self):
        game = Minesweeper(2, 2, 1)
        mine_pos = None
        for i in range(2):
            for j in range(2):
                if game.board[i][j].is_mine:
                    mine_pos = (i, j)
                else:
                    game.reveal(i, j)
        self.assertTrue(game.won)
        self.assertTrue(game.game_over)

if __name__ == '__main__':
    unittest.main()
