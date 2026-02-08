import random
from typing import List, Tuple, Set

class Cell:
    def __init__(self):
        self.is_mine = False
        self.is_revealed = False
        self.is_flagged = False
        self.adjacent_mines = 0

class Minesweeper:
    def __init__(self, width: int, height: int, mines: int):
        self.width = width
        self.height = height
        self.mines = min(mines, width * height)
        self.board = [[Cell() for _ in range(width)] for _ in range(height)]
        self.game_over = False
        self.won = False
        self._place_mines()
        self._calculate_adjacent()

    def _place_mines(self) -> None:
        positions = random.sample([(x, y) for x in range(self.height) 
                                for y in range(self.width)], self.mines)
        for x, y in positions:
            self.board[x][y].is_mine = True

    def _calculate_adjacent(self) -> None:
        for x in range(self.height):
            for y in range(self.width):
                if not self.board[x][y].is_mine:
                    count = 0
                    for dx, dy in self._get_neighbors(x, y):
                        if self.board[dx][dy].is_mine:
                            count += 1
                    self.board[x][y].adjacent_mines = count

    def _get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.height and 0 <= new_y < self.width:
                    neighbors.append((new_x, new_y))
        return neighbors

    def reveal(self, x: int, y: int) -> None:
        if not (0 <= x < self.height and 0 <= y < self.width):
            return
        cell = self.board[x][y]
        if cell.is_revealed or cell.is_flagged:
            return
        cell.is_revealed = True
        
        if cell.is_mine:
            self.game_over = True
            return

        if cell.adjacent_mines == 0:
            for nx, ny in self._get_neighbors(x, y):
                self.reveal(nx, ny)

        self._check_win()

    def toggle_flag(self, x: int, y: int) -> None:
        if not (0 <= x < self.height and 0 <= y < self.width):
            return
        cell = self.board[x][y]
        if not cell.is_revealed:
            cell.is_flagged = not cell.is_flagged

    def _check_win(self) -> None:
        for x in range(self.height):
            for y in range(self.width):
                cell = self.board[x][y]
                if not cell.is_mine and not cell.is_revealed:
                    return
        self.won = True
        self.game_over = True

    def display(self) -> None:
        print("  " + " ".join(str(i) for i in range(self.width)))
        for x in range(self.height):
            print(f"{x} ", end="")
            for y in range(self.width):
                cell = self.board[x][y]
                if cell.is_flagged:
                    print("F", end=" ")
                elif not cell.is_revealed:
                    print(".", end=" ")
                elif cell.is_mine:
                    print("*", end=" ")
                elif cell.adjacent_mines == 0:
                    print(" ", end=" ")
                else:
                    print(cell.adjacent_mines, end=" ")
            print()

def play_game():
    while True:
        try:
            width = int(input("Enter board width (5-30): "))
            height = int(input("Enter board height (5-30): "))
            mines = int(input("Enter number of mines: "))
            if not (5 <= width <= 30 and 5 <= height <= 30 and mines > 0):
                raise ValueError
            break
        except ValueError:
            print("Invalid input. Please try again.")

    game = Minesweeper(width, height, mines)
    
    while not game.game_over:
        game.display()
        try:
            action = input("Enter action (r/f) row col (e.g. 'r 3 4' or 'f 2 1'): ").split()
            if len(action) != 3:
                raise ValueError
            cmd, x, y = action[0].lower(), int(action[1]), int(action[2])
            if cmd == 'r':
                game.reveal(x, y)
            elif cmd == 'f':
                game.toggle_flag(x, y)
            else:
                raise ValueError
        except (ValueError, IndexError):
            print("Invalid input. Format: [r/f] row col")
            continue

    game.display()
    if game.won:
        print("Congratulations! You won!")
    else:
        print("Game Over! You hit a mine!")

    return input("Play again? (y/n): ").lower().startswith('y')

if __name__ == "__main__":
    while play_game():
        pass
