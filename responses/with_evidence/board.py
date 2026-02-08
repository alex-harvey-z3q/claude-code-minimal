from dataclasses import dataclass
from typing import List, Set, Tuple
import random

@dataclass
class Cell:
    is_mine: bool = False
    is_opened: bool = False
    is_flagged: bool = False
    adjacent_mines: int = 0

class Board:
    def __init__(self, width: int, height: int, mine_count: int):
        self.width = width
        self.height = height
        self.mine_count = mine_count
        self.cells = [[Cell() for _ in range(width)] for _ in range(height)]
        self.mines_placed = False
        self.opened_cells = 0
        
    def place_mines(self, first_x: int, first_y: int) -> None:
        """Place mines avoiding first click position and adjacents"""
        safe_cells = {(first_x + dx, first_y + dy) 
                     for dx in [-1,0,1] for dy in [-1,0,1]
                     if 0 <= first_x + dx < self.width and 0 <= first_y + dy < self.height}
        
        all_positions = [(x, y) for x in range(self.width) 
                        for y in range(self.height) 
                        if (x, y) not in safe_cells]
        
        mine_positions = random.sample(all_positions, self.mine_count)
        
        for x, y in mine_positions:
            self.cells[y][x].is_mine = True
            
        # Calculate adjacent mine counts
        for y in range(self.height):
            for x in range(self.width):
                if not self.cells[y][x].is_mine:
                    self.cells[y][x].adjacent_mines = self._count_adjacent_mines(x, y)
        
        self.mines_placed = True

    def _count_adjacent_mines(self, x: int, y: int) -> int:
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < self.width and 
                    0 <= new_y < self.height and 
                    self.cells[new_y][new_x].is_mine):
                    count += 1
        return count

    def reveal(self, x: int, y: int) -> bool:
        """Reveal cell and return False if mine hit"""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True
        
        cell = self.cells[y][x]
        if cell.is_opened or cell.is_flagged:
            return True
            
        cell.is_opened = True
        self.opened_cells += 1
        
        if cell.is_mine:
            return False
            
        if cell.adjacent_mines == 0:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    self.reveal(x + dx, y + dy)
        return True

    def toggle_flag(self, x: int, y: int) -> None:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        cell = self.cells[y][x]
        if not cell.is_opened:
            cell.is_flagged = not cell.is_flagged

    def is_win(self) -> bool:
        return self.opened_cells == (self.width * self.height - self.mine_count)
