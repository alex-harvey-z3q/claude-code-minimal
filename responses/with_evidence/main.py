from game import MinesweeperGame
from constants import BEGINNER, INTERMEDIATE, EXPERT

def get_difficulty() -> tuple:
    while True:
        print("\nSelect difficulty:")
        print("1. Beginner (9x9, 10 mines)")
        print("2. Intermediate (16x16, 40 mines)")
        print("3. Expert (30x16, 99 mines)")
        print("4. Custom")
        
        choice = input("Choice (1-4): ").strip()
        
        if choice == '1':
            return BEGINNER
        elif choice == '2':
            return INTERMEDIATE
        elif choice == '3':
            return EXPERT
        elif choice == '4':
            try:
                width = int(input("Width: "))
                height = int(input("Height: "))
                mines = int(input("Mines: "))
                if width > 0 and height > 0 and mines > 0 and mines < width * height:
                    return (width, height, mines)
            except ValueError:
                pass
        print("Invalid choice, try again")

def main():
    while True:
        width, height, mines = get_difficulty()
        game = MinesweeperGame(width, height, mines)
        
        while not game.game_over:
            print("\n" + game.get_display_board())
            print("\nEnter move as 'x y action'")
            print("Actions: r (reveal), f (flag)")
            
            try:
                x, y, action = input("> ").strip().split()
                x, y = int(x), int(y)
                if action not in ['r', 'f']:
                    raise ValueError
                
                if not game.play_move(x, y, action):
                    print("\nBOOM! Game Over!")
                    print(game.get_display_board())
                    break
                    
                if game.game_over:
                    print("\nCongratulations! You won!")
                    print(game.get_display_board())
                    
            except (ValueError, IndexError):
                print("Invalid input! Format: x y action")
                
        if input("\nPlay again? (y/n): ").lower() != 'y':
            break

if __name__ == "__main__":
    main()
