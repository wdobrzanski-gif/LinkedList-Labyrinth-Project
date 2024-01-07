"""
Program to try to escape from a customize labyrinth.

Author:
Will Dobrzanski - wdobrzanski@sandiego.edu
"""
import labyrinth
from labyrinth import MazeCell, Item

# Change the following variable (YOUR_NAME) to contain your FULL name.

"""
!!!WARNING!!!

Once you've set set this constant and started exploring your maze,
do NOT edit the value of YOUR_NAME. Changing YOUR_NAME will change which
maze you get back, which might invalidate all your hard work!
"""

YOUR_NAME = "Will Dobrzanski"

# Change these following two constants to contain the paths out of your mazes.
# You'll need to use the debugger to help you out!

PATH_OUT_OF_MAZE        = "SSENEEWWNEEWWSSSENE"
PATH_OUT_OF_TWISTY_MAZE = "WWNENSWWNNNNW"

def is_path_to_freedom(start: MazeCell, moves: str) -> bool:
    """
    Given a location in a maze, returns whether the given sequence of
    steps will let you escape the maze. The steps should be given as
    a string made from N, S, E, and W for north/south/east/west without
    spaces or other punctuation symbols, such as "WESNNNS"

    To escape the maze, you need to find the Potion, the Spellbook, and
    the Wand. You can only take steps in the four cardinal directions,
    and you can't move in directions that don't exist in the maze.

    Precondition: start is not None

    Args:
        start (MazeCell): The start location in the maze.
        moves (str): The sequence of moves.

    Raises:
        ValueError: If <moves> contains any character other than N, S, E, or W

    Returns:
        (bool) Whether that sequence of moves picks up the needed items
               without making nay illegal moves.
    """

    if start is None:
        raise AssertionError("Start cannot be None")

    if not all(move in 'NSEW' for move in moves):
        raise ValueError("Invalid move character in the sequence")

    current_cell = start
    gathered_items = set()

    for move in moves:
        if move == 'N':
            if current_cell.north:
                current_cell = current_cell.north
            else:
                return False
        elif move == 'S':
            if current_cell.south:
                current_cell = current_cell.south
            else:
                return False
        elif move == 'E':
            if current_cell.east:
                current_cell = current_cell.east
            else:
                return False
        elif move == 'W':
            if current_cell.west:
                current_cell = current_cell.west
            else:
                return False

        if current_cell.whats_here != Item.NOTHING:
            gathered_items.add(current_cell.whats_here)

    return (
        Item.POTION in gathered_items and
        Item.SPELLBOOK in gathered_items and
        Item.WAND in gathered_items
    )




def main() -> None:
    """ Generates two types of labyrinths and checks whether the user has
    successfully found the path out of them.

    DO NOT MODIFY THIS CODE IN ANY WAY!!!
    """
    start_location = labyrinth.maze_for(YOUR_NAME)

    print("Ready to explore the labyrinth!")
    # Set a breakpoint here to explore your personal labyrinth!

    if is_path_to_freedom(start_location, PATH_OUT_OF_MAZE):
        print("Congratulations! You've found a way out of your labyrinth.")
    else:
        print("Sorry, but you're still stuck in your labyrinth.")


    twisty_start_location = labyrinth.twisty_maze_for(YOUR_NAME)

    print("Ready to explore the twisty labyrinth!")
    # Set a breakpoint here to explore your personal TWISTY labyrinth!

    if is_path_to_freedom(twisty_start_location, PATH_OUT_OF_TWISTY_MAZE):
        print("Congratulations! You've found a way out of your twisty labyrinth.")
    else:
        print("Sorry, but you're still stuck in your twisty labyrinth.")



if __name__ == "__main__":
    main()
