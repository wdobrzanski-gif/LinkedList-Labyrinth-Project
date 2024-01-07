"""
Unit test code for main.is_path_to_freedom.
"""

from labyrinth import MazeCell, Item
from main import is_path_to_freedom
import pytest


def linear_maze() -> MazeCell:
    """ Creates a linear maze. """
    cell0 = MazeCell() # starting cell
    cell1 = MazeCell()
    cell2 = MazeCell()
    cell3 = MazeCell()

    cell1.whats_here = Item.SPELLBOOK
    cell2.whats_here = Item.POTION
    cell3.whats_here = Item.WAND

    cell0.east = cell1
    cell1.west = cell0
    cell1.east = cell2
    cell2.west = cell1
    cell2.east = cell3
    cell3.west = cell2

    return cell0

def square_maze() -> MazeCell:
    """ Creates a 2x2 maze """
    cell0 = MazeCell() # starting cell
    cell1 = MazeCell()
    cell2 = MazeCell()
    cell3 = MazeCell()

    cell1.whats_here = Item.POTION
    cell2.whats_here = Item.SPELLBOOK
    cell3.whats_here = Item.WAND

    cell0.north = cell1
    cell1.south = cell0
    cell1.west = cell2
    cell2.east = cell1
    cell2.south = cell3
    cell3.north = cell2

    return cell0

def test_linear_maze():
    start = linear_maze()
    assert is_path_to_freedom(start, "EEE")

def test_square_maze():
    start = square_maze()
    assert is_path_to_freedom(start, "NWS")

def test_linear_extra_steps():
    """ Check that a non-optimal route still works. """
    start = linear_maze()
    assert is_path_to_freedom(start, "EWEEE")

def test_linear_missing_items():
    """ Didn't gather all of the items. """
    start = linear_maze()
    assert is_path_to_freedom(start, "EE") == False

def test_linear_dont_double_count():
    """ Check that we got 3 unique items. """
    start = linear_maze()
    assert is_path_to_freedom(start, "EEW") == False

def test_linear_wrong_dir():
    """ Check that we notice when we've tried to go a direction we shouldn't
    be able to go. """
    start = linear_maze()
    assert is_path_to_freedom(start, "ENE") == False

def test_invalid_dir():
    """ Check for raising ValueError on illegal characters in path. """
    start = linear_maze()
    with pytest.raises(ValueError):
        is_path_to_freedom(start, "BAD")

def test_precondition_enforced():
    """ Check for enforcing that start isn't None. """
    with pytest.raises(AssertionError):
        is_path_to_freedom(None, "EEE")

if __name__ == "__main__":
    pytest.main()
