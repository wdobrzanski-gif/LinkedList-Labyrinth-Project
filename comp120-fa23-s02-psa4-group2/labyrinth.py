"""
Module: Labyrinth

Code to generate random labyrinths to trap unsuspecting students who
don't know how to use a debugger.

Author: Dr. Sat Garcia (sat@sandiego.edu)
"""

from enum import Enum, auto
from typing import Optional
import math, random
from collections import deque

class Item(Enum):
    """ Type representing an item that may appear in the maze. """
    NOTHING = 0
    POTION = 1
    SPELLBOOK = 2
    WAND = 3

class MazeCell:
    """ Type representing a cell in a maze. """

    whats_here: Item

    north: Optional['MazeCell']
    south: Optional['MazeCell']
    east: Optional['MazeCell']
    west: Optional['MazeCell']

    def __init__(self):
        self.whats_here = Item.NOTHING  # One of "", "Potion", "Spellbook", and "Wand"

        self.north = None
        self.south = None
        self.east = None
        self.west = None


# Size of a normal maze.
NUM_ROWS = 4
NUM_COLS = 4

# Size of a twisty maze.
TWISTY_MAZE_SIZE = 12



def maze_for(name: str) -> MazeCell:
    """
    Returns a maze specifically tailored to the given name.

    We've implemented this function for you. You don't need to write it
    yourself.

    Please don't make any changes to this function - we'll be using our
    reference version when testing your code, and it would be a shame if
    the maze you solved wasn't the maze we wanted you to solve!

    Returns:
        (MazeCell) The starting location.
    """

    # Set the seed based on a string of the <name> and the dimensions of the
    # maze. Python's random library uses sha512 to convert the string to an
    # int for the seeding.
    seed_string = f"{name} {NUM_ROWS}x{NUM_COLS}"
    random.seed(a=seed_string)

    maze = makeMaze(NUM_ROWS, NUM_COLS)

    linearMaze = []
    for row in range(len(maze)):
        for col in range(len(maze[0])):
            linearMaze.append(maze[row][col])

    distances = allPairsShortestPaths(linearMaze)

    # FIXME tuple assignment
    locations = remoteLocationsIn(distances)
    #print(locations)

    """ Place the items. """
    linearMaze[locations[1]].whats_here = Item.SPELLBOOK
    linearMaze[locations[2]].whats_here = Item.POTION
    linearMaze[locations[3]].whats_here = Item.WAND

    """ We begin in position 0. """
    return linearMaze[locations[0]]


def twisty_maze_for(name: str) -> MazeCell:
    """
    Returns a twisty maze specifically tailored to the given name.

    Please don't make any changes to this function - we'll be using our
    reference version when testing your code, and it would be a shame if the
    maze you solved wasn't the maze we wanted you to solve!

    Returns:
        (MazeCell) The starting location.
    """

    # Set the seed based on a string of the <name> and the size of the maze.
    # Python's random library uses SHA512 to convert the string to an int for
    # the seeding.
    seed_string = f"{name} twisty-{TWISTY_MAZE_SIZE}"
    random.seed(a=seed_string)

    maze = makeTwistyMaze(TWISTY_MAZE_SIZE)

    # Find the distances between all pairs of nodes.
    distances = allPairsShortestPaths(maze)

    # Select a 4-tuple maximizing the minimum distances between points, and
    # use that as our item/start locations.

    # FIXME tuple assignment
    locations = remoteLocationsIn(distances)

    # Place the items there.
    maze[locations[1]].whats_here = Item.SPELLBOOK
    maze[locations[2]].whats_here = Item.POTION
    maze[locations[3]].whats_here = Item.WAND

    return maze[locations[0]]


def areAdjacent(first: MazeCell, second: MazeCell) -> bool:
    """ Returns if two nodes are adjacent. """
    return second in [first.east, first.west, first.north, first.south]


def allPairsShortestPaths(maze: list[MazeCell]) -> list[list[int]]:
    """Uses the Floyd-Warshall algorithm to compute the shortest paths between all
    pairs of nodes in the maze. The result is a table where table[i][j] is the
    shortest path distance between maze[i] and maze[j]."""


    # Floyd-Warshall algorithm. Fill the grid with "infinity" values.

    result: list[list[int]] = []
    for i in range(len(maze)):
        r = [len(maze)+1] * len(maze)
        result.append(r)


    # Set distances of nodes to themselves at 0.
    for i in range(len(maze)):
        result[i][i] = 0

    # Set distances of edges to 1.
    for i in range(len(maze)):
        for j in range(len(maze)):
            if areAdjacent(maze[i], maze[j]):
                result[i][j] = 1


    # Dynamic programming step. Keep expanding paths by allowing for paths
    # between nodes.
    for i in range(len(maze)):
        next_result: list[list[int]] = []
        for j in range(len(maze)):
            row = []
            for k in range(len(maze)):
               val = min(result[j][k], result[j][i] + result[i][k])
               row.append(val)

            next_result.append(row)

        result = next_result

    return result


def scoreOf(nodes: list[int], distances: list[list[int]]) -> list[int]:
    """Given a list of distinct nodes, returns the "score" for their distances,
    which is a sequence of numbers representing pairwise distances in sorted
    order."""

    result = []

    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            result.append(distances[nodes[i]][nodes[j]])

    result.sort()
    return result


def lexicographicallyFollows(lhs: list[int], rhs: list[int]) -> bool:
    """ Lexicographical comparison of two lists; they're assumed to have the
    same length. """
    assert len(lhs) == len(rhs)

    for i in range(len(lhs)):
        if lhs[i] != rhs[i]:
            return lhs[i] > rhs[i]

    return False


def remoteLocationsIn(distances: list[list[int]]) -> list[int]:
    """ Given a grid, returns a combination of four nodes whose overall score
    (sorted list of pairwise distances) is as large as possible in a
    lexicographical sense.  """

    result = [0, 1, 2, 3]

    # We could do this recursively, but since it's "only" four loops we'll
    # just do that instead. :-)
    for i in range(len(distances)):
        for j in range(i+1, len(distances)):
            for k in range(j+1, len(distances)):
                for l in range(k+1, len(distances)):
                    curr = [i, j, k, l]
                    if lexicographicallyFollows(scoreOf(curr, distances), scoreOf(result, distances)):
                        result = curr

    return result


def clearGraph(nodes: list[MazeCell]) -> None:
    """ Clears all the links between the given group of nodes. """
    for node in nodes:
        node.whats_here = Item.NOTHING
        node.north = node.south = node.east = node.west = None


class Port(Enum):
    """ Enumerated type representing one of the four ports leaving a MazeCell.  """
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


""" Returns a random unassigned link from the given node, or None if
 * they are all assigned.
 """
def randomFreePortOf(cell: MazeCell) -> Optional[Port]:
    ports = []

    if cell.east is None:
        ports.append(Port.EAST)
    if cell.west is None:
        ports.append(Port.WEST)
    if cell.north is None:
        ports.append(Port.NORTH)
    if cell.south is None:
        ports.append(Port.SOUTH)

    if len(ports) == 0:
        return None

    return random.choice(ports)


def link(start: MazeCell, end: MazeCell, link: Port) -> None:
    """ Links one MazeCell to the next using the specified port. """

    if link == Port.EAST:
        start.east = end
    elif link == Port.WEST:
        start.west = end
    elif link == Port.NORTH:
        start.north = end
    elif link == Port.SOUTH:
        start.south = end
    else:
        raise RuntimeError("Unknown port")


def erdosRenyiLink(nodes: list[MazeCell]) -> bool:
    """ Use a variation of the Erdos-Renyi random graph model. We set the
    probability of any pair of nodes being connected to be ln(n) / n,
    then artificially constrain the graph so that no node has degree
    four or more. We generate mazes this way until we find one that's
    conencted. """

    # High probability that everything is connected.
    threshold = math.log(len(nodes)) / len(nodes)

    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            if (random.random() <= threshold):
                iLink = randomFreePortOf(nodes[i])
                jLink = randomFreePortOf(nodes[j])

                # Oops, no free links.
                if iLink is None or jLink is None:
                    return False

                link(nodes[i], nodes[j], iLink)
                link(nodes[j], nodes[i], jLink)

    return True


def isConnected(maze: list[MazeCell]) -> bool:
    """ Returns whether the given maze is connected. Uses a BFS. """
    visited: set[MazeCell] = set()

    frontier = deque() # use as a queue

    frontier.append(maze[0])

    while len(frontier) != 0:
        curr: MazeCell = frontier.popleft();

        if curr not in visited:
            visited.add(curr)

            if (curr.east  != None):
                frontier.append(curr.east)
            if (curr.west  != None):
                frontier.append(curr.west)
            if (curr.north != None):
                frontier.append(curr.north)
            if (curr.south != None):
                frontier.append(curr.south)

    return len(visited) == len(maze)


def makeTwistyMaze(numNodes: int) -> list[MazeCell]:
    """ Generates a random twisty maze. This works by repeatedly generating
    random graphs until a connected one is found.  """

    result = []
    for i in range(numNodes):
        result.append(MazeCell())

    # Keep generating mazes until we get a connected one.
    clearGraph(result)
    while (not erdosRenyiLink(result)) or (not isConnected(result)):
        clearGraph(result)

    return result


class EdgeBuilder:
    """ Type representing an edge between two maze cells. """
    start: MazeCell
    end: MazeCell

    fromPort: Port
    toPort: Port

    def __init__(self, start: MazeCell, end: MazeCell, fromPort: Port, toPort: Port) -> None:
        self.start     = start
        self.end       = end
        self.fromPort = fromPort
        self.toPort   = toPort


def allPossibleEdgesFor(maze: list[list[MazeCell]]) -> list[EdgeBuilder]:
    """ Returns all possible edges that could appear in a grid maze. """

    result = []
    for row in range(len(maze)):
        for col in range(len(maze[row])):
            if (row + 1) < len(maze):
                result.append(EdgeBuilder(maze[row][col], maze[row + 1][col], Port.SOUTH, Port.NORTH))

            if (col + 1) < len(maze[row]):
                result.append(EdgeBuilder(maze[row][col], maze[row][col + 1], Port.EAST,  Port.WEST))

    return result


def repFor(reps: dict[MazeCell, MazeCell], cell: MazeCell) -> MazeCell:
    """ Union-find FIND operation. """
    while reps[cell] != cell:
        cell = reps[cell]

    return cell


def shuffleEdges(edges: list[EdgeBuilder]) -> None:
    """ Shuffles the edges using the Fischer-Yates shuffle. """
    for i in range(len(edges)):
        j = random.randrange(len(edges) - i) + i

        temp = edges[i]
        edges[i] = edges[j]
        edges[j] = temp


def makeMaze(numRows: int, numCols: int) -> list[list[MazeCell]]:
    """ Creates a random maze of the given size using a randomized Kruskal's
    algorithm. Edges are shuffled and added back in one at a time, provided
    that each insertion links two disconnected regions.  """

    maze: list[list[MazeCell]] = []
    for _ in range(numRows):
        row = [MazeCell() for _ in range(numCols)]
        maze.append(row)


    edges = allPossibleEdgesFor(maze)
    shuffleEdges(edges)


    # Union-find structure, done without path compression because N is small.
    representatives = {}
    for row in range(numRows):
        for col in range(numCols):
            elem = maze[row][col]
            representatives[elem] = elem


    # Run a randomized Kruskal's algorithm to build the maze.
    edgesLeft = numRows * numCols - 1

    i = 0
    while edgesLeft > 0 and i < len(edges):
        edge = edges[i]

        # See if they're linked already.
        rep1 = repFor(representatives, edge.start)
        rep2 = repFor(representatives, edge.end)

        # If not, link them.
        if rep1 != rep2:
            representatives[rep1] = rep2

            link(edge.start, edge.end, edge.fromPort)
            link(edge.end, edge.start, edge.toPort)

            edgesLeft -= 1

        i += 1

    if (edgesLeft != 0):
        raise RuntimeError("Edges remain?") # Internal error!

    return maze
