import numpy as np
from matplotlib import pyplot
from scipy.signal import convolve2d

# Define some initial patterns for the Game of Life
glider = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]])
blinker = np.array([
    [0, 0, 0],
    [1, 1, 1],
    [0, 0, 0]
])
glider_gun = np.array([
    [0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0]
])


# Define a class for the Game of Life
class Game:
    def __init__(self, Size):
        """
        Initialize the Game of Life.

        Args:
            Size (int): The size of the game board (Size x Size).
        """
        self.board = np.zeros((Size, Size))

    def play(self):
        """
        Start playing the Game of Life interactively.

        This method continuously updates and displays the game board until
        the user stops the game using Ctrl+C.
        """
        print("Playing life. Press ctrl + c to stop.")
        pyplot.ion()
        while True:
            self.move()
            self.show()
            pyplot.pause(0.0000005)

    def move(self):
        """
        Perform one step of the Game of Life.

        This method computes the next generation of the game board based on
        the rules of the Game of Life.
        """
        STENCIL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        NeighbourCount = convolve2d(self.board, STENCIL, mode='same')

        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                self.board[i, j] = 1 if (NeighbourCount[i, j] == 3
                                         or (NeighbourCount[i, j] == 2
                                             and self.board[i, j])) else 0

    def __setitem__(self, key, value):
        """
        Set a specific cell on the game board to a given value.

        Args:
            key (tuple): A tuple containing the coordinates (x, y) of the cell.
            value (int): The value to set the cell to (0 or 1).
        """
        self.board[key] = value

    def show(self):
        """
        Display the current state of the game board.
        """
        pyplot.clf()
        pyplot.matshow(self.board, fignum=0, cmap='binary')
        pyplot.show()

    def insert(self, pattern, location):
        """
        Insert a pattern into the game board at a specific location.

        Args:
            pattern (Pattern): The pattern to insert.
            location (tuple): The coordinates (x, y) where the pattern should
            be placed.
        """
        x, y = location
        grid_shape = pattern.grid.shape
        x_start = x - grid_shape[0] // 2
        y_start = y - grid_shape[1] // 2

        self.board[x_start:x_start + grid_shape[0],
                   y_start:y_start + grid_shape[1]] = pattern.grid


# Define a class for representing patterns in the Game of Life
class Pattern:
    def __init__(self, grid):
        """
        Initialize a pattern with a given grid.

        Args:
            grid (list): A 2D list representing the pattern's initial
            configuration.
        """
        self.grid = np.array(grid)

    def flip_vertical(self):
        """
        Flip the pattern vertically and return a new Pattern object.
        """
        return Pattern(self.grid[::-1])

    def flip_horizontal(self):
        """
        Flip the pattern horizontally and return a new Pattern object.
        """
        return Pattern(self.grid[:, ::-1])

    def flip_diag(self):
        """
        Flip the pattern diagonally and return a new Pattern object.
        """
        return Pattern(self.grid.T)

    def rotate(self, n):
        """
        Rotate the pattern by 90 degrees 'n' times and return a new Pattern
        object.

        Args:
            n (int): The number of 90-degree rotations to apply.
        """
        return Pattern(np.rot90(self.grid, n))
