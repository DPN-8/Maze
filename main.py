from PIL import Image, ImageDraw

import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np



matplotlib.use('Agg')

class Node:
    def __init__(self, state, parent, action, distance):
        self.state = state
        self.parent = parent
        self.action = action
        self.distance = distance

class Frontier:
    def __init__(self):
        self.space = []

    def add(self, val):
        self.space.append(val)

    def is_empty(self):
        return len(self.space) == 0

    def remove_state(self, state):
        for node in self.space:
            if node.state == state:
                self.space.remove(node)
                return

    def contains(self, state):
        return any(node.state == state for node in self.space)

    def get_best(self):
        best_node = None
        min_distance = float('inf')
        for node in self.space:
            if node.distance < min_distance:
                min_distance = node.distance
                best_node = node

        if best_node is None:
            raise Exception('No best node in the frontier')
        return best_node

class Maze:
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            file_content = file.read()

            self.state = []
            i = 0
            for line in file_content.splitlines():
                row = []
                j = 0
                for char in line:
                    if char == ' ':
                        row.append(True)
                    elif char == '#':
                        row.append(False)
                    elif char == 'A':
                        self.start = (i, j)
                        row.append(True)
                    elif char == 'B':
                        self.end = (i, j)
                        row.append(True)
                    else:
                        raise Exception(f"Not a valid character in the maze '{char}'")
                    j += 1
                self.state.append(row)
                i += 1

        x_end, y_end = self.end
        self.manhattan = []

        for i in range(len(self.state)):
            row = []
            for j in range(len(self.state[0])):
                if self.state[i][j]:
                    distance = abs(i - x_end) + abs(j - y_end)
                    row.append(distance)
                else:
                    row.append('#')
            self.manhattan.append(row)

    def get_neighbours(self, node):
        state = node.state
        neighbors = []
        row = state[0]
        col = state[1]
        if col > 0 and self.state[row][col - 1]:  # left
            neighbors.append(Node((row, col - 1), node, 'LEFT', self.manhattan[row][col - 1]))
        if col < len(self.state[row]) - 1 and self.state[row][col + 1]:  # right
            neighbors.append(Node((row, col + 1), node, 'RIGHT', self.manhattan[row][col + 1]))
        if row > 0 and self.state[row - 1][col]:  # up
            neighbors.append(Node((row - 1, col), node, 'UP', self.manhattan[row - 1][col]))
        if row < len(self.state) - 1 and self.state[row + 1][col]:  # down
            neighbors.append(Node((row + 1, col), node, 'DOWN', self.manhattan[row + 1][col]))
        return neighbors

    def save_maze_image(self, file_name='maze_image.png'):
        try:
            maze_image = np.zeros((*np.array(self.state).shape, 3), dtype=np.uint8)
            maze_image[np.array(self.state)] = [255, 255, 255]
            maze_image[~np.array(self.state)] = [0, 0, 0]
            plt.imshow(maze_image)
            plt.axis('off')
            plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
            plt.close()

        except Exception as e:
            print(f"Error occurred while saving the maze image: {str(e)}")

    def save_solution_image(self, solution_cells, file_name='maze_solution.png'):
        try:
            cell_size = 50

            maze_image = np.zeros((len(self.state) * cell_size, len(self.state[0]) * cell_size, 3), dtype=np.uint8)
            for i in range(len(self.state)):
                for j in range(len(self.state[0])):
                    color = [0, 0, 255] if self.state[i][j] else [0, 0, 0]
                    maze_image[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size] = color

            image = Image.fromarray(maze_image)
            draw = ImageDraw.Draw(image)

            for i in range(len(solution_cells) - 1):
                x1, y1 = solution_cells[i]
                x2, y2 = solution_cells[i + 1]
                draw.line(
                    [(y1 * cell_size + cell_size // 2, x1 * cell_size + cell_size // 2),
                    (y2 * cell_size + cell_size // 2, x2 * cell_size + cell_size // 2)],
                    fill=(255, 255, 0), width=5
                )

            start_x, start_y = self.start
            end_x, end_y = self.end
            draw.rectangle([(start_y * cell_size, start_x * cell_size), ((start_y + 1) * cell_size, (start_x + 1) * cell_size)], fill=(255, 0, 0))
            draw.rectangle([(end_y * cell_size, end_x * cell_size), ((end_y + 1) * cell_size, (end_x + 1) * cell_size)], fill=(0, 255, 0))

            image.save(file_name)
            image.show()

        except Exception as e:
            print(f"Error occurred while saving the solution image: {str(e)}")

    def solve(self):
        self.frontier = Frontier()
        self.num_steps = 0

        self.node = Node(self.start, None, None, self.manhattan[self.start[0]][self.start[1]])
        self.frontier.add(self.node)
        visited = set()

        while not self.frontier.is_empty():
            node = self.frontier.get_best()
            print(f"Visiting node: {node.state}")

            if node.state == self.end:
                print("Reached the end node!")
                actions = []
                cells = []

                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                return

            visited.add(node.state)
            for i in self.get_neighbours(node):
                if not self.frontier.contains(i.state) and i.state not in visited:
                    self.frontier.add(i)
            self.frontier.remove_state(node.state)

        print("No solution found")

    def print_output(self):
        try:
            with open('maze_output.txt', 'w') as file:
                for row in self.manhattan:
                    line = ''.join(str(cell) + ' ' for cell in row)
                    file.write(line + '\n')
        except Exception as e:
            print(f"Error occurred while printing the maze {str(e)}")


maze = Maze(sys.argv[1])
maze.solve()
if hasattr(maze, 'solution'):
    maze.save_solution_image(maze.solution[1])
else:
    print("No solution found")
