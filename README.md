# Maze Solver

This project provides a solution to solve mazes using the A* search algorithm. The maze solver reads a maze from a text file, finds the shortest path from the start (`A`) to the end (`B`), and visualizes the solution.

## Features

- **Reads Maze from File:** The maze is read from a text file where walls are represented by `#`, open spaces by spaces, start point by `A`, and end point by `B`.
- **A* Search Algorithm:** The solver uses the A* search algorithm to find the shortest path from the start to the end.
- **Visualization:** The solution path is visualized and saved as an image. Walls are represented in black, open spaces in white, the start point in red, the end point in green, and the solution path in yellow.

## Installation

Ensure you have Python installed. This project requires the following Python libraries:

- `PIL`
- `numpy`
- `matplotlib`

You can install these dependencies using `pip`:

```bash
pip install pillow numpy matplotlib
```

## Usage
### Prepare the Maze File: Create a text file containing your maze. Use the following characters:

- `#` for walls
- ` ` for open spaces
- `A` for the start point
- `B` for the end point
- **Run the Solver**: Execute the maze solver script, passing the maze file as an argument:
```bash
python maze_solver.py maze.txt
```
## Code Structure
- **Node Class:** Represents a node in the maze with its state, parent node, action, and distance.
- **Frontier Class**: Manages the frontier for the A* search algorithm.
- **Maze Class**: Handles maze initialization, neighbor identification, maze solving using A*, and visualization of the maze and solution path.

## Example output

![maze_solution](https://github.com/DPN-8/Maze/assets/114368649/9a8ee9f9-698e-4848-af60-5f17c957b15f)



