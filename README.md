# Ramsey theorem solutions using cuda

## Description

[Ramsey theorem on wikipedia](https://en.wikipedia.org/wiki/Ramsey%27s_theorem)

Let's define _solution for R(N, N)_ or shorter _solution for N_ as an information enough to represent complete graph with edges of 2 possible color
such that it doesn't contain a monochromatic complete subgraph with N vertices. Which means that Ramsey number is bigger than the number of 
vertices in your solution.

For the sake of this program, which is done just to solve fun problem and use my GPU, I made 2 __very__ strong assumptions on solutions
to make program fast to write and also to bring computational complexity to something feasible by my GPU but still hard enough to fully load it.

### First assumption:

If you numerate vertices, let's say, in clockwise direction starting from 0 on a random vertex,
you can represent one of two colors of edge connecting vertex 0 to vertex K by using bit in array with index K.
And that representation will be the same for every vertex chosen as 0.
So basically solution is symmetric with respect to vertices which means that you can use an array of length K to represent a solution.
Bit array is 64-bit int (unsigned long long) because it's the least possible to represent solutions for 5 which is the biggest N that makes sense.

### Second assumption:

Only concern if you set SEARCH_SYMMETRIC_SOLUTIONS true. As you might have guessed this assumption is about symmetric solutions but this time with respect
to direction of numerating, i.e. solution not only representable as a bit array, but that array now is a palindrome (or almost so).

Of course both of that assumptions decrease the number of solutions possible to find,
but it still works great for N = 3 (up to 5 vertices) and N = 4 (up to 17 vertices).

## Usage
Just be sure to have CUDA installed and OpenMP supported. You should configure cudaConfig.cuh to match your GPU.\
Then you  can configure runConfig.cuh and run the program using CMake.
