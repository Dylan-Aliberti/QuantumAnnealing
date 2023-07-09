# QuantumAnnealing
This is a small project for getting experience with quantum annealing. In order to use the quantum annealer, a problem has to be formulated into either a Quadratic Binary Optimization (QUBO) problem, or an Ising problem (which is equivalent). D'Wave has adiabatic quantum computers which are capable of solving this kind of problem using quantum annealing. <br>

The problem that we try so solve is a factorization problem. We factorise a number R into P*Q, where R is assumed to be the product of two prime numbers. This could for example be used to break RSA encryption.

In this project, two approaches are considered for formulating the problem into a QUBO. <br>
The first approach is by modelling binary multiplication in terms of logical gates, which is taken from D'Wave's example: <br> 
https://github.com/dwave-examples/factoring-notebook/blob/master/01-factoring-overview.ipynb <br>

The second approach is based on work from Jiang et al, which reduces the number of variables: <br>
https://www.nature.com/articles/s41598-018-36058-z <br>

Due to the limited time spent on this project, do not consider it finished nor stable.
