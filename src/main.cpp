#include <deal.II/base/mpi.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <iostream>

#include "assembly.h"
#include "system_handler.h"

namespace ns_solver {

using namespace dealii;

template <int dim> class Solver {
public:
  Solver<dim>(const std::string filename,
              grid::FiniteElementDescription fe_params,
              assembly::EquationParameters eqn_params);
  void run();

private:
  MPI_Comm mpi_comm;
  ConditionalOStream pcout;

  grid::SystemHandler<dim> system_handler;
  assembly::Assembler<dim> assembler;
};

template <int dim>
Solver<dim>::Solver(const std::string filename,
                    grid::FiniteElementDescription fe_params,
                    assembly::EquationParameters eqn_params)
    : mpi_comm(MPI_COMM_WORLD),
      pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_comm) == 0)),
      system_handler(filename, mpi_comm, pcout, fe_params),
      assembler(mpi_comm, pcout, &system_handler, eqn_params){};

template <int dim> void Solver<dim>::run() {
  this->pcout << "Running..." << std::endl;

  this->system_handler.initialise_dofs();

  for (uint i = 0; i < 1000; i++) {
    this->assembler.assemble_system();

    SolverControl cn;
    TrilinosWrappers::SolverDirect solver(cn);
    this->pcout << "Solve number " << i << ":\n\tVelocity solve..."
                << std::endl;
    solver.initialize(this->system_handler.system_matrix.block(0, 0));
    solver.solve(this->system_handler.sol_0.block(0),
                 this->system_handler.rhs.block(0));
    this->system_handler.constraints.distribute(this->system_handler.sol_0);

    this->assembler.assemble_pressure();

    this->pcout << "\tPressure solve..." << std::endl;
    TrilinosWrappers::SolverCG solver_cg(cn);
    solver.initialize(this->system_handler.system_matrix.block(1, 1));
    solver.solve(this->system_handler.sol_0.block(1),
                 this->system_handler.rhs.block(1));

    this->pcout << "\tVelocity update..." << std::endl;
    TrilinosWrappers::MPI::Vector tmp;
    tmp.reinit(this->system_handler.sol_0.block(0));

    this->system_handler.system_matrix.block(0, 1).vmult(
        tmp, this->system_handler.sol_0.block(1));
    this->system_handler.sol_0.block(0).add(-1.0, tmp);

    this->pcout << "Solution Norms:" << "\n\tBlock 0: "
                << this->system_handler.sol_0.block(0).l2_norm()
                << "\n\tBlock 1: "
                << this->system_handler.sol_0.block(1).l2_norm() << std::endl;

    if (i % 10 == 0)
      this->system_handler.export_solution(i / 10);
  }
}

} // namespace ns_solver

int main(int argc, char* argv[]) {

  assembly::EquationParameters eqn_params;
  grid::FiniteElementDescription fe_params;

  fe_params.debug = false;
  fe_params.min_refine = 4;
  fe_params.max_refine = 10;
  fe_params.degree = 1;

  eqn_params.timestep = 1e-4;
  eqn_params.nu = 1e-2;
  eqn_params.density = 1;

  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 2);
  ns_solver::Solver<2> solver("mesh.msh", fe_params, eqn_params);
  solver.run();
}
