#include <deal.II/base/mpi.h>
#include <deal.II/grid/grid_in.h>
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
  this->assembler.assemble_system();

  this->pcout << "Matrix Norms:" << "\n\tBlock (0,0):"
              << this->system_handler.system_matrix.block(0, 0).l1_norm()
              << "\n\tBlock (0,1):"
              << this->system_handler.system_matrix.block(0, 1).l1_norm()
              << "\n\tBlock (1,0):"
              << this->system_handler.system_matrix.block(1, 0).l1_norm()
              << "\n\tBlock (1,1):"
              << this->system_handler.system_matrix.block(1, 1).l1_norm()
              << "\n\nVector Norms:" << "\n\tBlock 0: "
              << this->system_handler.rhs.block(0).l2_norm()
              << "\n\tBlock 1: " << this->system_handler.rhs.block(1).l2_norm()
              << std::endl;
}

} // namespace ns_solver

int main(int argc, char* argv[]) {

  assembly::EquationParameters eqn_params;
  grid::FiniteElementDescription fe_params;

  fe_params.debug = false;
  fe_params.min_refine = 4;
  fe_params.max_refine = 10;
  fe_params.degree = 1;

  eqn_params.timestep = 1e-2;
  eqn_params.nu = 1e-2;
  eqn_params.density = 1;

  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 2);
  ns_solver::Solver<2> solver("mesh.msh", fe_params, eqn_params);
  solver.run();
}
