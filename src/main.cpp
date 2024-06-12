#include <deal.II/base/mpi.h>
#include <deal.II/grid/grid_in.h>
#include <iostream>

#include "assembly.h"
#include "system_handler.h"

namespace ns_solver {

using namespace dealii;

template <int dim> class Solver {
public:
  Solver<dim>(const std::string filename);
  void run();

private:
  MPI_Comm mpi_comm;
  ConditionalOStream pcout;

  grid::SystemHandler<dim> system_handler;
  assembly::Assembler<dim> assembler;
};

template <int dim>
Solver<dim>::Solver(const std::string filename)
    : mpi_comm(MPI_COMM_WORLD),
      pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_comm) == 0)),
      system_handler(filename, mpi_comm, pcout),
      assembler(mpi_comm, pcout, &system_handler){};

template <int dim> void Solver<dim>::run() {
  this->pcout << "Running..." << std::endl;

  this->system_handler.initialise_dofs();
}

} // namespace ns_solver

int main(int argc, char* argv[]) {

  dealii::Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 2);
  ns_solver::Solver<2> solver("mesh.msh");
  solver.run();
}
