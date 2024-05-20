#ifndef SYS_HANDLER
#define SYS_HANDLE

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/grid/tria.h>
#include <fstream>
#include <iostream>

namespace grid {

using namespace dealii;

template <int dim> class SystemHandler {
public:
  SystemHandler<dim>(const std::string filename, MPI_Comm mpi_communicator,
                     ConditionalOStream pcout);
  void print_grid_information();

private:
  friend class Solver;

  MPI_Comm mpi_communicator;
  ConditionalOStream pcout;

  parallel::distributed::Triangulation<dim> triangulation;
};

template <int dim>
SystemHandler<dim>::SystemHandler(const std::string filename,
                                  MPI_Comm mpi_communicator,
                                  ConditionalOStream pcout)
    : mpi_communicator(mpi_communicator), pcout(pcout),
      triangulation(
          mpi_communicator,
          typename Triangulation<dim>::MeshSmoothing(
              Triangulation<dim>::MeshSmoothing::smoothing_on_refinement |
              Triangulation<dim>::MeshSmoothing::smoothing_on_coarsening)) {
  if (dim != 2) {
    ExcNotImplemented(
        "Only two dimensional problems are currently implemented");
  }
  GridIn<dim> grid_in;
  grid_in.attach_triangulation(this->triangulation);
  std::ifstream f(filename);
  grid_in.read_msh(f);

  this->print_grid_information();
}

template <int dim> void SystemHandler<dim>::print_grid_information() {
  this->pcout << "Sucessfully loaded triangulation:\n\tNumber of cells: ";
  this->pcout << this->triangulation.n_active_cells() << std::endl;

  std::string out_name = "mesh";
  GridOut grid_out;
  grid_out.write_mesh_per_processor_as_vtu(this->triangulation, out_name);
}

} // namespace grid

#endif
