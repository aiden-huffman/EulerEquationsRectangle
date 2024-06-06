#ifndef SYS_HANDLER
#define SYS_HANDLER

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/types.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/component_mask.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/numerics/vector_tools_boundary.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <fstream>
#include <iostream>

namespace grid {

using namespace dealii;

template <int dim> class SystemHandler {
public:
  SystemHandler<dim>(const std::string filename, MPI_Comm mpi_communicator,
                     ConditionalOStream pcout);
  void print_grid_information();

  MPI_Comm mpi_comm;
  ConditionalOStream pcout;

  parallel::distributed::Triangulation<dim> triangulation;
  DoFHandler<dim> dof_handler;
  FESystem<dim> fe_system;

  using dof_type = types::global_dof_index;
  std::vector<uint> fe_sub_blocks;
  std::vector<uint> dofs_per_block;
  std::vector<std::pair<dof_type, dof_type>> dof_block_ranges;

  AffineConstraints<double> constraints;

  void initialise_dofs();
  void export_solution();
};

template <int dim>
SystemHandler<dim>::SystemHandler(const std::string filename, MPI_Comm mpi_comm,
                                  ConditionalOStream pcout)
    : mpi_comm(mpi_comm), pcout(pcout),
      triangulation(
          mpi_comm,
          typename Triangulation<dim>::MeshSmoothing(
              Triangulation<dim>::MeshSmoothing::smoothing_on_refinement |
              Triangulation<dim>::MeshSmoothing::smoothing_on_coarsening)),
      dof_handler(triangulation), fe_system(FE_Q<dim>(2), dim, FE_Q<dim>(1), 1),
      fe_sub_blocks(dim + 1, 0) {

  if (dim != 2) {
    ExcNotImplemented(
        "Only two dimensional problems are currently implemented");
  }

  GridIn<dim> grid_in;
  grid_in.attach_triangulation(this->triangulation);
  std::ifstream f(filename);
  grid_in.read_msh(f);

  this->print_grid_information();

  fe_sub_blocks[dim] = 1;
}

template <int dim> void SystemHandler<dim>::initialise_dofs() {

  this->pcout << "Collecting degrees of freedom:" << std::endl;

  /*-------------------------------------------------------------------------

    RENUMBERING

  -------------------------------------------------------------------------*/
  this->dof_handler.distribute_dofs(this->fe_system);
  DoFRenumbering::Cuthill_McKee(this->dof_handler);
  DoFRenumbering::component_wise(this->dof_handler, fe_sub_blocks);

  /*-------------------------------------------------------------------------

    GRAB DOF RANGES

  -------------------------------------------------------------------------*/
  this->dofs_per_block =
      DoFTools::count_dofs_per_fe_block(this->dof_handler, this->fe_sub_blocks);
  using dof_type = types::global_dof_index;

  dof_type lower = 0;
  for (uint i = 0; i < dim; i++) {
    this->dof_block_ranges.push_back(
        std::pair<dof_type, dof_type>(lower, lower + this->dofs_per_block[i]));

    lower += this->dofs_per_block[i];
  }

  this->pcout << "\tTotal DoFs: " << this->dof_handler.n_dofs() << std::endl
              << "\tBlock 0: (" << dof_block_ranges[0].first << ", "
              << dof_block_ranges[0].second << ")\n"
              << "\tBlock 1: (" << dof_block_ranges[1].first << ", "
              << dof_block_ranges[1].second << ")" << std::endl;

  /*-------------------------------------------------------------------------

    CONSTRAINTS:

      p(left) = 1
      p(right) = 0

      u(rectangle) = 0
      u(top) = 0
      u(bottom) = 0

  -------------------------------------------------------------------------*/

  IndexSet owned_index_set = this->dof_handler.locally_owned_dofs();
  IndexSet relevant_set;
  DoFTools::extract_locally_relevant_dofs(this->dof_handler, relevant_set);

  this->constraints.clear();
  this->constraints.reinit(relevant_set);
  DoFTools::make_hanging_node_constraints(this->dof_handler, this->constraints);

  /*-------------------------------------------------------------------------

    PRESSURE

  -------------------------------------------------------------------------*/
  {
    if (Utilities::MPI::this_mpi_process(mpi_comm) == 0) {

      FEValuesExtractors::Scalar pressure(dim);
      ComponentMask pressure_mask = this->fe_system.component_mask(pressure);

      VectorTools::interpolate_boundary_values(
          this->dof_handler, 1, Functions::ConstantFunction<dim>(1.0, dim + 1),
          pressure_mask);

      VectorTools::interpolate_boundary_values(
          this->dof_handler, 3, Functions::ZeroFunction<dim>(dim + 1),
          pressure_mask);
    }
  }

  /*-------------------------------------------------------------------------

    VELOCITY

  -------------------------------------------------------------------------*/

  FEValuesExtractors::Vector velocity(0);
  ComponentMask velocity_mask = this->fe_system.component_mask(velocity);

  {
    VectorTools::interpolate_boundary_values(
        this->dof_handler, 0, Functions::ZeroFunction<dim>(dim + 1),
        this->constraints, velocity_mask);
    VectorTools::interpolate_boundary_values(
        this->dof_handler, 1, Functions::ZeroFunction<dim>(dim + 1),
        this->constraints, velocity_mask);
  }

  this->constraints.make_consistent_in_parallel(owned_index_set, relevant_set,
                                                this->mpi_comm);
  this->constraints.close();
}

template <int dim>
void SystemHandler<dim>::export_solution() {

};

template <int dim> void SystemHandler<dim>::print_grid_information() {
  this->pcout << "Sucessfully loaded triangulation:\n\tNumber of cells: ";
  this->pcout << this->triangulation.n_active_cells() << std::endl;

  std::string out_name = "mesh";
  GridOut grid_out;
  grid_out.write_mesh_per_processor_as_vtu(this->triangulation, out_name);
}

} // namespace grid

#endif
