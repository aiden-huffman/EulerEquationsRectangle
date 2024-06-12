#ifndef SYS_HANDLER
#define SYS_HANDLER

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
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

#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_boundary.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <fstream>
#include <iostream>

namespace grid {

using namespace dealii;

struct FiniteElementDescription {
  uint min_refine, max_refine;
  uint degree;
  bool debug;
};
template <int dim> class SystemHandler {
public:
  SystemHandler<dim>(const std::string filename, MPI_Comm mpi_communicator,
                     ConditionalOStream pcout,
                     FiniteElementDescription fe_params);
  void print_grid_information();

  MPI_Comm mpi_comm;
  ConditionalOStream pcout;

  FiniteElementDescription fe_params;

  parallel::distributed::Triangulation<dim> triangulation;
  DoFHandler<dim> dof_handler;
  FESystem<dim> fe_system;
  AffineConstraints<double> constraints;

  using dof_type = types::global_dof_index;
  std::vector<uint> fe_sub_blocks;
  std::vector<uint> dofs_per_block;
  std::vector<std::pair<dof_type, dof_type>> dof_block_ranges;

  // We will use a block structure to build the various systems
  // we need. However, since the projection method is an
  // operator splitting method, we know that the off-diagonals are
  // zero. We can make use of this to reduce memory consumption, but
  // will not do that in this implementation.
  TrilinosWrappers::BlockSparseMatrix system_matrix;

  TrilinosWrappers::MPI::BlockVector rhs;
  TrilinosWrappers::MPI::BlockVector sol_0;
  TrilinosWrappers::MPI::BlockVector sol_1;

  void initialise_dofs();

  using IndexPartitioning = std::vector<IndexSet>;
  void initialise_linear_system(IndexPartitioning local_part,
                                IndexPartitioning rel_part);

  void export_solution();
};

template <int dim>
SystemHandler<dim>::SystemHandler(const std::string filename, MPI_Comm mpi_comm,
                                  ConditionalOStream pcout,
                                  FiniteElementDescription fe_params)
    : mpi_comm(mpi_comm), pcout(pcout), fe_params(fe_params),
      triangulation(
          mpi_comm,
          typename Triangulation<dim>::MeshSmoothing(
              Triangulation<dim>::MeshSmoothing::smoothing_on_refinement |
              Triangulation<dim>::MeshSmoothing::smoothing_on_coarsening)),
      dof_handler(triangulation),
      fe_system(FE_Q<dim>(fe_params.degree + 1), dim,
                FE_Q<dim>(fe_params.degree), 1),
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

  IndexSet owned_index_set = this->dof_handler.locally_owned_dofs();
  IndexSet relevant_set;
  DoFTools::extract_locally_relevant_dofs(this->dof_handler, relevant_set);

  std::vector<IndexSet> local_partitioning;
  std::vector<IndexSet> relevant_partitioning;

  // Build the block partitioning
  for (uint i = 0; i < 2; i++) {
    local_partitioning.push_back(owned_index_set.get_view(
        dof_block_ranges[i].first, dof_block_ranges[i].second));
    relevant_partitioning.push_back(relevant_set.get_view(
        dof_block_ranges[i].first, dof_block_ranges[i].second));
  }

  /*-------------------------------------------------------------------------

    CONSTRAINTS:

      p(somewhere) = 0

      u(left) = (1, 0, 0)
      u(right) = no-stress
      u(rectangle) = 0
      u(top) = 0
      u(bottom) = 0


  -------------------------------------------------------------------------*/

  this->constraints.clear();
  this->constraints.reinit(relevant_set);
  DoFTools::make_hanging_node_constraints(this->dof_handler, this->constraints);

  /*-------------------------------------------------------------------------

    PRESSURE

  -------------------------------------------------------------------------*/
  {
    if (Utilities::MPI::this_mpi_process(mpi_comm) == 0) {
      FEValuesExtractors::Scalar pressure(dim);
      ComponentMask flow_mask = this->fe_system.component_mask(pressure);

      std::vector<IndexSet> pressure_dofs =
          DoFTools::locally_owned_dofs_per_component(this->dof_handler,
                                                     flow_mask);
      const types::global_dof_index first_pressure_dof =
          pressure_dofs[dim].nth_index_in_set(0);
      this->constraints.add_line(first_pressure_dof);
    }
  }

  /*-------------------------------------------------------------------------

    VELOCITY

  -------------------------------------------------------------------------*/

  FEValuesExtractors::Vector velocity(0);

  // Velocity mask only tracks x-component at first
  ComponentMask velocity_mask(dim + 1, false);
  velocity_mask.set(0, true);

  {
    VectorTools::interpolate_boundary_values(
        this->dof_handler, 2, Functions::ConstantFunction<dim>(1.0, dim + 1),
        this->constraints, velocity_mask);

    for (uint i = 1; i < dim; i++) {
      velocity_mask.set(i, true);
    }

    VectorTools::interpolate_boundary_values(
        this->dof_handler, 0, Functions::ZeroFunction<dim>(dim + 1),
        this->constraints, velocity_mask);
    VectorTools::interpolate_boundary_values(
        this->dof_handler, 2, Functions::ZeroFunction<dim>(dim + 1),
        this->constraints, velocity_mask);
  }

  this->constraints.make_consistent_in_parallel(owned_index_set, relevant_set,
                                                this->mpi_comm);
  this->constraints.close();

  this->initialise_linear_system(local_partitioning, relevant_partitioning);
}

template <int dim>
void SystemHandler<dim>::initialise_linear_system(
    std::vector<IndexSet> local_part, std::vector<IndexSet> rel_part) {

  this->system_matrix.clear();

  {
    TrilinosWrappers::BlockSparsityPattern sp(local_part, local_part, rel_part,
                                              this->mpi_comm);

    DoFTools::make_sparsity_pattern(this->dof_handler, sp, this->constraints,
                                    false,
                                    Utilities::MPI::this_mpi_process(mpi_comm));

    sp.compress();

    this->system_matrix.reinit(sp);
  }

  this->sol_0.reinit(rel_part, mpi_comm);
  this->sol_1.reinit(this->sol_0);

  this->rhs.reinit(local_part, rel_part, mpi_comm);

  this->pcout << "Solution sizes:" << std::endl;
  this->pcout << "\tBlock 0: " << this->sol_0.block(0).size()
              << "\n\tBlock 1: " << this->sol_0.block(1).size() << std::endl;
  this->pcout << "Right hand side sizes:" << std::endl;
  this->pcout << "\tBlock 0: " << this->rhs.block(0).size()
              << "\n\tBlock 1: " << this->rhs.block(1).size() << std::endl;
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
