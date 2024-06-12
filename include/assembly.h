#ifndef EULER_EQ
#define EULER_EQ

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <deal.II/base/types.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_vector.h>

#include "system_handler.h"

namespace assembly {

using namespace dealii;

struct EquationParameters {

  EquationParameters(double timestep, double density, double nu)
      : timestep(timestep), density(density), nu(nu){};
  EquationParameters(EquationParameters& params)
      : timestep(params.timestep), density(params.density), nu(params.nu){};

  double timestep;
  double density;
  double nu;
};
namespace scratch {
template <int dim> struct SystemCell {
  SystemCell(const FiniteElement<dim>& fe, const Quadrature<dim>& quad_formula,
             const UpdateFlags update_flags);
  SystemCell(const SystemCell& system_cell);

  FEValues<dim> fe_vals;

  std::vector<Tensor<1, dim>> u_sol_0;
  std::vector<Tensor<1, dim>> u_lap_0;
  std::vector<Tensor<2, dim>> u_grad_0;

  std::vector<Tensor<1, dim>> u_sol_1;
  std::vector<Tensor<1, dim>> u_lap_1;
  std::vector<Tensor<2, dim>> u_grad_1;

  std::vector<Tensor<1, dim>> p_grad_0;
};

template <int dim> struct PreconCell {
  PreconCell(const FiniteElement<dim>& fe, const Quadrature<dim>& quad_formula,
             const UpdateFlags update_flags);
  PreconCell(const PreconCell& system_cell);
};
} // namespace scratch

namespace copy {

template <int dim> struct LocalData {
  LocalData(const FiniteElement<dim>& fe);
  LocalData(const LocalData<dim>& data);

  LocalData& operator=(const LocalData&) = default;

  FullMatrix<double> local_matrix;
  Vector<double> local_rhs;

  std::vector<types::global_dof_index> local_dof_indices;
};

template <int dim> struct LocalPrecon {

  LocalPrecon(const FiniteElement<dim>& fe);
  LocalPrecon(const LocalPrecon<dim>& data);

  LocalPrecon& operator=(const LocalPrecon&) = default;

  FullMatrix<double> local_matrix;
  Vector<double> local_rhs;

  std::vector<types::global_dof_index> local_dof_indices;
};

} // namespace copy

template <int dim> class Assembler {
public:
  Assembler<dim>(MPI_Comm mpi_comm, ConditionalOStream pcout,
                 grid::SystemHandler<dim>* ptr_system_handler);

  MPI_Comm mpi_comm;
  ConditionalOStream pcout;

  grid::SystemHandler<dim>* ptr_system_handler;

  using cell_type = typename DoFHandler<dim>::active_cell_iterator;
  void perform_assembly(const cell_type& cell,
                        scratch::SystemCell<dim>& scratch,
                        copy::LocalData<dim>& copy, bool assemble_matrix,
                        bool assemble_velocity, bool assemble_pressure);
  void perform_precon_assembly(scratch::PreconCell<dim>& scratch,
                               copy::LocalPrecon<dim>& copy);
};

template <int dim>
Assembler<dim>::Assembler(MPI_Comm mpi_comm, ConditionalOStream pcout,
                          grid::SystemHandler<dim>* ptr_system_handler)
    : mpi_comm(mpi_comm), pcout(pcout), ptr_system_handler(ptr_system_handler) {
}

template <int dim>
void Assembler<dim>::perform_assembly(const cell_type& cell,
                                      scratch::SystemCell<dim>& scratch,
                                      copy::LocalData<dim>& copy,
                                      bool assemble_matrix,
                                      bool assemble_velocity,
                                      bool assemble_pressure) {
  //---------------------------------------------------------------------------
  //
  // Solve using the projection method with second order forward differences
  //
  // Note that the matrices are relatively straight-forward if we are using a
  // first order method
  //
  // M U^* = U^N + dt * ( - U^N \cdot \nabla U^N + \nu \nabla^2 U^N)
  //
  //---------------------------------------------------------------------------
  copy.local_matrix = 0;
  copy.local_rhs = 0;

  scratch.fe_vals(cell);

  const uint dofs_per_cell = scratch.fe_vals.get_fe().n_dofs_per_cell();
  const uint n_q_points = scratch.fe_vals.get_quadrature().size();

  cell->get_dof_indices(copy.local_dof_indices);

  FEValuesExtractors::Vector velocity(0);
  FEValuesExtractors::Scalar pressure(dim);

  // Solution at previous timestep
  scratch.fe_vals[velocity].get_function_values(ptr_system_handler->sol_0,
                                                scratch.u_sol_0);
  scratch.fe_vals[velocity].get_function_gradients(ptr_system_handler->sol_0,
                                                   scratch.u_grad_0);
  scratch.fe_vals[velocity].get_function_laplacians(ptr_system_handler->sol_0,
                                                    scratch.u_lap_0);

  for (uint q = 0; q < n_q_points; q++) {

    for (uint k = 0; k < dofs_per_cell; k++) {
      scratch.u_shape_val[k] = scratch.fe_vals[velocity].value(k, q);
      scratch.p_shape_val[k] = scratch.fe_vals[pressure].value(k, q);
    }

    for (uint i = 0; i < dofs_per_cell; i++) {

      if (assemble_matrix) {
        for (uint j = 0; j < dofs_per_cell; j++) {
          copy.local_matrix[i][j] = scratch.u_shape_val[i] *
                                    scratch.u_shape_val[j] *
                                    scratch.fe_vals.JxW(q);
          copy.local_matrix[i][j] = scratch.p_shape_grad[i] *
                                    scratch.p_shape_grad[j] *
                                    scratch.fe_vals.JxW(q);
        }
      }

      if (assemble_velocity) {
        copy.local_rhs[i] = scratch.u_shape_val[i] *
                            (scratch.u_sol_0[q] +
                             this->ns_params.timestep *
                                 (-scratch.u_sol_0[q] * scratch.u_grad_0[q] +
                                  this->ns_params.nu * scratch.u_lap_0[q])) *
                            scratch.fe_vals.JxW(q);
      } else if (assemble_pressure) {
        copy.local_rhs[i] =
            (this->ns_params.density / this->ns_params.timestep) *
            (scratch.p_shape_val[i] * scratch.u_div_0[q]) *
            scratch.fe_vals.JxW(q);
      }
    }
  }
}

} // namespace assembly

#endif
