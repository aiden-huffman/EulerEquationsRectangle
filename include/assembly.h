#ifndef EULER_EQ
#define EULER_EQ

#include <cstdlib>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/base/types.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/grid/filtered_iterator.h>

#include "system_handler.h"

namespace assembly {

using namespace dealii;

namespace scratch {
template <int dim> struct SystemCell {
  SystemCell(const FiniteElement<dim>& fe, const Quadrature<dim>& quad_formula,
             const UpdateFlags update_flags);
  SystemCell(const SystemCell& system_cell);

  FEValues<dim> fe_vals;

  std::vector<Tensor<1, dim>> u_shape_val;
  std::vector<Tensor<2, dim>> u_shape_grad;

  std::vector<Tensor<1, dim>> u_sol_0;
  std::vector<Tensor<1, dim>> u_lap_0;
  std::vector<Tensor<1, dim>> u_adv_0;
  std::vector<double> u_div_0;
  std::vector<Tensor<2, dim>> u_grad_0;

  /*std::vector<Tensor<1, dim>> u_sol_1;
  std::vector<Tensor<1, dim>> u_lap_1;
  std::vector<Tensor<2, dim>> u_grad_1;*/

  std::vector<double> p_shape_val;
  std::vector<Tensor<1, dim>> p_shape_grad;
};

template <int dim>
SystemCell<dim>::SystemCell(const FiniteElement<dim>& fe,
                            const Quadrature<dim>& quad_formula,
                            const UpdateFlags update_flags)
    : fe_vals(fe, quad_formula, update_flags),
      u_shape_val(fe_vals.dofs_per_cell), u_shape_grad(fe_vals.dofs_per_cell),
      u_sol_0(quad_formula.size()), u_lap_0(quad_formula.size()),
      u_adv_0(quad_formula.size()), u_div_0(quad_formula.size()),
      u_grad_0(quad_formula.size()), p_shape_val(fe_vals.dofs_per_cell),
      p_shape_grad(fe_vals.dofs_per_cell) {}

template <int dim>
SystemCell<dim>::SystemCell(const SystemCell& system_cell)
    : fe_vals(system_cell.fe_vals.get_fe(),
              system_cell.fe_vals.get_quadrature(),
              system_cell.fe_vals.get_update_flags()),
      u_shape_val(system_cell.u_shape_val),
      u_shape_grad(system_cell.u_shape_grad), u_sol_0(system_cell.u_sol_0),
      u_lap_0(system_cell.u_lap_0), u_adv_0(system_cell.u_adv_0),
      u_div_0(system_cell.u_div_0), u_grad_0(system_cell.u_grad_0),
      p_shape_val(system_cell.p_shape_val),
      p_shape_grad(system_cell.p_shape_grad) {}

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

/*template <int dim> struct LocalPrecon {

  LocalPrecon(const FiniteElement<dim>& fe);
  LocalPrecon(const LocalPrecon<dim>& data);

  LocalPrecon& operator=(const LocalPrecon&) = default;

  FullMatrix<double> local_matrix;
  Vector<double> local_rhs;

  std::vector<types::global_dof_index> local_dof_indices;
};*/

template <int dim>
LocalData<dim>::LocalData(const FiniteElement<dim>& fe)
    : local_matrix(fe.n_dofs_per_cell(), fe.n_dofs_per_cell()),
      local_rhs(fe.n_dofs_per_cell()), local_dof_indices(fe.n_dofs_per_cell()) {
}

template <int dim>
LocalData<dim>::LocalData(const LocalData& data)
    : local_matrix(data.local_matrix), local_rhs(data.local_rhs),
      local_dof_indices(data.local_dof_indices) {}

} // namespace copy

struct EquationParameters {
  double timestep;
  double density;
  double nu;
};

template <int dim> class Assembler {
public:
  Assembler<dim>(MPI_Comm mpi_comm, ConditionalOStream pcout,
                 grid::SystemHandler<dim>* ptr_system_handler,
                 EquationParameters eqn_params);

  MPI_Comm mpi_comm;
  ConditionalOStream pcout;

  EquationParameters eqn_params;

  grid::SystemHandler<dim>* ptr_system_handler;

  using cell_type = typename DoFHandler<dim>::active_cell_iterator;
  void assemble_cell(const cell_type& cell, scratch::SystemCell<dim>& scratch,
                     copy::LocalData<dim>& copy, bool assemble_matrix = true,
                     bool assemble_velocity = true,
                     bool assemble_pressure = false);
  void copy_cell(const copy::LocalData<dim>& data);

  void assemble_system();
};

template <int dim>
Assembler<dim>::Assembler(MPI_Comm mpi_comm, ConditionalOStream pcout,
                          grid::SystemHandler<dim>* ptr_system_handler,
                          EquationParameters eqn_params)
    : mpi_comm(mpi_comm), pcout(pcout), eqn_params(eqn_params),
      ptr_system_handler(ptr_system_handler) {}

template <int dim>
void Assembler<dim>::assemble_cell(
    const typename DoFHandler<dim>::active_cell_iterator& cell,
    scratch::SystemCell<dim>& scratch, copy::LocalData<dim>& copy,
    bool assemble_matrix, bool assemble_velocity, bool assemble_pressure) {
  //---------------------------------------------------------------------------
  //
  // Solve using the projection method with second order forward differences
  //
  // Note that the matrices are relatively straight-forward if we are using a
  // first order method
  //
  // M U^* = U^N + dt * ( - U^N \cdot \nabla U^N + \nu \nabla^2 U^N)
  // \nabla^2 p^{n+1} = \nabla\cdot U^*
  //
  // u^{n+1} = u^* - \nabla p^{n+1}
  //
  // An IMEX scheme would be better, with the Laplacian stepped implicitly
  //
  //---------------------------------------------------------------------------
  copy.local_matrix = 0;
  copy.local_rhs = 0;

  scratch.fe_vals.reinit(cell);

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
  scratch.fe_vals[velocity].get_function_divergences(ptr_system_handler->sol_0,
                                                     scratch.u_div_0);
  scratch.fe_vals[velocity].get_function_laplacians(ptr_system_handler->sol_0,
                                                    scratch.u_lap_0);

  for (uint q = 0; q < n_q_points; q++) {

    for (uint k = 0; k < dofs_per_cell; k++) {
      scratch.u_shape_val[k] = scratch.fe_vals[velocity].value(k, q);
      scratch.u_shape_grad[k] = scratch.fe_vals[velocity].gradient(k, q);

      scratch.p_shape_val[k] = scratch.fe_vals[pressure].value(k, q);
      scratch.p_shape_grad[k] = scratch.fe_vals[pressure].gradient(k, q);
    }

    // gradients[q] is a vector of gradients of the field described by
    // fe_function at the 𝑞th quadrature point. The size of the vector accessed
    // by gradients[q] equals the number of components of the finite element,
    // i.e. gradients[q][c] returns the gradient of the 𝑐th vector component at
    // the 𝑞th quadrature point. Consequently, gradients[q][c][d] is the
    // derivative in coordinate direction 𝑑 of the 𝑐th vector component of the
    // vector field at quadrature point 𝑞 of the current cell.
    //
    // We want:
    //  u_i partial_i u_j
    //
    // The gradient is structured as:
    //  partial_j u_i
    //
    // Therefore, we need to take the transpose of u_grad_0, or just flip the
    // indices
    scratch.u_adv_0[q] = 0;
    for (uint i = 0; i < dim; i++) {
      for (uint j = 0; j < dim; j++) {
        scratch.u_adv_0[q][i] +=
            scratch.u_sol_0[q][i] * scratch.u_grad_0[q][j][i];
      }
    }

    for (uint i = 0; i < dofs_per_cell; i++) {

      if (assemble_matrix) {
        for (uint j = 0; j < dofs_per_cell; j++) {
          // <v_i, v_j> + <\nabla v_i, \nabla v_j>
          copy.local_matrix[i][j] +=
              (scratch.u_shape_val[i] * scratch.u_shape_val[j] +
               this->eqn_params.nu * scalar_product(scratch.u_shape_grad[i],
                                                    scratch.u_shape_grad[j])) *
              scratch.fe_vals.JxW(q);

          // <\nabla q_i, \nabla q_j>
          copy.local_matrix[i][j] += scratch.p_shape_grad[i] *
                                     scratch.p_shape_grad[j] *
                                     scratch.fe_vals.JxW(q);
        }
      }

      if (assemble_velocity) {
        // <v_i, u(q)\cdot\nabla u(q)>
        copy.local_rhs[i] += scratch.u_shape_val[i] * scratch.u_adv_0[q] *
                             scratch.fe_vals.JxW(q);
      } else if (assemble_pressure) {
        // <q_i, \nabla\cdot u(q)>
        copy.local_rhs[i] += (scratch.p_shape_val[i] * scratch.u_div_0[q]) *
                             scratch.fe_vals.JxW(q);
      }
    }
  }
}

template <int dim>
void Assembler<dim>::copy_cell(const copy::LocalData<dim>& data) {
  ptr_system_handler->constraints.distribute_local_to_global(
      data.local_matrix, data.local_rhs, data.local_dof_indices,
      ptr_system_handler->system_matrix, ptr_system_handler->rhs);
}

template <int dim> void Assembler<dim>::assemble_system() {

  ptr_system_handler->pcout << "\nRunning assembly" << std::endl;

  ptr_system_handler->rhs = 0;
  ptr_system_handler->system_matrix = 0;

  auto worker =
      [this](const typename DoFHandler<dim>::active_cell_iterator& cell,
             scratch::SystemCell<dim>& scratch, copy::LocalData<dim>& data) {
        this->assemble_cell(cell, scratch, data, true, true, false);
      };

  auto copier = [this](const copy::LocalData<dim>& data) {
    this->copy_cell(data);
  };

  using CellFilter =
      FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  WorkStream::run(CellFilter(IteratorFilters::LocallyOwnedCell(),
                             ptr_system_handler->dof_handler.begin_active()),
                  CellFilter(IteratorFilters::LocallyOwnedCell(),
                             ptr_system_handler->dof_handler.end()),
                  worker, copier,
                  scratch::SystemCell<dim>(
                      ptr_system_handler->fe_system,
                      QGauss<dim>(ptr_system_handler->fe_params.degree + 2),
                      update_values | update_gradients | update_JxW_values |
                          update_quadrature_points | update_hessians),
                  copy::LocalData<dim>(ptr_system_handler->fe_system));

  ptr_system_handler->rhs.compress(VectorOperation::add);
  ptr_system_handler->system_matrix.compress(VectorOperation::add);
}

} // namespace assembly

#endif
