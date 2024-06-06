#ifndef EULER_EQ
#define EULER_EQ

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_vector.h>

#include "system_handler.h"

namespace assembly {

using namespace dealii;

namespace scratch {
struct system_cell {};
struct precon_cell {};
} // namespace scratch

namespace copy {
struct local_data {};
struct local_precon {};
} // namespace copy

template <int dim> class EulerEquationAssembler {
public:
  EulerEquationAssembler<dim>(MPI_Comm mpi_comm, ConditionalOStream pcout,
                              grid::SystemHandler<dim> *ptr_system_handler);

  MPI_Comm mpi_comm;
  ConditionalOStream pcout;

  grid::SystemHandler<dim> *ptr_system_handler;

  void perform_assembly(scratch::system_cell &scratch, copy::local_data &copy);
  void perform_precon_assembly(scratch::precon_cell &scratch,
                               copy::local_precon &copy);
};

template <int dim>
EulerEquationAssembler<dim>::EulerEquationAssembler(
    MPI_Comm mpi_comm, ConditionalOStream pcout,
    grid::SystemHandler<dim> *ptr_system_handler)
    : mpi_comm(mpi_comm), pcout(pcout), ptr_system_handler(ptr_system_handler) {
}

} // namespace assembly

#endif
