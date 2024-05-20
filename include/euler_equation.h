#ifndef EULER_EQ
#define EULER_EQ

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_vector.h>

namespace assembly {

using namespace dealii;

namespace scratch {
struct cell_data {};
} // namespace scratch

namespace copy {
struct local_data {};
} // namespace copy

template <int dim> class EulerEquationAssembler {
public:
  EulerEquationAssembler<dim>(ConditionalOStream pcout);

private:
  friend class Solver;

  ConditionalOStream pcout;

  void perform_assembly(scratch::cell_data &scratch, copy::local_data &copy);
};

template <int dim>
EulerEquationAssembler<dim>::EulerEquationAssembler(ConditionalOStream pcout)
    : pcout(pcout) {}

} // namespace assembly

#endif
