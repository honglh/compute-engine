#include <gmock/gmock.h>

#include <array>

#include "larq_compute_engine/core/bgemm_functor.h"
#include "larq_compute_engine/core/bgemm_kernels_arm32.h"

namespace compute_engine {
namespace testing {

namespace ce = compute_engine;
using ce::core::Layout;

TEST(BGemmArm32Tests, BGemmArm32BP4x4) {
  using SrcScalar = std::uint32_t;
  using DstScalar = float;
  using AccumScalar = std::int32_t;

  ce::core::ReferenceBGemmFunctor<std::uint32_t, Layout::RowMajor,
                                  std::uint32_t, Layout::ColMajor, float,
                                  Layout::ColMajor, std::int32_t>
      bgemm_functor;

  /*
   * bgemm of a im2col output packed matrix
   * where :
   * a.shape = (m x k) and b.shape = (k x n)
   * a.layout = rowMajor and b.layout = colMajor and c.layout = colMajor
   *
   * Note: this test case is to test 32-bit packed for 4x4 kerenel. Thus
   * the lhs and rhs must sastify these conditions so we can compare results
   * with bgemm functor:
   *
   * m, n must be multiple of 4
   * k must be 4
   *
   * In real use, these conditions are sastified by the bitpacking process
   * outside the bgemm kernel function.
   */
  const int m = 12;
  const int k = 4;
  const int n = 12;

  const int lda = k;
  const int ldb = k;
  const int ldc = m;

  const int a_size = m * k;
  const int b_size = k * n;
  const int c_size = m * n;

  std::array<uint32_t, a_size> a;
  std::array<uint32_t, b_size> b;

  // Fill with random bits
  for (size_t i = 0; i < a_size; ++i) {
    a[i] = rand();
  }
  for (size_t i = 0; i < b_size; ++i) {
    b[i] = rand();
  }

  std::array<float, c_size> c_expected;
  bgemm_functor(m, n, k, a.data(), lda, b.data(), ldb, c_expected.data(), ldc);
  std::array<float, c_size> c;

  SrcScalar* lhs = a.data();
  SrcScalar* rhs = b.data();
  DstScalar* dst = c.data();

  const int lhs_rows = m;
  const int rhs_cols = n;
  const int depth = k;
  const int dst_rows = m;
  const int dst_cols = n;

  /*4x4 kernel layout */
  using LhsLayout = FixedKernelLayout<Order::kRowMajor, 4, 4>;
  using RhsLayout = FixedKernelLayout<Order::kColMajor, 4, 4>;
  using DstLayout = FixedKernelLayout<Order::kColMajor, 4, 4>;

  BinaryKernelParams<LhsLayout::kRows, RhsLayout::kCols, SrcScalar> params_,
      *params = &params_;
  ;
  params->lhs_base_ptr = lhs;
  params->rhs_base_ptr = rhs;
  params->dst_base_ptr = dst;
  ;

  /*
   * Our bgemm kernel has fused operation for output transformation
   * while the reference kernel does not. Thus for unit testing the
   * MAC operation only, we use some dummy constants that allow us
   * to check results against reference kernel.
   */
  std::array<float, dst_rows> post_activation_multiplier;
  std::array<float, dst_rows> post_activation_bias;
  for (size_t i = 0; i < dst_rows; ++i) {
    post_activation_multiplier[i] = 1.0;
  }
  for (size_t i = 0; i < dst_rows; ++i) {
    post_activation_bias[i] = 0.0;
  }
  params->clamp_min = std::numeric_limits<AccumScalar>::min();
  params->clamp_max = std::numeric_limits<AccumScalar>::max();
  ;
  params->flags &= ~RUY_ASM_FLAG_HAS_BIAS;
  params->post_activation_bias = post_activation_bias.data();
  params->post_activation_multiplier = post_activation_multiplier.data();

  /*
   * bgemm parameters
   */
  params->start_row = 0;
  params->last_row = lhs_rows - LhsLayout::kRows;
  params->start_col = 0;
  params->last_col = rhs_cols - RhsLayout::kCols;
  params->dst_rows = dst_rows;
  params->dst_cols = dst_cols;
  params->lhs_stride = depth * sizeof(SrcScalar);
  params->rhs_stride = depth * sizeof(SrcScalar);
  params->dst_stride = dst_rows * sizeof(DstScalar);
  params->depth = depth;
  params->backtransform_add = depth * std::numeric_limits<SrcScalar>::digits;

  BinaryKernelNeonOutOfOrder32BP4x4(*params);
  EXPECT_THAT(c, ::testing::ElementsAreArray(c_expected));
}

}  // end namespace testing
}  // end namespace compute_engine
