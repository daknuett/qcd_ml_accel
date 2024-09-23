#include <torch/extension.h>

#ifndef _OPENMP
#define _OPENMP
#endif
#include <omp.h>
#include <ATen/ParallelOpenMP.h>
#include <vector>

namespace qcd_ml_accel {

    at::Tensor v_pool4d_cpu(at::Tensor v_spincolor_field, at::Tensor block_size)
    {
        TORCH_CHECK(v_spincolor_field.dtype() == at::kComplexDouble);
        TORCH_CHECK(block_size.dtype() == at::kLong);

        at::Tensor block_size_contig = block_size.contiguous();
        TORCH_CHECK(v_spincolor_field.sizes().size() == 6);     // 4d + spin + color
        TORCH_CHECK(block_size_contig.sizes() == at::ArrayRef({(long)4}));            // This is list of block sizes
        TORCH_CHECK(block_size_contig.sizes()[0] == 4);

        const int64_t * const blk_sz_ptr = block_size_contig.data_ptr<int64_t>();

        for(int64_t mu = 0; mu < block_size_contig.sizes()[0]; mu++)
        {
            TORCH_CHECK(v_spincolor_field.sizes()[mu] % blk_sz_ptr[mu] == 0); // make sure partition works.
        }

        std::vector<int64_t> L_coarse;
        for(int64_t mu = 0; mu < block_size_contig.sizes()[0]; mu++)
        {
            L_coarse.push_back(v_spincolor_field.sizes()[mu] / blk_sz_ptr[mu]);  // compute coarse grid size
        }

        L_coarse.push_back(v_spincolor_field.sizes()[4]);
        L_coarse.push_back(v_spincolor_field.sizes()[5]);

        at::Tensor vsfield_contig = v_spincolor_field.contiguous();
        at::Tensor result = torch::zeros(L_coarse, vsfield_contig.options());


        const c10::complex<double> * const vsfield_ptr = vsfield_contig.data_ptr<c10::complex<double>>();
        c10::complex<double> * const result_ptr = result.data_ptr<c10::complex<double>>();


        int64_t const stride_x_c = L_coarse[1] 
                                * L_coarse[2]
                                * L_coarse[3]
                                * L_coarse[4]
                                * L_coarse[5];
        int64_t const stride_y_c = L_coarse[2]
                                * L_coarse[3]
                                * L_coarse[4]
                                * L_coarse[5];
        int64_t const stride_z_c = L_coarse[3]
                                * L_coarse[4]
                                * L_coarse[5];
        int64_t const stride_t_c = L_coarse[4]
                                * L_coarse[5];
        int64_t const stride_s_c = L_coarse[5];

        int64_t const stride_x = vsfield_contig.sizes()[1] 
                                * vsfield_contig.sizes()[2]
                                * vsfield_contig.sizes()[3]
                                * vsfield_contig.sizes()[4]
                                * vsfield_contig.sizes()[5];
        int64_t const stride_y = vsfield_contig.sizes()[2]
                                * vsfield_contig.sizes()[3]
                                * vsfield_contig.sizes()[4]
                                * vsfield_contig.sizes()[5];
        int64_t const stride_z = vsfield_contig.sizes()[3]
                                * vsfield_contig.sizes()[4]
                                * vsfield_contig.sizes()[5];
        int64_t const stride_t = vsfield_contig.sizes()[4]
                                * vsfield_contig.sizes()[5];
        int64_t const stride_s = vsfield_contig.sizes()[5];

        /* Loop over coarse grid */
        //for(int64_t xc = 0; xc < L_coarse[0]; xc ++)
        at::parallel_for(0, L_coarse[0], 1, [&](int64_t start, int64_t end){
        for(int64_t xc = start; xc < end; xc ++)
        {
            //for(int64_t yc = 0; yc < L_coarse[1]; yc ++)
            at::parallel_for(0, L_coarse[1], 1, [&](int64_t start, int64_t end){
            for(int64_t yc = start; yc < end; yc ++)
            {
                for(int64_t zc = 0; zc < L_coarse[2]; zc ++)
                {
                    for(int64_t tc = 0; tc < L_coarse[3]; tc ++)
                    {
                        /* Loop over block */

                        for(int64_t xi = 0; xi < blk_sz_ptr[0]; xi ++)
                        {
                            for(int64_t yi = 0; yi < blk_sz_ptr[1]; yi ++)
                            {
                                for(int64_t zi = 0; zi < blk_sz_ptr[2]; zi ++)
                                {
                                    for(int64_t ti = 0; ti < blk_sz_ptr[3]; ti ++)
                                    {
                                        /* Loop over spin and color index */

                                        for(int64_t si = 0; si < L_coarse[4]; si ++)
                                        {
                                            for(int64_t ci = 0; ci < L_coarse[5]; ci ++)
                                            {
                                                result_ptr[xc * stride_x_c
                                                        + yc * stride_y_c
                                                        + zc * stride_z_c
                                                        + tc * stride_t_c
                                                        + si * stride_s_c
                                                        + ci] += vsfield_ptr[(xc * blk_sz_ptr[0] + xi) * stride_x
                                                                            + (yc * blk_sz_ptr[1] + yi) * stride_y
                                                                            + (zc * blk_sz_ptr[2] + zi) * stride_z
                                                                            + (tc * blk_sz_ptr[3] + ti) * stride_t
                                                                            + si * stride_s
                                                                            + ci];


                                            }
                                        }

                                    }
                                }
                            }
                        }

                    }
                }
            }});
        }});

        return result;
    }

    at::Tensor v_unpool4d_cpu(at::Tensor v_spincolor_field, at::Tensor block_size)
    {
        TORCH_CHECK(v_spincolor_field.dtype() == at::kComplexDouble);
        TORCH_CHECK(block_size.dtype() == at::kLong);

        at::Tensor block_size_contig = block_size.contiguous();
        TORCH_CHECK(v_spincolor_field.sizes().size() == 6);     // 4d + spin + color
        TORCH_CHECK(block_size_contig.sizes() == at::ArrayRef({(long) 4}));            // This is list of block sizes
        TORCH_CHECK(block_size_contig.sizes()[0] == 4);

        const int64_t * const blk_sz_ptr = block_size_contig.data_ptr<int64_t>();

        std::vector<int64_t> L_fine;
        for(int64_t mu = 0; mu < block_size_contig.sizes()[0]; mu++)
        {
            L_fine.push_back(v_spincolor_field.sizes()[mu] * blk_sz_ptr[mu]);  // compute coarse grid size
        }

        L_fine.push_back(v_spincolor_field.sizes()[4]);
        L_fine.push_back(v_spincolor_field.sizes()[5]);

        at::Tensor vsfield_contig = v_spincolor_field.contiguous();
        at::Tensor result = torch::zeros(L_fine, vsfield_contig.options());
        const c10::complex<double> * const vsfield_ptr = vsfield_contig.data_ptr<c10::complex<double>>();
        c10::complex<double> * const result_ptr = result.data_ptr<c10::complex<double>>();

        int64_t const stride_x_c = vsfield_contig.sizes()[1] 
                                * vsfield_contig.sizes()[2]
                                * vsfield_contig.sizes()[3]
                                * vsfield_contig.sizes()[4]
                                * vsfield_contig.sizes()[5];
        
        int64_t const stride_y_c = vsfield_contig.sizes()[2]
                                * vsfield_contig.sizes()[3]
                                * vsfield_contig.sizes()[4]
                                * vsfield_contig.sizes()[5];

        int64_t const stride_z_c = vsfield_contig.sizes()[3]
                                * vsfield_contig.sizes()[4]
                                * vsfield_contig.sizes()[5];
        int64_t const stride_t_c = vsfield_contig.sizes()[4]
                                * vsfield_contig.sizes()[5];
        int64_t const stride_s_c = vsfield_contig.sizes()[5];

        int64_t const stride_x = L_fine[1] 
                                * L_fine[2]
                                * L_fine[3]
                                * L_fine[4]
                                * L_fine[5];
        
        int64_t const stride_y = L_fine[2]
                                * L_fine[3]
                                * L_fine[4]
                                * L_fine[5];

        int64_t const stride_z = L_fine[3]
                                * L_fine[4]
                                * L_fine[5];
        int64_t const stride_t = L_fine[4]
                                * L_fine[5];
        int64_t const stride_s = L_fine[5];

        at::parallel_for(0, vsfield_contig.sizes()[0], 1, [&](int64_t start, int64_t end){
        for(int64_t xc = start; xc < end; xc ++)
        {
            at::parallel_for(0, vsfield_contig.sizes()[1], 1, [&](int64_t start, int64_t end){
            for(int64_t yc = start; yc < end; yc ++)
            {
                for(int64_t zc = 0; zc < vsfield_contig.sizes()[2]; zc++)
                {
                    for(int64_t tc = 0; tc < vsfield_contig.sizes()[3]; tc++)
                    {
                        for(int64_t si = 0; si < vsfield_contig.sizes()[4]; si++)
                        {
                            for(int64_t ci = 0; ci < vsfield_contig.sizes()[5]; ci++)
                            {

                                for(int64_t xi = 0; xi < blk_sz_ptr[0]; xi++)
                                {
                                    for(int64_t yi = 0; yi < blk_sz_ptr[1]; yi++)
                                    {
                                        for(int64_t zi = 0; zi < blk_sz_ptr[2]; zi++)
                                        {
                                            for(int64_t ti = 0; ti < blk_sz_ptr[3]; ti++)
                                            {
                                                result_ptr[(xc * blk_sz_ptr[0] + xi) * stride_x
                                                        + (yc * blk_sz_ptr[1] + yi) * stride_y
                                                        + (zc * blk_sz_ptr[2] + zi) * stride_z
                                                        + (tc * blk_sz_ptr[3] + ti) * stride_t
                                                        + si * stride_s
                                                        + ci] = vsfield_ptr[xc * stride_x_c
                                                                            + yc * stride_y_c
                                                                            + zc * stride_z_c
                                                                            + tc * stride_t_c
                                                                            + si * stride_s_c
                                                                            + ci];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }});
        }});

        return result;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

    TORCH_LIBRARY(qcd_ml_accel, m) {
       m.def("v_pool4d(Tensor v_spincolor_field, Tensor block_size) -> Tensor");
       m.def("v_unpool4d(Tensor v_spincolor_field, Tensor block_size) -> Tensor");
     }



    TORCH_LIBRARY_IMPL(qcd_ml_accel, CPU, m) {
      m.impl("v_pool4d", &v_pool4d_cpu);
      m.impl("v_unpool4d", &v_unpool4d_cpu);
    }

}

