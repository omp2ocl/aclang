---
layout: page
title: AClang
permalink: /
---

ACLang is an open source LLVM Clang based compiler that implements the
OpenMP Accelerator Model. It adds a new runtime library to LLVM/CLang
that supports OpenMP offloading to accelerators like GPUs. Kernel
functions are extracted from OpenMP annotated regions and are
dispatched as OpenCL or SPIR code to be loaded and compiled by OpenCL
drivers before being executed by the device. AClang leverages on the
ISL implementation of the polyhedral model to implement a multilevel
tiling optimization on the extracted kernels. AClang also provides a
vectorization pass developed specifically to exploit the vector
instructions available in OpenCL. This whole process is transparent
and does not require any programmer intervention.

## How it works

The version 4.0 of the  OpenMP standard introduces new directives that
enable the transfer of  computation to heterogeneous computing devices
(e.g.  GPUs  or  DSP).  We  use this  programming  model  to  transfer
computation to  an accelerator  that supports  OpenCL 1.2  or greater,
giving  to   the  programmer  the   ability  to  quickly   expand  the
computational power exploring their target devices.

The following example shows how ACLang works from a programmer
perspective. It presents two loops from the “Matrix Vector Product and
Transpose” (mvt) program of the Polybench benchmark suite after they
have been annotated with OpenMP 4.0 pragmas:

{% highlight C %}
// Problem size
#define N 8192

void mvt_gpu(float* a, float* x1, float* x2,
                       float* y1, float* y2)
{
  int i,j;

  #pragma omp target data device(GPU) map(to: a[ :N*N])
  {
    #pragma omp target map(to: y1[:N]) map(tofrom: x1[:N])
    #pragma omp parallel for simd
    for (i=0; i<N; i++)
      for (j=0; j<N; j++)
        x1[i] = x1[i] + a[i*N + j] * y1[j];

    #pragma omp target map(to: y2[:N]) map(tofrom: x2[:N])
    #pragma omp parallel for simd
    for (i=0; i<N; i++)
      for (j=0; j<N; j++)
        x2[i] = x2[i] + a[j*N + i] * y2[j];
  }
}
{% endhighlight %}

In the first loop the program computes the matrix vector
multiplication followed by the transpose between a and y1 storing the
result into vector x1. The second loop does a similar task for a, y2
and x2. As shown in Listing above, the target clause defines the portion
of the program that will be executed by the device (GPU in the
example). The map clause details the mapping of the data between the
host and the target device. For example in the first kernel of
Listing above inputs (a and y1) are mapped to the GPU, and array x1 is
mapped to/from the GPU. This means that array x1 is read and written
during the kernel execution in the GPU. This strategy offers maximal
flexibility to the developer to decide which part of the code is
profitable to run on which architecture.

Host code to perform data offloading in ACLang is handled
automatically during the LLVM/IR generation phase and occurs in
between the begin and end scopes of pragmas “omp target [data]
map”. Also, during this phase ACLang extracts annotated loops from the
compiler AST and transforms them into OpenCL kernels in source code
format. Moreover, ACLang also optimizes the extracted OpenCL
kernels. For example as shown in Listing below it tiles and vectorizes
the first loop of the code in example above transforming it to OpenCL
kernel with blocks and threads suitable to run on any GPU containing
vector instructions.

{% highlight C %}
__kernel void mvt_gpu_0(__global float *a,
                        __global float *x1,
                        __global float *y1) {
  int b0 = get_group_id(0);
  int t0 = get_local_id(0);
  __private float4 _ft0;
  __private float4 _ft1;
  __private float4 _ft2;
  _ft0 = vload4(0, &x1[(4*b0) + t0]);
  for (int c1 = 0; c1 <= 8191; c1 += 4){
    _ft1 = vload4(0, &a[((32768*b0) + (8192*t0)) + c1]);
    _ft2 = vload4(0, &y1[c1]);
    _ft0 = _ft0 + (_ft1 * _ft2);
  }
  x1[(4*b0) + t0] = ((_ft0.x + _ft0.y) + _ft0.z) + _ft0.w;
}
{% endhighlight %}

You can look at [Unibench](https://github.com/omp2ocl/Unibench)
repository if you want to see more examples.


## Documentation, Installation, Configuration

All the information is provided [in the Wiki](https://github.com/omp2ocl/aclang/wiki).
