
func.func @fc_relu(
  %lhs:  tensor<512x512xf32>,
  %rhs:  tensor<512x512xf32>,
  %bias: tensor<512x512xf32>,
  %out:  tensor<512x512xf32>
) -> tensor<512x512xf32> {
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%out: tensor<512x512xf32>) -> tensor<512x512xf32>
  
  %ew_bias = linalg.elementwise kind=#linalg.elementwise_kind<add>
    ins(%matmul, %bias: tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%out: tensor<512x512xf32>) -> tensor<512x512xf32>

  %c0 = arith.constant dense<0.0> : tensor<512x512xf32>
  %relu = linalg.elementwise kind=#linalg.elementwise_kind<max_signed>
    ins(%ew_bias, %c0: tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%out: tensor<512x512xf32>) -> tensor<512x512xf32>

  return %relu : tensor<512x512xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op,
      %arg1: !transform.op<"linalg.matmul">,
      %arg2: !transform.op<"linalg.elementwise">
    ) {
    // maps exactly to %ew_bias and %relu in previous
    //   there should be as many results and there are payload ops
    //   if there are more, the extra are mapped to the 'overflow_result' index
    %add, %max = transform.split_handle %arg2
      : (!transform.op<"linalg.elementwise">) -> (!transform.any_op, !transform.any_op)
    // %add -> %ew_bias
    // %max -> %relu

    %tiled_max, %loop = transform.structured.tile_using_forall %max tile_sizes [8, 32]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // we tile the %relu operation into tiles of size 8x32
    //   %tiled_max -> the inner tiled operation
    //   %loop      -> the handle to the loop

    %add_fused, %loop_0 = transform.structured.fuse_into_containing_op %add into %loop
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    // we want to fuse %ew_bias into the %loop
    //   q: what handles are now invalidated?
    //   a:   %add was fused into the loop, and is now invalidated
    //  %add_fused -> %ew_bias IN the loop
    //  %loop_0 = %loop -> the scf.forall

    %matmul_fused, %loop_1 = transform.structured.fuse_into_containing_op %arg1 into %loop_0
      : (!transform.op<"linalg.matmul">, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    // we want to fuse the %matmul into the %loop
    //  %matmul_fused             -> the matmul in the loop
    //  %loop_1 = %loop_0 = %loop -> the scf.forall

    // now that we have tiled the ops
    //   we want to further tile the %matmul_fused into 4x4 matmuls
    %tiled_2, %loop_2 = transform.structured.tile_using_forall %add_fused tile_sizes [4, 4]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %matmul_fused_2, %loop_3 =
        transform.structured.fuse_into_containing_op %matmul_fused into %loop_2
          : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    
    // you cannot do it the other way (fuse the add into the matmul)
    //   error: could not find next producer to fuse into container
    // what is a producer?

    %_, %outline_target = transform.structured.tile_using_forall %tiled_2 tile_sizes [1]
          : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.fuse_into_containing_op %matmul_fused_2 into %outline_target
          : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
    %func, %call = transform.loop.outline %outline_target
                   {func_name = "outlined"}
        : (!transform.any_op) -> (!transform.any_op, !transform.op<"func.call">)

    // this would work if you have an intrinsic called "microkernel"
    // transform.my.change_call_target %call, "microkernel" : !transform.op<"func.call">

    transform.yield
  }

  transform.sequence failures(propagate) {
    ^bb0(%arg0: !transform.any_op,
        %arg1: !transform.op<"linalg.matmul">,
        %arg2: !transform.op<"linalg.elementwise">):
    transform.debug.emit_remark_at %arg1, "matmul" : !transform.op<"linalg.matmul">
    transform.debug.emit_remark_at %arg2, "elemwise_binaries" : !transform.op<"linalg.elementwise">
    transform.yield
  }
}