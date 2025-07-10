
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
      %root: !transform.any_op{transform.readonly}
    ) {
    %matmul, %ew1, %ew2 = transform.collect_matching @match_matmul_elemwise in %root
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %ew = transform.merge_handles %ew1, %ew2
      : !transform.any_op
    transform.include @print_matmul failures(propagate)  (%matmul)
      : (!transform.any_op) -> ()

    transform.yield
  }

  transform.named_sequence @match_elemwise(
      %entry: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %entry ["linalg.elementwise"]
      : !transform.any_op
    transform.yield %entry : !transform.any_op
  }
  transform.named_sequence @match_matmul(
      %entry: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %entry ["linalg.matmul"] : !transform.any_op
    transform.yield %entry : !transform.any_op
  }

  transform.named_sequence @match_matmul_elemwise(
    %last: !transform.any_op {transform.readonly}
  ) -> (!transform.any_op, !transform.any_op, !transform.any_op)
  {
    transform.match.operation_name %last ["linalg.elementwise"] : !transform.any_op
    %middle = transform.get_producer_of_operand %last[0] : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name %middle ["linalg.elementwise"] : !transform.any_op
    %matmul = transform.get_producer_of_operand %middle[0] : (!transform.any_op) -> !transform.any_op
    transform.match.operation_name %matmul ["linalg.matmul"] : !transform.any_op
    transform.yield %matmul, %middle, %last : !transform.any_op, !transform.any_op, !transform.any_op
  }

  // This is a rewriter sequence.
  transform.named_sequence @print_elemwise(
      %elemwise_binary: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at
      %elemwise_binary, "elementwise binary" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @print_matmul(
      %matmul: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %matmul, "matmul" : !transform.any_op
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