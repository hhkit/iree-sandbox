func.func @attention(
    %q : tensor<1024x128xf32>,
    %k : tensor<1024x128xf32>,
    %v : tensor<1024x128xf32>
) -> tensor<1024x128xf32>
{
    %s = tensor.empty(): tensor<1024x1024xf32>
    %m = tensor.empty(): tensor<1024x1024xf32>
    %o = tensor.empty(): tensor<1024x128xf32>

    linalg.matmul_transpose_b
        ins(%q, %k : tensor<1024x128xf32>, tensor<1024x128xf32>) 
        outs(%s: tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    linalg.softmax
        dimension(1)
        ins(%s:  tensor<1024x1024xf32>)
        outs(%m:  tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    linalg.matmul
        ins(%m, %v: tensor<1024x1024xf32>, tensor<1024x128xf32>)
        outs(%o: tensor<1024x128xf32>) -> tensor<1024x128xf32>
    func.return %o: tensor<1024x128xf32>
}
