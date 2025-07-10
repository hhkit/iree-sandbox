func.func @test (%0: vector<8xf32>)
{
    %1 = arith.constant dense<1.0> : vector<8xf32>
    %init = arith.constant 0.0 : f32
    %result = vector.contract {
        indexing_maps = [
            affine_map<(i) -> (i)>,
            affine_map<(i) -> (i)>,
            affine_map<(i) -> ()>
        ],
        iterator_types = ["reduction"]
    } %0, %1, %init : vector<8xf32>, vector<8xf32> into f32
    return
}

func.func @matmul (%0: vector<4x3xf32>, %1: vector<3x4xf32>) -> vector<4x4xf32>
{
    %init = arith.constant dense<0.0> : vector<4x4xf32>
    %res = vector.contract {
        indexing_maps = [
            affine_map<(i,j,k)->(i, k)>,
            affine_map<(i,j,k)->(k, j)>,
            affine_map<(i,j,k)->(i, j)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
    } %0, %1, %init : vector<4x3xf32>, vector<3x4xf32> into vector<4x4xf32>
    return %res : vector<4x4xf32>
}

func.func @linalg_matmul (%lhs: memref<4x3xf32>, %rhs: memref<3x4xf32>) -> memref<4x4xf32> {
    %init = arith.constant dense<0.0> : memref<4x4xf32>
    linalg.generic {
        indexing_maps = [
            affine_map<(i,j,k)->(i, k)>,
            affine_map<(i,j,k)->(k, j)>,
            affine_map<(i,j,k)->(i, j)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%lhs, %rhs: memref<4x3xf32>, memref<3x4xf32>)
      outs(%init: memref<4x4xf32>) {
        ^bb(%lhs_elem: f32, %rhs_elem: f32, %init_elem: f32):
            %0 = arith.mulf %lhs_elem, %rhs_elem : f32
            %1 = arith.addf %init_elem, %0: f32
            linalg.yield %1: f32
      }
    return %init : memref<4x4xf32>
}

func.func @relu (%0: memref<?xf32>) -> memref<?xf32> {
    %ind = arith.constant 0: index
    %dim = memref.dim %0, %ind: memref<?xf32>
    %init = memref.alloc(%dim): memref<?xf32> // allocate memory of equal dimension

    linalg.generic {
        indexing_maps = [ affine_map<(i)->(i)>, affine_map<(i)->(i)> ],
        iterator_types = ["parallel"]
    } ins(%0: memref<?xf32>)
      outs(%init: memref<?xf32>) {
        ^bb(%elem: f32, %_: f32): // second arg is meaningless?
            %c0 = arith.constant 0.0: f32
            %cmp = arith.cmpf ogt, %elem, %c0 : f32
            %1 = arith.select %cmp, %elem, %c0 : f32
            linalg.yield %1 : f32
      }
    return %init: memref<?xf32>
}

func.func @tiled_matmul(%lhs: tensor<40x30xf32>, %rhs: tensor<30x40xf32>) -> tensor<40x40xf32> {
    %init = arith.constant dense<0.0> : tensor<40x40xf32>

    scf.forall
        (%i, %j) in (10, 10)
        shared_outs(%shared = %init)
        -> (tensor<40x40xf32>) {
            %i2 = affine.apply affine_map<(d0) -> (d0 * 4)>(%i)
            %j2 = affine.apply affine_map<(d0) -> (d0 * 4)>(%j)

            // what the fuck are these numbers
            // tensor.extract_slice PTR DIM STEP
            %lhs_slice = tensor.extract_slice %lhs[%i2, 0] [4, 30] [1, 1]
                : tensor<40x30xf32> to tensor<4x30xf32>
            %rhs_slice = tensor.extract_slice %rhs[0, %j2] [30, 4] [1, 1]
                : tensor<30x40xf32> to tensor<30x4xf32>
            %result_slice = tensor.extract_slice %shared[%i2, %j2] [4, 4] [1,1]
                : tensor<40x40xf32> to tensor<4x4xf32>
            
            // matmul
            // %matmul = linalg.generic {
            //     indexing_maps = [
            //         affine_map<(i,j,k)-> (i,k)>,
            //         affine_map<(i,j,k)-> (k,j)>,
            //         affine_map<(i,j,k)-> (i,j)>
            //     ],
            //     iterator_types = ["parallel", "parallel", "reduction"]
            // } ins (%lhs_slice, %rhs_slice: tensor<4x30xf32>, tensor<30x4xf32>)
            //   outs(%result_slice: tensor<4x4xf32>) {
            //     ^bb(%lhs_elem: f32, %rhs_elem: f32, %init_elem: f32):
            //         %0 = arith.mulf %lhs_elem, %rhs_elem : f32
            //         %1 = arith.addf %init_elem, %0: f32
            //         linalg.yield %1: f32
            //   } -> tensor<4x4xf32>
            %matmul = linalg.matmul 
                ins (%lhs_slice, %rhs_slice: tensor<4x30xf32>, tensor<30x4xf32>)
                outs(%result_slice: tensor<4x4xf32>) -> tensor<4x4xf32>
            

            scf.forall.in_parallel {
                tensor.parallel_insert_slice %matmul into %shared[%i2, %j2] [4, 4] [1, 1]
                    : tensor<4x4xf32> into tensor<40x40xf32>
            } 
        }

    return %init : tensor<40x40xf32>
}