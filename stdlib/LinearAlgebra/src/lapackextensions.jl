function trtrsy(uplo::Char, RL::AbstractMatrix{T}, S::AbstractMatrix{T}) where {T}
    trtrsy!(uplo, copy(RL), copy(S))
end

function trtrsy!(uplo::Char, RL::AbstractMatrix{T}, S::AbstractMatrix{T}) where {T}
    # Algorithm: 'Matrix Inversion Using Cholesky Decomposition', Aravindh Krishnamoorthy, Deepak Menon, arXiv:1111.4144.
    N = size(RL,1)
    RL = Matrix{T}(RL)
    S = Matrix{T}(S)
    if uplo == 'U'
        @inbounds begin
            for j=N:-1:1
                # Compute element (j,j)
                S[j,j] = convert(T, real(S[j,j]/RL[j,j]))
    
                # Compute elements (1,j)--(j-1,j)
                for i=j-1:-1:1
                    # for k=1:i
                    #     S[k,j] = S[k,j] - RL[k,i+1]*S[i+1,j]
                    # end
                    GC.@preserve S RL BLAS.axpy!(i, -S[i+1,j], pointer(RL, i*N+1), 1, pointer(S, (j-1)*N+1), 1)
                    S[i,j] = S[i,j]/RL[i,i]
                end
            
                # Update elements from other columns
                for i=j-1:-1:1
                    # for k=1:i
                    #     S[k,i] = S[k,i] - RL[k,j]*conj(S[i,j])
                    # end
                    GC.@preserve S RL BLAS.axpy!(i, -conj(S[i,j]), pointer(RL, (j-1)*N+1), 1, pointer(S, (i-1)*N+1), 1)
                end
            end
        end
        S = Hermitian(S, :U)
    else # if uplo == 'L'
        @inbounds begin
            for i=1:N
                # Compute element (i,i)
                S[i,i] = convert(T, real(S[i,i]/RL[i,i]))
    
                # Compute elements (i,1)--(i,i-1)
                for j=i+1:N
                    # for k=j:N
                    #     S[k,i] = S[k,i] - RL[k,j-1]*S[j-1,i]
                    # end
                    GC.@preserve S RL BLAS.axpy!(N-j+1, -S[j-1,i], pointer(RL, (j-2)*N+j), 1, pointer(S, (i-1)*N+j), 1)
                    S[j,i] = S[j,i]/RL[j,j]
                end
            
                # Update elements from other row
                for j=i+1:N
                    # for k=j:N
                    #     S[k,j] = S[k,j] - RL[k,i]*conj(S[j,i])
                    # end
                    GC.@preserve S RL BLAS.axpy!(N-j+1, -conj(S[j,i]), pointer(RL, (i-1)*N+j), 1, pointer(S, (j-1)*N+j), 1)
                end
            end
        end
        S = Hermitian(S, :L)
    end
    return S
end

function invbk(B::BunchKaufman{T}) where {T}
    p = invperm(B.p)
    # Algorithm based on: 'Matrix Inversion Using Cholesky Decomposition', Aravindh Krishnamoorthy, Deepak Menon, arXiv:1111.4144.
    if B.uplo == 'U'
        Y = trtrsy('L', B.U', inv(Tridiagonal(B.U)*B.D))[p,p]
    else # if B.uplo == 'L'
        Y = trtrsy('U', B.L', inv(Tridiagonal(B.L)*B.D))[p,p]
    end
    return Y
end

function invchol(C::Cholesky{T}) where {T}
    # Algorithm: 'Matrix Inversion Using Cholesky Decomposition', Aravindh Krishnamoorthy, Deepak Menon, arXiv:1111.4144.
    R = C.U
    return trtrsy('U', R, inv(Diagonal(R)))
end
