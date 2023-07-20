function trtrsy!(uplo::Char, RL::AbstractMatrix{T}, S::AbstractMatrix{T}) where {T}
    # Algorithm: 'Matrix Inversion Using Cholesky Decomposition', Aravindh Krishnamoorthy, Deepak Menon, arXiv:1111.4144.
    N = size(RL,1)
    if uplo == 'U'
        @inbounds begin
            for j=N:-1:1
                # k = N,...,j+1
                for k=N:-1:j+1
                    for i=1:j
                        S[i,j] = S[i,j] - RL[i,k]*conj(S[j,k])
                    end
                end
                # k = j,...,1
                for k=j:-1:1
                    S[k,j] = S[k,j]/RL[k,k]
                    for i=1:k-1
                        S[i,j] = S[i,j] - RL[i,k]*S[k,j]
                    end
                end
            end
        end
        S = Hermitian(S, :U)
    else # if uplo == 'L'
        @inbounds begin
            for i=1:N
                for j=1:i
                    for k=1:i-1
                        S[i,j] = S[i,j] - RL[i,k]*S[k,j]
                    end
                    S[i,j] = i == j ? convert(T, real(S[i,j]/RL[i,i])) : S[i,j]/RL[i,i]
                    S[j,i] = conj(S[i,j])
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
        Y = trtrsy!('L', B.U', inv(Tridiagonal(B.U)*B.D))[p,p]
    else # if B.uplo == 'L'
        Y = trtrsy!('U', B.L', inv(Tridiagonal(B.L)*B.D))[p,p]
    end
    return Y
end

function invchol(C::Cholesky{T}) where {T}
    # Algorithm: 'Matrix Inversion Using Cholesky Decomposition', Aravindh Krishnamoorthy, Deepak Menon, arXiv:1111.4144.
    L = copy(C.L)
    return trtrsy!('L', L, Matrix{T}(inv(Diagonal(L))))
end
