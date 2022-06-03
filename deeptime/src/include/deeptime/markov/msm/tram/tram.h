//
// Created by Maaike on 15/11/2021.
//

#pragma once

#include <cstdio>
#include <cassert>
#include <utility>
#include "tram_input.h"
#include "dtram_input.h"

namespace deeptime::markov::tram {

namespace detail {

template<typename dtype>
constexpr static dtype prior() { return 0.0; }

template<typename dtype>
constexpr static dtype logPrior() { return -std::numeric_limits<dtype>::infinity(); }
}

template<typename inputtype>
struct UpdateBiasedConfEnergies;

template<typename dtype>
struct UpdateBiasedConfEnergies<TRAMInput<dtype>> {
    static void compute() {

    }
};

// natural logarithm of the statistical weight per sample, log \mu^k(x).
// If thermState =-1, this is the unbiased statistical sample weight, log \mu(x).
// This method gets called by pybind! Either directly or via the call to computeloglikelihood.
// TODO move this to a separate file? It is not used during (d)TRAM estimation
template<typename dtype>
static const std::vector<dtype> computeSampleWeightsLog(const DTraj &dtraj,
                                                        const BiasMatrix<dtype> &biasMatrix,
                                                        const np_array<dtype> &thermStateEnergies,
                                                        const np_array_nfc<dtype> &modifiedStateCountsLog,
                                                        StateIndex thermStateIndex) {
    auto nThermStates = static_cast<StateIndex>(thermStateEnergies.size());

    std::vector<dtype> sampleWeights(dtraj.size());
    std::vector<dtype> scratch(nThermStates);

    ArrayBuffer<DTraj, 1> dtrajBuf{dtraj};
    ArrayBuffer<BiasMatrix<dtype>, 2> biasMatrixBuf{biasMatrix};
    ArrayBuffer<np_array_nfc<dtype>, 2> modifiedStateCountsLogBuf{modifiedStateCountsLog};
    ArrayBuffer<np_array_nfc<dtype>, 1> thermStateEnergiesBuf{thermStateEnergies};

    for (auto x = 0; x < dtraj.size(); ++x) {
        auto i = dtrajBuf(x);
        if (i < 0) {
            // The sample weight has no meaning, as it does not exist in the connected set. The value is set to -inf
            // so that the weight is zero and the sample does not contribute to the output when computing observables.
            sampleWeights[x] = -std::numeric_limits<dtype>::infinity();
        } else {
            int o = 0;
            for (StateIndex l = 0; l < nThermStates; ++l) {
                if (modifiedStateCountsLogBuf(l, i) > -std::numeric_limits<dtype>::infinity()) {
                    scratch[o++] = modifiedStateCountsLogBuf(l, i) - biasMatrixBuf(x, l);
                }
            }
            auto log_divisor = numeric::kahan::logsumexp_sort_kahan_inplace(scratch.begin(), o);
            if (thermStateIndex == -1) {// get unbiased sample weight
                sampleWeights[x] = -log_divisor;
            } else { // get biased sample weight for given thermState index
                sampleWeights[x] = -biasMatrixBuf(x, thermStateIndex) - log_divisor
                                   + thermStateEnergiesBuf(thermStateIndex);
            }
        }
    }
    return sampleWeights;
}


// Compute the sample likelihood from the traminput. This method only gets called by the TRAM struct, not by pybind.
// This is the MBAR part of the TRAM likelihood, coming from the continuous sample configurations.
// log [ \prod_k \prod_i \prod_{x \in X_i} \mu(x) f_i^k ]
template<typename dtype>
static const dtype computeSampleDependentLikelihood(const TRAMInput<dtype> &input,
                                                    const np_array_nfc<dtype> &biasedConfEnergies,
                                                    const np_array_nfc<dtype> &modifiedStateCountsLog) {
    auto nThermStates = input.nThermStates();

    // use threadsafe arraybuffers
    std::vector<ArrayBuffer<BiasMatrix<dtype>, 2>> biasMatrixBuf{begin(input.biasMatrices()),
                                                                 end(input.biasMatrices())};
    auto biasMatrixPtr = biasMatrixBuf.data();

    ArrayBuffer<CountsMatrix, 2> biasedConfEnergiesBuf{biasedConfEnergies};
    auto biasedConfEnergiesPtr = &biasedConfEnergiesBuf;

    ArrayBuffer<CountsMatrix, 2> stateCountsBuf{input.stateCounts()};
    auto stateCountsPtr = &stateCountsBuf;

    ArrayBuffer<np_array_nfc<dtype>, 2> modifiedStateCountsLogBuf{modifiedStateCountsLog};
    auto modifiedStateCountsLogPtr = &modifiedStateCountsLogBuf;

    auto inputPtr = &input;
    auto scratch_LL = std::vector<dtype>(input.nMarkovStates());

    #pragma omp parallel for default(none) firstprivate(nThermStates, inputPtr, biasMatrixPtr, stateCountsPtr, biasedConfEnergiesPtr \
                                                        modifiedStateCountsLogPtr, cumNSamples) shared(sampleWeights, scratch_LL)
    for (auto i = 0; i < inputPtr->nMarkovStates(); ++i) {
        std::vector<dtype> scratch(nThermStates);

        for (auto x = 0; x < inputPtr->nSamples(i); ++x) {
            int o = 0;

            for (StateIndex k = 0; k < nThermStates; ++k) {
                // discrete sample log-likelihood \sum_{k=1}^K \sum_{i=1}^m N_i^k * f_i^k
                // compute once for every thermodynamic state and add to total
                if (x == 0) {
                    scratch_LL[i] += ((*stateCountsPtr)(k, i) + tram::detail::prior<dtype>()) * (*biasedConfEnergiesPtr)(k, i);
                }
                if ((*modifiedStateCountsLogPtr)(k, i) > -std::numeric_limits<dtype>::infinity()) {
                    scratch[o++] = (*modifiedStateCountsLogPtr)(k, i) - biasMatrixPtr[i](x, k);
                }
            }
            auto logDivisor = numeric::kahan::logsumexp_sort_kahan_inplace(scratch.begin(), o);
            scratch_LL[i] -= logDivisor;
        }
    }
    return std::accumulate(scratch_LL.begin(), scratch_LL.end(), 0);
}

// TRAM log-likelihood that comes from the transitions, i.e. the MSM part of the TRAM likelihood (or the entire
// dTRAM likelihood)
// \sum_{i,j,k}c_{ij}^{(k)}\log p_{ij}^{(k)}
template<typename dtype>
static const auto
computeTransitionLikelihood(const CountsMatrix &stateCounts, const CountsMatrix &transitionCounts,
                            const np_array_nfc<dtype> &transitionMatrices) {

    auto nThermStates = static_cast<StateIndex>(stateCounts.shape(0));
    auto nMarkovStates = static_cast<StateIndex>(stateCounts.shape(1));

    ArrayBuffer<CountsMatrix, 3> transitionCountsBuf{transitionCounts};
    auto transitionCountsPtr = &transitionCountsBuf;

    ArrayBuffer<np_array_nfc<dtype>, 3> transitionMatricesBuf{transitionMatrices};
    auto transitionMatricesPtr = &transitionMatricesBuf;

    dtype LL = 0;

    #pragma omp parallel for default(none) firstprivate(nThermStates, nMarkovStates, transitionCountsPtr, \
                                                            transitionMatricesPtr) reduction(+:LL) collapse(2)
    for (StateIndex k = 0; k < nThermStates; ++k) {
        for (StateIndex i = 0; i < nMarkovStates; ++i) {

            for (StateIndex j = 0; j < nMarkovStates; ++j) {
                auto CKij = (*transitionCountsPtr)(k, i, j);
                if (CKij > 0) {
                    if (i == j) {
                        LL += (static_cast<dtype>(CKij) + tram::detail::prior<dtype>()) *
                              std::log((*transitionMatricesPtr)(k, i, j));
                    } else {
                        LL += CKij * std::log((*transitionMatricesPtr)(k, i, j));
                    }
                }
            }
        }
    }
    return LL;
}


template<typename inputtype>
struct TRAM {
public:
    using dtype = typename inputtype::value_type;

    TRAM(const np_array_nfc<dtype> &biasedConfEnergies,
         const np_array_nfc<dtype> &lagrangianMultLog,
         const np_array_nfc<dtype> &modifiedStateCountsLog)
            : nThermStates_(biasedConfEnergies.shape(0)),
              nMarkovStates_(biasedConfEnergies.shape(1)),
              biasedConfEnergies_(np_array_nfc<dtype>({nThermStates_, nMarkovStates_})),
              lagrangianMultLog_(ExchangeableArray<dtype, 2>({nThermStates_, nMarkovStates_}, 0.)),
              modifiedStateCountsLog_(np_array_nfc<dtype>({nThermStates_, nMarkovStates_})),
              thermStateEnergies_(ExchangeableArray<dtype, 1>(std::vector<StateIndex>{nThermStates_}, 0.)),
              markovStateEnergies_(np_array_nfc<dtype>(std::vector<StateIndex>{nMarkovStates_})),
              transitionMatrices_(np_array_nfc<dtype>({nThermStates_, nMarkovStates_, nMarkovStates_})),
              statVectors_(ExchangeableArray<dtype, 2>(std::vector({nThermStates_, nMarkovStates_}), 0.)),
              scratch_(std::unique_ptr<dtype[]>(new dtype[std::max(nMarkovStates_, nThermStates_)])) {

        std::copy(lagrangianMultLog.data(), lagrangianMultLog.data() + lagrangianMultLog.size(),
                  lagrangianMultLog_.first()->mutable_data());
        std::copy(biasedConfEnergies.data(), biasedConfEnergies.data() + biasedConfEnergies.size(),
                  biasedConfEnergies_.mutable_data());
        std::copy(modifiedStateCountsLog.data(), modifiedStateCountsLog.data() + modifiedStateCountsLog.size(),
                  modifiedStateCountsLog_.mutable_data());
    }

    const auto &biasedConfEnergies() const {
        return biasedConfEnergies_;
    }

    const auto &lagrangianMultLog() const {
        return *lagrangianMultLog_.first();
    }

    const auto &modifiedStateCountsLog() const {
        return modifiedStateCountsLog_;
    }

    const auto &thermStateEnergies() const {
        return *thermStateEnergies_.first();
    }

    const auto &markovStateEnergies() const {
        return markovStateEnergies_;
    }

    const auto &transitionMatrices() const {
        return transitionMatrices_;
    }


    // estimation loop to estimate the free energies. The current iteration number and iteration error are
    // returned through the callback function if it is provided. log-likelihoods are returned if
    // trackLogLikelihoods = true. By default, this is false, since calculating the log-likelihood required
    // an estimation of the transition matrices, which slows down estimation.
    void estimate(std::shared_ptr<inputtype> &tramInput, std::size_t maxIter, dtype maxErr,
                  std::size_t callbackInterval = 1, bool trackLogLikelihoods = false,
                  const py::object *callback = nullptr) {

        input_ = tramInput;

        double iterationError{0};

        for (decltype(maxIter) iterationCount = 0; iterationCount < maxIter; ++iterationCount) {

            // Magic happens here...
            selfConsistentUpdate();

            // shift all energies by min(energies) so that the minimum energy equals zero.
            shiftEnergiesToHaveZeroMinimum();

            // Tracking of energy vectors for error calculation.
            updateThermStateEnergies();

            // compare new thermStateEnergies_ and statVectors with old to get the
            // iteration error (= how much the energies changed).
            iterationError = computeIterationError();

            dtype logLikelihood{-inf};
            if (trackLogLikelihoods) {

                // log likelihood depends on transition matrices. Compute them first.
                if constexpr(std::is_same_v<TRAMInput<dtype>, inputtype>) {
                    computeTransitionMatrices();
                } else {
                    // TODO
                }
                logLikelihood = computeTransitionLikelihood(input_->stateCounts(), input_->transitionCounts(),
                                                            transitionMatrices_);
                if constexpr(std::is_same_v<TRAMInput<dtype>, inputtype>) {
                    logLikelihood += computeSampleDependentLikelihood(*input_, biasedConfEnergies_, modifiedStateCountsLog_);
                }
            }

            // Send convergence info back to user by calling a python callback function
            if (callback != nullptr && callbackInterval > 0 && iterationCount % callbackInterval == 0) {
                py::gil_scoped_acquire guard;
                (*callback)(callbackInterval, iterationError, logLikelihood);
            }

            if (iterationError < maxErr) {
                // We have converged!
                break;
            }
        }
        // Done iterating. Compute all energies for the thermodynamic states and markov states.

        if constexpr(std::is_same_v<TRAMInput<dtype>, inputtype>) {
            updateMarkovStateEnergies();
        } else {
            // TODO
        }
        updateThermStateEnergies();
        normalize();

        // And compute final transition matrices
        if constexpr(std::is_same_v<TRAMInput<dtype>, inputtype>) {
            computeTransitionMatrices();
        } else {
            // TODO
        }
    }

private:
    std::shared_ptr<inputtype> input_;

    StateIndex nThermStates_;
    StateIndex nMarkovStates_;

    np_array_nfc<dtype> biasedConfEnergies_;
    ExchangeableArray<dtype, 2> lagrangianMultLog_;
    np_array_nfc<dtype> modifiedStateCountsLog_;

    ExchangeableArray<dtype, 1> thermStateEnergies_;
    np_array_nfc<dtype> markovStateEnergies_;
    np_array_nfc<dtype> transitionMatrices_;

    // Array to keep track of statVectors_(K, i) = exp(thermStateEnergies_(K) - biasedConfEnergies_(K, i))
    // difference between previous and current statvectors is used for calculating the iteration error to check whether
    // convergence is achieved.
    ExchangeableArray<dtype, 2> statVectors_;

    // scratch matrices used to facilitate calculation of logsumexp
    std::unique_ptr<dtype[]> scratch_;

    constexpr static dtype inf = std::numeric_limits<dtype>::infinity();

    void selfConsistentUpdate() {
        // Self-consistent update of the TRAM equations.
        updateLagrangianMult();
        updateStateCounts();
        updateBiasedConfEnergies();
    }

    void updateLagrangianMult() {
        lagrangianMultLog_.exchange();
        auto oldLagrangianMultLogBuf = lagrangianMultLog_.secondBuf();
        auto newLagrangianMultLogBuf = lagrangianMultLog_.firstBuf();

        auto biasedConfEnergiesBuf = biasedConfEnergies_.template unchecked<2>();

        auto transitionCountsBuf = input_->transitionCountsBuf();
        auto stateCountsBuf = input_->stateCountsBuf();

        auto nThermStates = nThermStates_, nMarkovStates = nMarkovStates_;

        #pragma omp parallel for default(none) firstprivate(nThermStates, nMarkovStates, oldLagrangianMultLogBuf, \
                                                            newLagrangianMultLogBuf, biasedConfEnergiesBuf, \
                                                            transitionCountsBuf, stateCountsBuf)
        for (StateIndex k = 0; k < nThermStates; ++k) {
            std::vector<dtype> scratch(nMarkovStates);

            for (StateIndex i = 0; i < nMarkovStates; ++i) {
                if (0 == stateCountsBuf(k, i)) {
                    newLagrangianMultLogBuf(k, i) = -inf;
                    continue;
                }
                std::size_t o = 0;

                for (StateIndex j = 0; j < nMarkovStates; ++j) {
                    auto CKij = transitionCountsBuf(k, i, j);
                    // special case: most variables cancel out, here
                    if (i == j) {
                        scratch[o++] = (0 == CKij) ?
                                       tram::detail::logPrior<dtype>() : std::log(
                                        tram::detail::prior<dtype>() + (dtype) CKij);
                    } else {
                        auto CK = CKij + transitionCountsBuf(k, j, i);
                        if (0 != CK) {
                            auto divisor = numeric::kahan::logsumexp_pair<dtype>(
                                    oldLagrangianMultLogBuf(k, j) - biasedConfEnergiesBuf(k, i)
                                    - oldLagrangianMultLogBuf(k, i) + biasedConfEnergiesBuf(k, j), 0.0);
                            scratch[o++] = std::log((dtype) CK) - divisor;
                        }
                    }
                }
                newLagrangianMultLogBuf(k, i) = numeric::kahan::logsumexp_sort_kahan_inplace(scratch.begin(), o);
            }
        }
    }

    // update conformation energies based on one observed trajectory
    // TODO version for TRAM and for dTRAM!
    void updateBiasedConfEnergies() {
        auto biasedConfEnergiesBuf = biasedConfEnergies_.template mutable_unchecked<2>();
        auto modifiedStateCountsLogBuf = modifiedStateCountsLog_.template unchecked<2>();

        auto nThermStates = nThermStates_, nMarkovStates = nMarkovStates_;
        auto input = input_;

        if constexpr(std::is_same_v<TRAMInput<dtype>, inputtype>) {
            std::fill(biasedConfEnergies_.mutable_data(), biasedConfEnergies_.mutable_data() +
                                                          biasedConfEnergies_.size(), inf);

            std::vector<ArrayBuffer<BiasMatrix<dtype>, 2>> biasMatrixBuffers(input_->biasMatrices().begin(),
                                                                             input_->biasMatrices().end());
            auto biasMatrixPtr = biasMatrixBuffers.data();

            #pragma omp parallel for default(none) firstprivate(nMarkovStates, nThermStates, input, biasMatrixPtr, \
                                                            biasedConfEnergiesBuf, modifiedStateCountsLogBuf)
            for (StateIndex i = 0; i < nMarkovStates; ++i) {
                std::vector<dtype> scratch(nThermStates);

                for (std::int32_t x = 0; x < input->nSamples(i); ++x) {
                    std::size_t o = 0;
                    for (StateIndex k = 0; k < nThermStates; ++k) {
                        if (modifiedStateCountsLogBuf(k, i) > -inf) {
                            scratch[o++] = modifiedStateCountsLogBuf(k, i) - biasMatrixPtr[i](x, k);
                        }
                    }
                    dtype divisor = numeric::kahan::logsumexp_sort_kahan_inplace(scratch.begin(), o);

                    for (StateIndex k = 0; k < nThermStates; ++k) {
                        biasedConfEnergiesBuf(k, i) = -numeric::kahan::logsumexp_pair(
                                -biasedConfEnergiesBuf(k, i), -(divisor + biasMatrixPtr[i](x, k)));
                    }
                }
            }
        }
        if constexpr(std::is_same_v<dTRAMInput<dtype>, inputtype>) {
            auto lagrangianMultLogBuf = lagrangianMultLog_.firstBuf();

            #pragma omp parallel for default(none) firstprivate(nMarkovStates, nThermStates, input, \
                                                            biasedConfEnergiesBuf, modifiedStateCountsLogBuf)

            for (StateIndex i = 0; i < nMarkovStates; ++i) {
                std::vector<dtype> scratch(nThermStates * nMarkovStates);
                std::size_t o = 0;
                int Ci = 0;

                for (StateIndex k = 0; k < nThermStates; ++k) {
                    for (StateIndex j = 0; j < nMarkovStates; ++j) {
                        auto CKij = input->transitionCounts(k, i, j);
                        auto CKji = input->transitionCounts(k, j, i);
                        /* add counts to Ci */
                        Ci += CKji;
                        /* special case: most variables cancel out, here */
                        // TODO: can we take out the biasCoefficients here by assuming they are added to the biasedConfEnergies?
                        if (i == j) {
                            scratch[o] = (0 == CKij) ?
                                         detail::logPrior<dtype>() : log(detail::prior<dtype>() + (double) CKij);
                            scratch[o++] += biasedConfEnergiesBuf - input->biasCoefficients(k, i);
                            continue;
                        }
                        auto CK = CKij + CKji;
                        /* special case */
                        if (CK > 0) {
                            auto divisor = _logsumexp_pair(
                                    lagrangianMultLogBuf(k, j) - biasedConfEnergiesBuf(k, i),
                                    lagrangianMultLogBuf(k, i) - biasedConfEnergiesBuf(k, j));
                            scratch[o++] =
                                    log((double) CK) - input->biasCoefficients(k, i) + lagrangianMultLogBuf(k, j) -
                                    divisor;
                        }
                    }
                }
                /* patch Ci and the total divisor together */
                auto newConfEnergy = _logsumexp_sort_kahan_inplace(scratch, o) - log(
                        nThermStates * detail::logPrior<dtype>() + (double) Ci);
                for (StateIndex l = 0; l < nThermStates; ++l) {
                    biasedConfEnergiesBuf(l, i) = newConfEnergy + input->biasCoefficients(l, i);
                }
            }
        }
    }


    void updateStateCounts() {
        auto biasedConfEnergiesBuf = biasedConfEnergies_.template unchecked<2>();
        auto lagrangianMultLogBuf = lagrangianMultLog_.firstBuf();
        auto modifiedStateCountsLogBuf = modifiedStateCountsLog_.template mutable_unchecked<2>();

        auto stateCountsBuf = input_->stateCountsBuf();
        auto transitionCountsBuf = input_->transitionCountsBuf();

        #pragma omp parallel for default(none) firstprivate(biasedConfEnergiesBuf, lagrangianMultLogBuf, \
                                                            modifiedStateCountsLogBuf, stateCountsBuf, \
                                                            transitionCountsBuf) collapse(2)
        for (StateIndex k = 0; k < nThermStates_; ++k) {
            for (StateIndex i = 0; i < nMarkovStates_; ++i) {

                std::vector<dtype> scratch;
                scratch.reserve(nMarkovStates_);

                if (0 == stateCountsBuf(k, i)) {
                    modifiedStateCountsLogBuf(k, i) = -inf;
                } else {
                    auto Ci = 0;
                    for (StateIndex j = 0; j < nMarkovStates_; ++j) {
                        auto CKij = transitionCountsBuf(k, i, j);
                        auto CKji = transitionCountsBuf(k, j, i);
                        Ci += CKji;
                        // special case: most variables cancel out, here
                        if (i == j) {
                            auto CKijLog = (0 == CKij) ? tram::detail::logPrior<dtype>() : std::log(
                                    tram::detail::prior<dtype>() + (dtype) CKij);
                            scratch.push_back(CKijLog + biasedConfEnergiesBuf(k, i));
                        } else {
                            auto CK = CKij + CKji;

                            if (CK > 0) {
                                auto divisor = numeric::kahan::logsumexp_pair(
                                        lagrangianMultLogBuf(k, j) - biasedConfEnergiesBuf(k, i),
                                        lagrangianMultLogBuf(k, i) - biasedConfEnergiesBuf(k, j));
                                scratch.push_back(std::log((dtype) CK) + lagrangianMultLogBuf(k, j) - divisor);
                            }
                        }
                    }
                    auto NC = stateCountsBuf(k, i) - Ci;
                    auto extraStateCounts = (0 < NC) ? std::log((dtype) NC) + biasedConfEnergiesBuf(k, i) : -inf;
                    modifiedStateCountsLogBuf(k, i) = numeric::kahan::logsumexp_pair(
                            numeric::kahan::logsumexp_sort_kahan_inplace(scratch.begin(), scratch.end()),
                            extraStateCounts);
                }
            }
        }
    }

// Get the error in the energies between this iteration and the previous one.
    dtype computeIterationError() {
        updateStatVectors();

        dtype error1 = computeError(thermStateEnergies_, nThermStates_);
        dtype error2 = computeError(statVectors_, nThermStates_ * nMarkovStates_);

        return std::max(error1, error2);
    }

    void updateStatVectors() {
        // move current values to old
        statVectors_.exchange();

        // compute new values
        auto statVectorsBuf = statVectors_.firstBuf();
        auto thermStateEnergiesBuf = thermStateEnergies_.firstBuf();
        auto biasedConfEnergiesBuf = biasedConfEnergies_.template unchecked<2>();

        #pragma omp parallel for default(none) firstprivate(statVectorsBuf, \
                                                            thermStateEnergiesBuf, biasedConfEnergiesBuf)
        for (StateIndex k = 0; k < nThermStates_; ++k) {
            for (StateIndex i = 0; i < nMarkovStates_; ++i) {
                statVectorsBuf(k, i) = std::exp(thermStateEnergiesBuf(k) - biasedConfEnergiesBuf(k, i));
            }
        }
    }

// TODO version for TRAM and for dTRAM!
    void updateMarkovStateEnergies() {
        auto markovStateEnergiesBuf = markovStateEnergies_.template mutable_unchecked<1>();
        auto nThermStates = nThermStates_, nMarkovStates = nMarkovStates_;
        auto input = input_;

        if constexpr(std::is_same_v<TRAMInput<dtype>, inputtype>) {
            // first reset all confirmation energies to infinity
            std::fill(markovStateEnergies_.mutable_data(), markovStateEnergies_.mutable_data() + nMarkovStates_, inf);

            std::vector<ArrayBuffer<BiasMatrix<dtype>, 2>> biasMatrixBuffers(input_->biasMatrices().begin(),
                                                                             input_->biasMatrices().end());
            auto biasMatrixPtr = biasMatrixBuffers.data();

            auto modifiedStateCountsLogBuf = modifiedStateCountsLog_.template unchecked<2>();

            // assume that markovStateEnergies_ were set to INF by the caller on the first call
            #pragma omp parallel for default(none) firstprivate(nMarkovStates, nThermStates, input, biasMatrixPtr, \
                                                            markovStateEnergiesBuf, modifiedStateCountsLogBuf)
            for (StateIndex i = 0; i < nMarkovStates; ++i) {
                std::vector<dtype> scratch(nThermStates);

                for (auto x = 0; x < input->nSamples(i); ++x) {
                    std::size_t o = 0;
                    for (StateIndex k = 0; k < nThermStates; ++k) {
                        if (modifiedStateCountsLogBuf(k, i) > -inf) {
                            scratch[o++] = modifiedStateCountsLogBuf(k, i) - biasMatrixPtr[i](x, k);
                        }
                    }
                    dtype divisor = numeric::kahan::logsumexp_sort_kahan_inplace(scratch.begin(), o);
                    markovStateEnergiesBuf(i) = -numeric::kahan::logsumexp_pair(-markovStateEnergiesBuf(i), -divisor);
                }
            }
        }
        // Markov state energies are identical for each therm states. We can simply unbias them by subtracting the bias
        // coefficients.
        // This is an artefact of computing dTRAM in the space of the biasedConfEnergies.
        // It could be computed in the space of markovStateEnergies directly, but by computing in biasedConfEnergy
        // space reduces duplicate code and makes it easier to combine TRAM with dTRAM iterations.
        if constexpr(std::is_same_v<dTRAMInput<dtype>, inputtype>) {
            auto biasedConfEnergiesBuf = biasedConfEnergies_.template unchecked<2>();
            for (StateIndex i = 0; i < nMarkovStates; ++i) {
                markovStateEnergiesBuf(i) = biasedConfEnergiesBuf(0, i) - input->biasCoefficients(0, i);
            }
        }
    }

    void updateThermStateEnergies() {
        // move current values to old
        thermStateEnergies_.exchange();

        // compute new
        auto biasedConfEnergiesBuf = biasedConfEnergies_.template unchecked<2>();
        auto thermStateEnergiesBuf = thermStateEnergies_.firstBuf();
        auto scratch = scratch_.get();

        for (StateIndex k = 0; k < nThermStates_; ++k) {
            for (StateIndex i = 0; i < nMarkovStates_; ++i) {
                scratch[i] = -biasedConfEnergiesBuf(k, i);
            }
            thermStateEnergiesBuf(k) = -numeric::kahan::logsumexp_sort_kahan_inplace(scratch, nMarkovStates_);
        }
    }

// Shift all energies by min(biasedConfEnergies_) so the energies don't drift to
// very large values.
    void shiftEnergiesToHaveZeroMinimum() {
        auto biasedConfEnergiesBuf = biasedConfEnergies_.template mutable_unchecked<2>();
        auto thermStateEnergiesBuf = thermStateEnergies_.firstBuf();

        auto ptr = biasedConfEnergies_.data();
        auto shift = *std::min_element(ptr, ptr + biasedConfEnergies_.size());

        #pragma omp parallel for default(none) firstprivate(nThermStates_, biasedConfEnergiesBuf, thermStateEnergiesBuf, shift)
        for (StateIndex k = 0; k < nThermStates_; ++k) {
            thermStateEnergiesBuf(k) -= shift;

            for (StateIndex i = 0; i < nMarkovStates_; ++i) {
                biasedConfEnergiesBuf(k, i) -= shift;
            }
        }
    }

    void normalize() {
        auto biasedConfEnergiesBuf = biasedConfEnergies_.template mutable_unchecked<2>();
        auto markovStateEnergiesBuf = markovStateEnergies_.template mutable_unchecked<1>();
        auto thermStateEnergiesBuf = thermStateEnergies_.firstBuf();

        for (StateIndex i = 0; i < nMarkovStates_; ++i) {
            scratch_[i] = -markovStateEnergiesBuf(i);
        }
        auto f0 = -numeric::kahan::logsumexp_sort_kahan_inplace(scratch_.get(), nMarkovStates_);

        for (StateIndex i = 0; i < nMarkovStates_; ++i) {
            markovStateEnergiesBuf(i) -= f0;
            for (StateIndex k = 0; k < nThermStates_; ++k) {
                biasedConfEnergiesBuf(k, i) -= f0;
            }
        }
        for (StateIndex k = 0; k < nThermStates_; ++k) {
            thermStateEnergiesBuf(k) -= f0;
        }

        // update the state counts because they also include biased conf energies.
        // If this is not done after normalizing, the log likelihood computations will
        // not produce the correct output, due to incorrect values for mu(x).
        updateStateCounts();
    }

    void computeTransitionMatrices() {
        auto biasedConfEnergiesBuf = biasedConfEnergies_.template unchecked<2>();
        auto lagrangianMultLogBuf = lagrangianMultLog_.firstBuf();

        auto transitionCountsBuf = input_->transitionCountsBuf();
        auto transitionMatricesBuf = transitionMatrices_.template mutable_unchecked<3>();

        auto nThermStates = nThermStates_;
        auto nMarkovStates = nMarkovStates_;

        if constexpr(std::is_same_v<TRAMInput<dtype>, inputtype>) {

            #pragma omp parallel for default(none) firstprivate(nThermStates, nMarkovStates, biasedConfEnergiesBuf, \
                                                            lagrangianMultLogBuf, transitionCountsBuf, \
                                                            transitionMatricesBuf)
            for (StateIndex k = 0; k < nThermStates; ++k) {
                std::vector<dtype> scratch(nMarkovStates, 0);
                for (StateIndex i = 0; i < nMarkovStates; ++i) {
                    for (StateIndex j = 0; j < nMarkovStates; ++j) {
                        transitionMatricesBuf(k, i, j) = 0.0;
                        auto C = transitionCountsBuf(k, i, j) + transitionCountsBuf(k, j, i);

                        if (C > 0) { // skip if there were no transitions
                            if (i == j) {
                                // special case: diagonal element
                                transitionMatricesBuf(k, i, j) = 0.5 * C * exp(-lagrangianMultLogBuf(k, i));
                            } else {
                                // regular case
                                auto divisor = numeric::kahan::logsumexp_pair(
                                        lagrangianMultLogBuf(k, j) - biasedConfEnergiesBuf(k, i),
                                        lagrangianMultLogBuf(k, i) - biasedConfEnergiesBuf(k, j));
                                transitionMatricesBuf(k, i, j) = C * exp(-(biasedConfEnergiesBuf(k, j) + divisor));
                            }
                            scratch[i] += transitionMatricesBuf(k, i, j);
                        }
                    }
                }
                // normalize transition matrix
                auto maxSumIt = std::max_element(std::begin(scratch), std::end(scratch));

                dtype maxSum;
                if (maxSumIt == std::end(scratch) || *maxSumIt == 0) {
                    maxSum = 1.0; // completely empty T matrix -> generate Id matrix
                } else {
                    maxSum = *maxSumIt;
                }

                for (StateIndex i = 0; i < nMarkovStates; ++i) {
                    for (StateIndex j = 0; j < nMarkovStates; ++j) {
                        if (i == j) {
                            transitionMatricesBuf(k, i, i) =
                                    (transitionMatricesBuf(k, i, i) + maxSum - scratch[i]) / maxSum;
                            if (0 == transitionMatricesBuf(k, i, i) && 0 < transitionCountsBuf(k, i, i)) {
                                std::stringstream ss;
                                ss << "# Warning: zero diagonal element T[" << i << "," << i
                                   << "] with non-zero counts.";
                                #pragma omp critical
                                {
                                    py::gil_scoped_acquire gil;
                                    py::print(ss.str());
                                }
                            }
                        } else {
                            transitionMatricesBuf(k, i, j) = transitionMatricesBuf(k, i, j) / maxSum;
                        }
                    }
                }
            }
        }
        if constexpr(std::is_same_v<dTRAMInput<dtype>, inputtype>) {
            for (StateIndex k = 0; k < nThermStates; ++k) {
                std::vector<dtype> scratch(nMarkovStates, 0);

                for (StateIndex i = 0; i < nMarkovStates; ++i) {
                    std::size_t o = 0;

                    for (StateIndex j = 0; j < nMarkovStates; ++j) {
                        transitionMatricesBuf(k, i, j) = 0.0;
                        /* special case: diagonal element */
                        if (i == j) {
                            scratch[o] = (0 == input_->transitionCounts(k, i, j)) ?
                                         detail::logPrior<dtype>() : log(
                                            detail::prior<dtype>() + (double) input_->transitionCounts(k, i, j));
                            scratch[o] -= lagrangianMultLogBuf(k, i);
                            transitionMatricesBuf(k, i, j) = exp(scratch[o++]);
                            continue;
                        }
                        auto C = input_->transitionCounts(k, i, j) + input_->transitionCounts(k, j, i);
                        /* special case: this element is zero */
                        if (C > 0) {
                            auto divisor = _logsumexp_pair(
                                    lagrangianMultLogBuf(k, j) - biasedConfEnergiesBuf(k, i),
                                    lagrangianMultLogBuf(k, i) - biasedConfEnergiesBuf(k, j));
                            scratch[o] = log((double) C) - biasedConfEnergiesBuf(k, j) - divisor;
                            transitionMatricesBuf(k, i, j) = exp(scratch[o++]);
                        }
                    }
                    /* compute the diagonal elements from the other elements in this line */
                    auto sum = exp(_logsumexp_sort_kahan_inplace(scratch, o));
                    if (0.0 == sum) {
                        for (StateIndex j = 0; j < nMarkovStates; ++j)
                            transitionMatricesBuf(k, i, j) = 0.0;
                        transitionMatricesBuf(k, i, i) = 1.0;
                    } else if (1.0 != sum) {
                        for (StateIndex j = 0; j < nMarkovStates; ++j)
                            transitionMatricesBuf(k, i, j) /= sum;
                    }
                }
            }
        }
    }
};

// compute the log-likelihood of observing the input data under the current
// biasedConfEnergies_. This computes the parameter-dependent part of the likelihood, i.e. the factor -b^k(x) is
// omitted as it is constant.
// This loglikelihood computation gets called py pybind! It expects only one (concatenated) trajectory.
// TODO move this to a separate file? It is not used during (d)TRAM estimation
template<typename dtype>
static const dtype computeLogLikelihood(const DTraj &dtraj,
                                        const BiasMatrix<dtype> &biasMatrix,
                                        const np_array_nfc<dtype> &biasedConfEnergies,
                                        const np_array_nfc<dtype> &modifiedStateCountsLog,
                                        const np_array<dtype> &thermStateEnergies,
                                        const CountsMatrix &stateCounts,
                                        const CountsMatrix &transitionCounts,
                                        const np_array_nfc<dtype> &transitionMatrices) {

    // first get likelihood of all discrete quantities (transition likelihood and free energies times state counts)
    auto logLikelihood = computeTransitionLikelihood(stateCounts, transitionCounts, transitionMatrices);

    // then compute log of all sample weights, and add to log likelihood.
    auto sampleWeights = computeSampleWeightsLog(dtraj, biasMatrix, thermStateEnergies, modifiedStateCountsLog, -1);
    logLikelihood += numeric::kahan::logsumexp_sort_kahan_inplace(sampleWeights.begin(), sampleWeights.end());

    return logLikelihood;
}


template<typename dtype>
np_array_nfc<dtype> initLagrangianMult(const CountsMatrix &transitionCounts) {
    auto nThermStates = static_cast<StateIndex>(transitionCounts.shape(0));
    auto nMarkovStates = static_cast<StateIndex>(transitionCounts.shape(1));

    np_array_nfc<dtype> lagrangianMultLog({nThermStates, nMarkovStates});
    auto lagrangianMultLogBuf = lagrangianMultLog.template mutable_unchecked<2>();

    ArrayBuffer<CountsMatrix, 3> transitionCountsBuf{transitionCounts};

    for (StateIndex k = 0; k < nThermStates; ++k) {
        for (StateIndex i = 0; i < nMarkovStates; ++i) {
            dtype sum = 0.0;
            for (StateIndex j = 0; j < nMarkovStates; ++j) {
                sum += (transitionCountsBuf(k, i, j) + transitionCountsBuf(k, j, i));
            }
            lagrangianMultLogBuf(k, i) = std::log(sum / 2);
        }
    }
    return lagrangianMultLog;
}

}
