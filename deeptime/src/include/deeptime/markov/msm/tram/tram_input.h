//
// Created by Maaike on 14/12/2021.
//

#pragma once

#include "common.h"
#include "dtram_input.h"


namespace deeptime::markov::tram {


template<typename dtype>
class TRAMInput : public dTRAMInput<dtype> {
public:
    using size_type = typename BiasMatrices<dtype>::size_type;

    TRAMInput(CountsMatrix &&stateCounts, CountsMatrix &&transitionCounts, BiasMatrices<dtype> biasMatrices,
              BiasMatrix<dtype> &&biasCoefficients)
            : dTRAMInput<dtype>(std::move(stateCounts), std::move(transitionCounts), std::move(biasCoefficients)) {
        biasMatrices_ = std::move(biasMatrices);
        cumNSamples();
        validateInput();
    }

    TRAMInput() = default;

    TRAMInput(const TRAMInput &) = delete;

    TRAMInput &operator=(const TRAMInput &) = delete;

    TRAMInput(TRAMInput &&) noexcept = default;

    TRAMInput &operator=(TRAMInput &&) noexcept = default;

    ~TRAMInput() = default;

    void validateInput() const {
        detail::throwIfInvalid(!biasMatrices_.empty(), "We need bias matrices.");
        std::for_each(begin(biasMatrices_), end(biasMatrices_),
                      [nThermStates = this->stateCounts().shape(0)](const auto &biasMatrix) {
                          detail::throwIfInvalid(biasMatrix.ndim() == 2,
                                                 "biasMatrix has an incorrect number of dimension. ndims should be 2.");
                          detail::throwIfInvalid(biasMatrix.shape(1) == nThermStates,
                                                 "biasMatrix.shape[1] should be equal to transitionCounts.shape[0].");
                      });
    }

    const auto &biasMatrix(size_type i) const {
        return biasMatrices_[i];
    }

    const auto &biasMatrices() const {
        return biasMatrices_;
    }

    auto nSamples(size_type i) const {
        return biasMatrices_[i].shape(0);
    }

    auto nSamples() const {
        decltype(nSamples(0)) total{};
        for (size_type i = 0; i < biasMatrices_.size(); ++i) {
            total += nSamples(i);
        }
        return total;
    }

    const auto &cumNSamples() const {
        return cumNSamples_;
    }

private:
    BiasMatrices<dtype> biasMatrices_;
    std::vector<size_type> cumNSamples_;
};

}
