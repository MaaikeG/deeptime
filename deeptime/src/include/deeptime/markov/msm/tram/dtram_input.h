//
// Created by Maaike on 14/12/2021.
//

#pragma once

#include "common.h"

namespace deeptime::markov::tram {

namespace detail {
constexpr void throwIfInvalid(bool isValid, const std::string &message) {
    if (!isValid) {
        throw std::runtime_error(message);
    }
}
}


template<typename dtype>
class dTRAMInput {
public:
    using size_type = typename BiasMatrices<dtype>::size_type;
    using value_type = dtype;

    dTRAMInput(CountsMatrix &&stateCounts, CountsMatrix &&transitionCounts, std::optional<BiasMatrix<dtype>> &&biasCoefficients)
            : stateCounts_(std::move(stateCounts)),
              transitionCounts_(std::move(transitionCounts)),
              biasCoefficients_(std::move(biasCoefficients)) {
        validateInput();
    }

    dTRAMInput() = default;

    dTRAMInput(const dTRAMInput &) = delete;

    dTRAMInput &operator=(const dTRAMInput &) = delete;

    dTRAMInput(dTRAMInput &&) noexcept = default;

    dTRAMInput &operator=(dTRAMInput &&) noexcept = default;

    ~dTRAMInput() = default;

    void validateInput() const {
        detail::throwIfInvalid(stateCounts_.shape(0) == transitionCounts_.shape(0),
                               "stateCounts.shape(0) should equal transitionCounts.shape(0)");
        detail::throwIfInvalid(stateCounts_.shape(1) == transitionCounts_.shape(1),
                               "stateCounts.shape(1) should equal transitionCounts.shape(1)");
        detail::throwIfInvalid(transitionCounts_.shape(1) == transitionCounts_.shape(2),
                               "transitionCounts.shape(1) should equal transitionCounts.shape(2)");
        if (biasCoefficients_) {
            detail::throwIfInvalid(stateCounts_.shape() == biasCoefficients_->shape(),
                                   "stateCounts.shape() should equal biasCoefficients.shape()");
        }
    }

    const auto& transitionCounts() const {
        return transitionCounts_;
    }

    auto transitionCountsBuf() const {
        return transitionCounts_.template unchecked<3>();
    }

    const auto& stateCounts() const {
        return stateCounts_;
    }

    auto stateCountsBuf() const {
        return stateCounts_.template unchecked<2>();
    }

    auto nThermStates() const {
        return transitionCounts_.shape(0);
    }

    auto nMarkovStates() const {
        return stateCounts_.shape(1);
    }


private:
    CountsMatrix stateCounts_;
    CountsMatrix transitionCounts_;
    std::optional<BiasMatrix<dtype>> biasCoefficients_;
};

}
