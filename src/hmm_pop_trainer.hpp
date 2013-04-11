#ifndef __HMM_POP_TRAINER_HPP__
#define __HMM_POP_TRAINER_HPP__

#include "hmm.hpp"

namespace pslrhmm {

	template<typename HMM>
	class PopulationTrainer {
		struct HmmScorePair {
			double score;
			HMM* hmm;

			bool operator<(const HmmScorePair& other) const {
				return this->score < other.score;
			}

			HmmScorePair() {
				score = 0.0;
				hmm = new HMM();
			}
		};

		typedef typename HMM::Emission Emission;
		std::vector< HmmScorePair > population;
        HmmScorePair bestpair;

	public:
		PopulationTrainer(size_t population_size) {
			for (size_t i=0; i<population_size; i++) {
				population.push_back(HmmScorePair());
			}
            bestpair = population.front();
            bestpair.hmm = bestpair.hmm->clone();
		}

		HMM& best() {
			return *bestpair.hmm;
		}

		double bestScore() {
			return bestpair.score;
		}

		void initRandom(typename HMM::Random& r, size_t states,
				std::vector<Emission> alphabet = std::vector<Emission>()) {
			BOOST_FOREACH(auto& hs, population) {
				hs.hmm->initRandom(r, states, alphabet);
			}
		}
		void initUniform(size_t states, std::vector<Emission> example) {
			BOOST_FOREACH(auto& hs, population) {
				hs.hmm->initUniform(states, example);
			}
		}
		void initUniform(size_t states, Emission example) {
			BOOST_FOREACH(auto& hs, population) {
				hs.hmm->initUniform(states, example);
			}		
		}

		void baum_welch(const std::vector< typename HMM::Sequence >& sequences) {
			size_t pop_size = population.size();

			for (size_t i=0; i<pop_size; i++) {
				auto& hs = population[i];
				hs.hmm->baum_welch(sequences);
				hs.score = hs.hmm->calcSequenceLikelihoodNorm(sequences);
			}

			std::stable_sort(population.begin(), population.end());

            if (population.back().score > bestpair.score) {
                delete bestpair.hmm;
                bestpair = population.back();
                bestpair.hmm = bestpair.hmm->clone();
            }
		}
	};
}

#endif
