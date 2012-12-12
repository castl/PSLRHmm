#include "hmm.hpp"

#include <cmath>
#include <limits>
#include <boost/random/uniform_01.hpp>

using namespace std;

namespace pslrhmm {
	void HMM::initRandom(Random& r, size_t states, std::vector<Emission*> alphabet) {
		boost::random::uniform_01<> u;

		this->states.clear();
		this->init_prob.clear();

		for (size_t i=0; i<states; i++) {
			State* s = new State(*this);
			this->states.push_back(s);
			this->init_prob[s] = u(r);
		}

		this->init_prob.normalize();


		BOOST_FOREACH(auto s, this->states) {
			BOOST_FOREACH(auto t, this->states) {
				s->transition_probs[t] = u(r);
			}
			s->transition_probs.normalize();

			BOOST_FOREACH(auto e, alphabet) {
				s->emissions_probs[e] = u(r);
			}
			s->emissions_probs.normalize();
		}
	}

	void HMM::generateSequence(Random& r, Sequence& seq, size_t len) const {
		const State* s = this->generateInitialState(r);
		for (size_t i=0; i<len; i++) {
			seq.push_back(s->generateEmission(r));
			s = s->generateState(r);
		}
	}

	const State* HMM::generateInitialState(Random& r) const {
		return init_prob.select(r);
	}


	struct LogDouble {
		class NegativeNumberException : public exception {
		public:
			double d;
			NegativeNumberException(double d) : d(d) { }
		};

		double l;

		LogDouble() : l(std::numeric_limits<double>::quiet_NaN()) { }
		LogDouble(double d) {
			(*this) = d;
		}

		void operator=(double d) {
			if (d == 0)
				l = numeric_limits<double>::quiet_NaN();
			else if (d > 0.0)
				l = std::log(d);
			else 
				throw NegativeNumberException(d);
		}

		LogDouble operator+(double d) const {
			return (*this) + LogDouble(d);
		}

		LogDouble operator+(LogDouble y) const {
			LogDouble x = *this;
			LogDouble r;
			if (isnan(x.l))
				return y;
			else if (isnan(y.l))
				return x;
			else if (x.l > y.l) {
				r.l = x.l + std::log(1 + std::exp(y.l - x.l));
			} else {
				r.l = y.l + std::log(1 + std::exp(x.l - y.l));
			}

			return r;
		}

		LogDouble operator*(double d) const {
			LogDouble r;
			r.l = this->l + std::log(d);
			return r;
		}

		LogDouble operator*(LogDouble d) const {
			LogDouble r;
			r.l = this->l + d.l;
			return r;
		}

		double exp() const {
			if (isnan(this->l))
				return 0.0;
			return std::exp(this->l);
		}

		double log() const {
			return this->l;
		}
	};

	double HMM::calcSequenceLikelihoodLog(const Sequence& seq) const {
		const size_t num_states = states.size();
		vector<LogDouble> alpha_old(num_states);
		vector<LogDouble> alpha    (num_states);

		// Initialization
		const Emission* e = seq[0];
		for (size_t i=0; i<num_states; i++) {
			const State* s = states[i];
			alpha[i] = Pi(s)*B(s, e);
		}

		for (size_t oi=1; oi<seq.size(); oi++) {
			e = seq[oi];
			alpha_old.swap(alpha);

			for (size_t i=0; i<num_states; i++) {
				const State* s = states[i];

				LogDouble t = 0.0;
				for (size_t j=0; j<num_states; j++) {
					t = t + alpha_old[j] * A(j, s);
				}
				alpha[i] = t * B(s, e);
			}
		}

		LogDouble sum = 0.0;
		for (size_t i=0; i<num_states; i++) {
			sum = sum + alpha[i];
		}
		return sum.l;	
	}
}