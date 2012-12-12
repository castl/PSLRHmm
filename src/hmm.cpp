#include "hmm.hpp"

#include <cmath>
#include <limits>
#include <boost/random/uniform_01.hpp>

using namespace std;

namespace pslrhmm {
	void HMM::initRandom(Random& r, size_t states, std::vector<Emission*> alphabet) {
		boost::uniform_01<> u;

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

	void HMM::initUniform(size_t states, std::vector<Emission*> alphabet) {
		double sinv = 1.0l / states;
		double einv = 1.0l / alphabet.size();
		this->states.clear();
		this->init_prob.clear();

		for (size_t i=0; i<states; i++) {
			State* s = new State(*this);
			this->states.push_back(s);
			this->init_prob[s] = sinv;
		}

		this->init_prob.normalize();


		BOOST_FOREACH(auto s, this->states) {
			BOOST_FOREACH(auto t, this->states) {
				s->transition_probs[t] = sinv;
			}
			s->transition_probs.normalize();

			BOOST_FOREACH(auto e, alphabet) {
				s->emissions_probs[e] = einv;
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
			if (std::isnan(x.l))
				return y;
			else if (std::isnan(y.l))
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
			if (std::isnan(this->l))
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


	void HMM::forward_scaled(const Sequence& seq,
				Matrix<double>& alpha, std::vector<double>& c) const {
		const size_t T = seq.size();
		const size_t num_states = states.size();
		alpha.resize(T, num_states);
		c.resize(T);

		for (size_t i=0; i<num_states; i++) {
			alpha(0, i) = Pi(i) * B(i, seq[0]);
		}
		c[0] = 1.0l / alpha.xSlice(0).sum();

		for (size_t t=1; t<T; t++) {
			for (size_t i=0; i<num_states; i++) {
				const State* si = states[i];
				double sum = 0.0;
				for (size_t j=0; j<num_states; j++) {
					sum += alpha(t - 1, j) * A(j, si);
				}
				alpha(t, i) = sum * B(i, seq[t]);
			}

			auto slice = alpha.xSlice(t);
			double ct = 1.0l / slice.sum();
			c[t] = ct;
			slice *= ct;
		}
	}

	void HMM::backward_scaled(const Sequence& seq,
				const Matrix<double>& alpha, const std::vector<double>& c,
				Matrix<double>& beta) const {
		const size_t T = seq.size();
		const size_t num_states = states.size();
		assert(c.size() == T);	

		beta.resize(T, num_states);
		for (size_t i=0; i<num_states; i++) {
			size_t t = T-1;
			beta(t, i) = 1 * c[t];
		}

		for (int64_t t = T - 2; t >= 0; t--) {
			double ct = c[t];
			for (size_t i=0; i<num_states; i++) {
				double sum = 0.0;
				for (size_t j=0; j<num_states; j++) {
					sum += A(i, j) * B(j, seq[t+1]) * beta(t+1, j);
				}
				beta(t, i) = sum * ct;
			}	
		}
	}

	void HMM::baum_welch(const vector<Sequence>& sequences) {

		const size_t num_states = states.size();

		Matrix<double> ahat_numerator(num_states, num_states);
		Matrix<double> ahat_denominator(num_states, num_states);

		// Indexed first by state number, then by emission
		vector< SparseVector<const Emission*> > bhat_numerator(num_states);
		vector< double > bhat_denominator(num_states);

		for (size_t l=0; l<sequences.size(); l++) {
			const Sequence& seq = sequences[l];
			const size_t T = seq.size();
			Matrix<double> alpha;
			std::vector<double> c;
			Matrix<double> beta;

			this->forward_scaled(seq, alpha, c);
			this->backward_scaled(seq, alpha, c, beta);

			// Updates for ahat
			for (size_t i=0; i<num_states; i++) {
				const State* si = states[i];
				for (size_t j=0; j<num_states; j++) {
					const State* sj = states[j];
					double numerator_sum = 0.0;
					double demoninator_sum = 0.0;
					for (size_t t=0; t<(T - 1); t++) {
						numerator_sum += 
							alpha(t, i) * A(si, sj) * B(sj, seq[t+1]) * beta(t + 1, j);
						demoninator_sum +=
							alpha(t, i) * beta(t, i) / c[t];
					}

					ahat_numerator(i, j) += numerator_sum;
					ahat_denominator(i, j) += demoninator_sum;
				}
			}

			// Updates for bhat
			for (size_t j=0; j<num_states; j++) {
				SparseVector<const Emission*>& bn = bhat_numerator[j];
				double& bd = bhat_denominator[j];
				for (size_t t=0; t<T; t++) {
					const Emission* e = seq[t];
					bn[e] += alpha(t, j) * beta(t, j) / c[t];
					bd += alpha(t, j) * beta(t, j) / c[t];
				}
			}
		}

		// Calculate final a, b
		for (size_t i=0; i<num_states; i++) {
			State* si = states[i];
			for (size_t j=0; j<num_states; j++) {
				State* sj = states[j];

				double Aij = ahat_numerator(i, j) / ahat_denominator(i, j);
				si->transition_probs.set(sj, Aij);
			}
			si->transition_probs.normalize();

			SparseVector<const Emission*>& bn = bhat_numerator[i];
			double bd = bhat_denominator[i];
			si->emissions_probs.clear();
			BOOST_FOREACH(auto p, bn) {
				si->emissions_probs[p.first] = p.second / bd;
			}
			si->emissions_probs.normalize();
		}
	}
}