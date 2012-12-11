#ifndef __pslrhmm_sparse_vector_hpp__
#define __pslrhmm_sparse_vector_hpp__

#include <map>
#include <cassert>
#include <boost/foreach.hpp>
#include <boost/random/uniform_01.hpp>

namespace pslrhmm {

	template<typename K>
	class SparseVector {
		typedef std::map<K, double> Map;
		Map vec;

	public:
		void normalize() {
			double total = 0.0;
			BOOST_FOREACH(auto p, vec) {
				total += p.second;
			}

			BOOST_FOREACH(auto& p, vec) {
				p.second /= total;
			}
		}

		// Select an item at random based on the this vector's discrete distribution
		template<typename Random>
		K select(Random& r) const {
			boost::random::uniform_01<> u;
			double v = u(r);
			BOOST_FOREACH(auto p, vec) {
				v -= p.second;
				if (v <= 0.0)
					return p.first;
			}
			assert(false && "Internal error: selection math didn't work");
		}

		void clear() {
			vec.clear();
		}

		double& operator[](const K k) {
			return vec[k];
		}

		double operator[](const K k) const {
			auto f = vec.find(k);
			if (f == vec.end())
				return 0.0;
			return f->second;
		}

		double get(const K k) const {
			auto f = vec.find(k);
			if (f == vec.end())
				return 0.0;
			return f->second;
		}

		typename Map::iterator begin() {
			return vec.begin();
		}

		typename Map::iterator end() {
			return vec.end();
		}
	};
}

#endif // __pslrhmm_sparse_vector_hpp__