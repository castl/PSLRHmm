#ifndef __pslrhmm_matrix_hpp__
#define __pslrhmm_matrix_hpp__

#include <map>
#include <cassert>
#include <boost/foreach.hpp>
#include <boost/random/uniform_01.hpp>

namespace pslrhmm {

	template<typename K>
	class Matrix {
		typedef std::vector<K> Vector;
		Vector vec;
		size_t X, Y;

	public:
		class linear_slice {
			friend class Matrix;
			K* data;
			size_t size;

			linear_slice(Matrix& m, size_t start, size_t end) {
				assert(start <= m.vec.size());
				assert(end <= m.vec.size());
				this->data = &m.vec.data()[start];
				this->size = end - start;
			 }

		public:
			K sum() {
				K r = K();
				for (size_t i=0; i<size; i++) {
					r += data[i];
				}
				return r;
			}

			void operator*=(K k) {
				for (size_t i=0; i<size; i++) {
					data[i] *= k;
				}
			}
		};

		Matrix() : X(0), Y(0) { }
		Matrix(size_t X, size_t Y, K k = K()) {
			resize(X, Y, k);
		}

		void resize(size_t X, size_t Y, K k = K()) {
			this->X = X;
			this->Y = Y;
			vec.clear();
			vec.resize(X*Y, k);
		}

		linear_slice xSlice(size_t x) {
			return linear_slice(*this, x*Y, x*Y + Y);
		}

		K& operator()(size_t x, size_t y) {
			assert(x < X);
			assert(y < Y);
			size_t idx = x*Y + y;
			assert(idx < vec.size());
			return vec[idx];
		}

		const K& operator()(size_t x, size_t y) const {
			assert(x < X);
			assert(y < Y);
			size_t idx = x*Y + y;
			assert(idx < vec.size());
			return vec[idx];
		}

		typename Vector::iterator begin() {
			return vec.begin();
		}

		typename Vector::iterator end() {
			return vec.end();
		}
	};
}

#endif // __pslrhmm_matrix_hpp__