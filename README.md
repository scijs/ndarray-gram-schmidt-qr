# ndarray-gram-schmidt

[![Build Status](https://travis-ci.org/rreusser/ndarray-gram-schmidt.svg?branch=master)](https://travis-ci.org/rreusser/ndarray-gram-schmidt)

A module for calculating the in-place [QR decomposition of a matrix](http://en.wikipedia.org/wiki/QR_decomposition)

## Introduction

The algorithm is the numerically stable variant of the Gram-Schmidt QR decomposition as found on p. 58 of Trefethen and Bau's [Numerical Linear Algebra](http://www.amazon.com/Numerical-Linear-Algebra-Lloyd-Trefethen/dp/0898713617). In pseudocode, the algorithm is:

```
for i = 1 to n
  v_i = a_i

for i = 1 to n
  r_ii = ||v_i||
  q_i = v_i / r_ii

  for j = i+1 to n
    r_ij = q_i' * v_j
    v_j = v_j - r_ij * q_i
```

Currently only real number matrices are supported and only square matrices are tested.

## Usage

The algorithm currently only calculates the in-place QR decomposition and returns true on successful completion.

```
var qr = require('ndarray-gram-schmidt'),
    pool = require('ndarray-scratch');

var A = ndarray( new Float64Array([1,2,7,4,5,1,7,4,9]), [3,3] );
var R = pool.zeros( A.shape, A.dtype );

qr( A, R );
```

Then the product A * R is approximately equal to the original matrix.

## Credits
(c) 2015 Ricky Reusser. MIT License
