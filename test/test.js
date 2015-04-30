'use strict';

var qr = require('../gram-schmidt.js'),
    assert = require('chai').assert,
    ndarray = require('ndarray'),
    pool = require('ndarray-scratch'),
    gemm = require("ndgemm"),
    ops = require('ndarray-ops');


describe("Gram-Schmidt QR", function() {

  it('computes the in-place QR factorization of a square matrix',function() {
    var i,j;
    var n=3, m=3;
    var A = ndarray(new Float64Array([1,2,3,4.5,5,6,7,8,3]), [n,m]);
    var Q = pool.zeros( A.shape, A.dtype );
    var R = pool.zeros( A.shape, A.dtype );
    var QR = pool.zeros( A.shape, A.dtype );
    var diff = pool.zeros( A.shape, A.dtype );

    var success = qr(A, R);

    assert(success);

    // Confirm that all sub-diagonal entries are close to zero:
    for(i=1; i<n; i++) {
      for(j=0; j<i; j++) {
        assert.closeTo( R.get(i,j), 0, 1e-8 );
      }
    }

    // Confirm that the matrices are basically equal:
    gemm( QR, A, R ); 
    A = ndarray(new Float64Array([1,2,3,4.5,5,6,7,8,3]), [n,m]); // <-- the original has been overwritten, so reassign
    ops.sub( diff, QR, A );
    var err2 = ops.norm2( diff ); // <-- not to be confused with the 2-norm of a matrix, but it's basically good enough
    assert.closeTo( err2, 0, 1e-8 );

  });

});
