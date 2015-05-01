'use strict';

var assert = require('assert');

var blas = require('ndarray-blas-level1'),
    cwise = require('cwise');

module.exports = function modifiedGramSchmidtQR( A, R ) {

  var i,j, rii, vi, qi, vj, rij;

  assert(A.dimension === 2);

  var n = A.shape[0];
  //var m = A.shape[1];

  for( i=0; i<n; i++ ) {

    // vi = ai
    vi = A.pick( null, i );

    // rii = ||vi||
    rii = blas.nrm2( vi );
    if( rii===0 ) { return false; }
    R.set(i, i, rii);

    // qi = vi/rii
    qi = A.pick( null, i );
    blas.cpsc( 1/rii, vi, qi );

    for( j=i+1; j<n; j++ ) {
      //rij = qi' * vj
      vj = A.pick( null, j );
      rij = blas.dot( qi, vj );
      R.set( i, j, rij );

      // vj = vj - rij * qi
      blas.axpy( -rij, qi, vj );
    }
  }

  return true;
};
