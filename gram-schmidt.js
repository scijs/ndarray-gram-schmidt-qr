'use strict';

var assert = require('assert');

var ndarray = require('ndarray'),
    ops = require('ndarray-ops'),
    cwise = require('cwise');

var nddot = cwise({
  args:["array", "array"],
  pre: function() {
    this.sum = 0;
  },
  body: function(a,b) {
    this.sum += a * b;
  },
  post: function() {
    return this.sum;
  }
});

var ndaxpy = cwise({
  args:["scalar", "array", "array"],
  body: function(alpha, x, y) {
    y += alpha * x;
  }
});

module.exports = function modifiedGramSchmidtQR( A, R ) {

  var i,j, rii, vi, qi, vj, rij;

  assert(A.dimension == 2);

  var n = A.shape[0];
  var m = A.shape[1];

  for( i=0; i<n; i++ ) {

    // vi = ai
    vi = A.pick( null, i );

    // rii = ||vi||
    rii = ops.norm2( vi );
    if( rii===0 ) { return false; } // Bail if rii == 0
    R.set(i, i, rii);

    // qi = vi/rii
    qi = A.pick( null, i );
    ops.assign( qi, vi );
    ops.divseq( qi, rii );

    for( j=i+1; j<n; j++ ) {
      //rij = qi' * vj
      vj = A.pick( null, j );
      rij = nddot( qi, vj );
      R.set( i, j, rij );

      // vj = vj - rij * qi
      ndaxpy( -rij, qi, vj );
    }
  }

  return true;
};
