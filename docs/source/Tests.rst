Tests
=====

Our continuous integration (CI) tests are implemented using GitHub Actions. This 
involves two main tests for consistency and validation:

1. Consistency of linear fits. Here we check that our simple examples
   :code:`examples/Ta_Linear_JCP2014` produces the same fitting coefficients 
   from a previous standard in :code:`20May21_Standard`.

2. Proper implementation of neural network forces. This test involves 
   calculating forces via finite difference and comparing to the analytical 
   forces calculated via automatic differentation that we fit to. This ensures 
   that neural network forces are properly coded. Finite difference 
   and analytical forces in an interatomic potential should agree within a 
   small amount such as :code:`1e-3` force units.