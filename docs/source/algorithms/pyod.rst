PyOD (experimental)
============

`PyOD <https://github.com/yzhao062/pyod>`_ is a package that implements a variety of out-of-distribution detection methods.  
We integrate these methods into YRC, combining them with the adaptive threshold selection procedure described in :doc:`Logit Methods <logit>`.

.. note::

   We currently maintain a static clone of the PyOD codebase and modify it for better customization.  
   Future changes to the official PyOD repository will not be reflected in our codebase.  

.. note::

   We have not tested all PyOD methods, so it is possible that some methods may fail.
   For users who wish to use these methods, we recommend working directly with our raw GitHub codebase to be able to make appropriate changes.
