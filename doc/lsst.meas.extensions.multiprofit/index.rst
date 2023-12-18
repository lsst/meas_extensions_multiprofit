.. py:currentmodule:: lsst.meas.extensions.multiprofit

.. _lsst.meas.extensions.multiprofit:

################################
lsst.meas.extensions.multiprofit
################################

.. Tasks for running the MultiProFit astronomical source modelling code on
.. multiband exposures/catalogs.

.. .. _lsst.meas.extensions.multiprofit-using:

.. Using lsst.meas.extensions.multiprofit
.. ======================================

``lsst.meas.extensions.multiprofit`` implements separate tasks for PSF and source model fitting
using ``lsst.multiprofit``. The tasks depend on more generic ``lsst.pipe.tasks.PipelineTask``
classes defined in ``lsst.pipe.tasks.fit_coadd_psf`` and ``lsst.pipe.tasks.fit_coadd_multiband``.
Note that as implied by the names, the PSF fitting task is single-band only.

Because these tasks produce patch-based Arrow/Astropy tables, a generic task to consolidate
the results into a single multiband tract-based table is also provided. It is anticipated that
this functionality will be folded into the tasks producing objectTable_tract instances in
``lsst.pipe.tasks.postprocess``.

.. toctree linking to topics related to using the module's APIs.

.. .. toctree::
..    :maxdepth: 1

.. _lsst.meas.extensions.multiprofit-contributing:

Contributing
============

``lsst.meas.extensions.multiprofit`` is developed at https://github.com/lsst-dm/meas_extensions_multiprofit.
You can find Jira issues for multiprofit through `search <https://jira.lsstcorp.org/issues/?jql=text%20~%20%22multiprofit%22>`_.

.. If there are topics related to developing this module (rather than using it), link to this from a toctree placed here.

.. .. toctree::
..    :maxdepth: 1

.. _lsst.meas.extensions.multiprofit-command-line-taskref:

Task reference
==============

.. _lsst.meas.extensions.multiprofit-pipeline-tasks:

Pipeline tasks
--------------

.. lsst-pipelinetasks::
   :root: lsst.meas.extensions.multiprofit

.. _lsst.meas.extensions.multiprofit-command-line-tasks:

Command-line tasks
------------------

.. lsst-cmdlinetasks::
   :root: lsst.meas.extensions.multiprofit

.. _lsst.meas.extensions.multiprofit-tasks:

Tasks
-----

.. lsst-tasks::
   :root: lsst.meas.extensions.multiprofit
   :toctree: tasks

.. _lsst.meas.extensions.multiprofit-configs:

Configurations
--------------

.. lsst-configs::
   :root: lsst.meas.extensions.multiprofit
   :toctree: configs

.. .. _lsst.meas.extensions.multiprofit-scripts:

.. Script reference
.. ================

.. .. TODO: Add an item to this toctree for each script reference topic in the scripts subdirectory.

.. .. toctree::
..    :maxdepth: 1

.. .. _lsst.meas.extensions.multiprofit-pyapi:

Python API reference
====================

.. automodapi:: lsst.meas.extensions.multiprofit.consolidate_astropy_table
   :no-inheritance-diagram:

.. automodapi:: lsst.meas.extensions.multiprofit.fit_coadd_multiband
   :no-inheritance-diagram:

.. automodapi:: lsst.meas.extensions.multiprofit.fit_coadd_psf
   :no-inheritance-diagram:
