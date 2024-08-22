.. py:currentmodule:: lsst.meas.extensions.multiprofit

.. _lsst.meas.extensions.multiprofit:

################################
lsst.meas.extensions.multiprofit
################################

:py:mod:`lsst.meas.extensions.multiprofit` implements separate tasks for PSF and source model fitting using :py:mod:`lsst.multiprofit`.
The tasks depend on more generic :py:class:`lsst.pipe.tasks.PipelineTask` classes defined in :py:mod:`lsst.pipe.tasks.fit_coadd_psf` and :py:mod:`lsst.pipe.tasks.fit_coadd_multiband`.
Both of the PSF and source model-fitting tasks operate row-by-row on single deblended objects from direct coadds.
Tickets such as `DM-42968 <https://rubinobs.atlassian.net/browse/DM-42968>`_ may add additional functionality like multi-object fitting and initializing models from previous fit parameters.

Because these tasks produce per-patch Arrow/Astropy tables, a generic task to consolidate the results into a single multiband tract-based table is provided in :py:mod:`lsst.meas.extensions.multiprofit.consolidate_astropy_table`.
It is anticipated that some of this functionality will be folded into the tasks producing objectTable_tract outputs in :py:mod:`lsst.pipe.tasks.postprocess`.

.. _lsst.meas.extensions.multiprofit-using:

Using lsst.meas.extensions.multiprofit
======================================

Besides the tasks classes themselves, various conveniently configured subclasses are provided in
:py:mod:`lsst.meas.extensions.multiprofit.pipetasks_fit`,
:py:mod:`lsst.meas.extensions.multiprofit.pipetasks_match`, and
:py:mod:`lsst.meas.extensions.multiprofit.pipetasks_merge`.

.. note::

    While use of these classes is not mandatory, it is recommended since they are standard configurations used in DRP pipelines and, amongst other things, set consistent dataset type names for commonly-used models.

To analyze the outputs, :py:mod:`lsst.meas.extensions.multiprofit.analysis_tools` provides subclasses of
:py:mod:`lsst.analysis.tools` tools for producing plots and metrics.

:py:mod:`lsst.meas.extensions.multiprofit.plots` provides functions and classes for visualizing individual
blends.
:py:mod:`lsst.meas.extensions.multiprofit.rebuild_coadd_multiband` expands on this functionality with
classes that can load and plot images from an entire patch while overlaying relevant model parameters.

.. toctree linking to topics related to using the module's APIs.

.. .. toctree::
..    :maxdepth: 1

.. _lsst.meas.extensions.multiprofit-contributing:

Contributing
============

``lsst.meas.extensions.multiprofit`` is developed at https://github.com/lsst/meas_extensions_multiprofit.
You can find Jira issues for this module under the `meas_extensions_multiprofit <https://rubinobs.atlassian.net/issues/?jql=component%20%3D%20meas_extensions_multiprofit>`_ component.

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
