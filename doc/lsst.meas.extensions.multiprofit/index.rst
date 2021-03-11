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

There is currently one master task, which is a subtask of a generic multiband
fitting ``lsst.pipe.tasks.PipelineTask`` (``lsst.pipe.tasks.fit_multiband.MultibandFitTask``); future tasks will likely follow this pattern.

.. toctree linking to topics related to using the module's APIs.

.. .. toctree::
..    :maxdepth: 1

.. _lsst.meas.extensions.multiprofit-contributing:

Contributing
============

``lsst.meas.extensions.multiprofit`` is developed at https://github.com/lsst-dm/meas_extensions_multiprofit.
You can find Jira issues for this module under the `meas_extensions_multiprofit <https://jira.lsstcorp.org/issues/?jql=project%20%3D%20DM%20AND%20component%20%3D%20meas_extensions_multiprofit>`_ component.

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

.. automodapi:: lsst.meas.extensions.multiprofit
   :no-main-docstr:
   :no-inheritance-diagram:
